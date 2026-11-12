"""Differential back-substitution: per-element bounds on rounding drift (`method="backsub"`).

The per-tensor interval analysis in `absint.py` is sound but its bounds grow with every layer:
one interval per tensor, and errors pushed forward through |W|, which assumes the errors of all
inputs of a neuron line up in sign. Following the differential-verification idea of NeuroDiff
(Paulsen, Wang, Wang, ASE 2020), this module tracks the *difference* symbolically:

    out_1 - out_2 = sum over units k of  J_k . d_k                                 (exact)

where d_k is the local difference that unit k injects (the rounding of program 1 minus the
rounding of program 2 inside the unit) and J_k is the mean-value Jacobian of the rest of the
network between the two programs' trajectories. For piecewise-linear networks this identity has
no remainder: relu(a) - relu(b) = s (a - b) with s in [0, 1], and max over a window is a convex
combination. For smooth operations s lies in the range of the derivative over the segment.
So with J_k enclosed in an interval Jacobian,

    |out_1 - out_2|_i  <=  sum_k  sum_e  |J_k[i, e]| |d_k[e]|.

`J_k` is applied by back-substitution (a vector-Jacobian product per output row, as in
DeepPoly/CROWN). The weights keep their signs, so cancellation across layers is kept; only
neurons whose sign may differ between the trajectories get an interval slope.

Every tensor carries five per-element bound kinds:

    V    input-domain variation: |F(x) - F(center)|       (0 for a point domain)
    E1   program 1 versus exact arithmetic
    E2   program 2 versus exact arithmetic
    D    program 1 versus program 2 (the certified quantity)
    R64  the float64 reference used for centers versus exact arithmetic

Ranges for slopes and rounding magnitudes are center +- (V + R64 + max(E1, E2)), per element.
Intermediate bounds come from a cheap forward rule (|J| applied forward) and are tightened by
back-substitution where the forward rule is loose; outputs are always back-substituted. The
local injections reuse the order models of `absint` / `fpmodel`: known trees (Strategy 1, with
exact cancellation of identical trees) or every admissible tree (Strategy 2), per element.

Analysis arithmetic: centers and bounds are float64. Back-substitution coefficients are float32
for speed, with an explicit relative inflation `tau` at every accumulation step; set
`analysis_dtype="float64"` for a run without that approximation.
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import fpmodel as fpm
from .absint import (CORRECTLY_ROUNDED, DSL_BACKENDS, EXACT_OPS, GELU_LIP, GELU_MIN,
                     INV_SQRT2, SILU_LIP, SILU_MIN, SQRT_2_OVER_PI, SQRT_PI, SWAPPED_SOURCE,
                     InconclusiveError, OrderOracle, _MatchState, _base_op, _exceptional,
                     _unsupported_op, _unsupported_path)
from .outcomes import Inconclusive

KINDS = ("V", "E1", "E2", "D", "R64")
KV, KE1, KE2, KD, K64 = range(5)
MIXING = {"matmul", "conv", "reduce_sum", "reduce_mean", "avg_pool", "softmax", "log_softmax",
          "layer_norm", "rms_norm"}
GELU_D2 = 0.7980          # sup |GELU''| (both forms): 0.797885
SILU_D2 = 0.5001          # sup |SiLU''| = 0.5
EXP_UNDERFLOW = -104.0    # exp(x) < 2^-150 for x below this: rounds to 0 or a subnormal


def _torch():
    import torch
    return torch


# --------------------------------------------------------------------------------------------
# Interval helpers (center-radius and lo-hi)
# --------------------------------------------------------------------------------------------

def _mag(lo, hi):
    return lo.abs().maximum(hi.abs())


def _min_abs(lo, hi):
    torch = _torch()
    return torch.where((lo <= 0) & (hi >= 0), torch.zeros_like(lo), lo.abs().minimum(hi.abs()))


def _imul(cm, cr, im, ir):
    """(cm +- cr) * (im +- ir) as center +- radius."""
    return cm * im, cm.abs() * ir + cr * (im.abs() + ir)


def _iv_mul(al, ah, bl, bh):
    torch = _torch()
    c = torch.stack([al * bl, al * bh, ah * bl, ah * bh])
    c = torch.nan_to_num(c, nan=0.0)
    return c.min(0).values, c.max(0).values


def _sum_to(x, shape, lead):
    """Reduce a [lead..., *broadcast(shape)] tensor to [lead..., *shape]."""
    nl = len(lead)
    full = x.shape[nl:]
    target = tuple(lead) + (1,) * (len(full) - len(shape)) + tuple(shape)
    return x.sum_to_size(target).reshape(tuple(lead) + tuple(shape))


# --------------------------------------------------------------------------------------------
# The analyzer
# --------------------------------------------------------------------------------------------

class BacksubAnalyzer:
    def __init__(self, match, input_domain, config, example_inputs):
        torch = _torch()
        self.m = match
        self.cfg = config
        self.domain = input_domain
        self.examples = example_inputs
        self.dev = torch.device("cuda" if any(getattr(x, "is_cuda", False)
                                              for x in example_inputs) else "cpu")
        self.cdt = torch.float64 if getattr(config, "analysis_dtype", "float32") == "float64" \
            else torch.float32
        self.tau = 0.0 if self.cdt == torch.float64 else 2.0 ** -12
        self.tol = float(getattr(config, "backsub_tol", 1e-2))
        self.max_rows = int(getattr(config, "backsub_batch", 0) or 0)
        self.window = int(getattr(config, "backsub_window", 0) or 0)
        self.prob = getattr(config, "guarantee", "formal") == "probabilistic"
        self.exact_ref = getattr(config, "reference", "program_1") == "exact"
        self.lam = float(getattr(config, "confidence", 1e-6))
        self.kappa = 1.0
        self.st = _MatchState(match)
        self.assumptions = set()
        self.oracle = OrderOracle(config, self.assumptions)
        self.k = match.canonical_1
        self.c0: Dict[str, object] = {}
        self.rng: Dict[str, tuple] = {}
        self.bnd: Dict[str, object] = {}
        self.inj: Dict[str, object] = {}
        self.ibnd: Dict[str, object] = {}       # forward bounds of unit-internal values
        self._jcache: Dict[str, dict] = {}
        self._res: Dict[str, object] = {}
        import collections
        self._contrib = collections.Counter()
        self.cut: Dict[str, bool] = {}
        self.unit_of: Dict[str, object] = {}
        self.reports: List[dict] = []
        self.stats = {"backsub_rows": 0, "backsub_seconds": 0.0}

    # ---- configuration ----

    def _ulp(self, name):
        table = dict(fpm.CUDA_FP32_ULP)
        table.update(getattr(self.cfg, "elementary_ulp", None) or {})
        if name not in table:
            raise _unsupported_op(f"no accuracy assumption for {name}")
        self.assumptions.add(f"{name} accurate to {table[name]} ulp")
        return float(table[name])

    def T(self, x, dtype=None):
        torch = _torch()
        return torch.as_tensor(np.asarray(x), device=self.dev).to(dtype or torch.float64)

    # ---- entry point ----

    def run(self):
        torch = _torch()
        t0 = time.time()
        self._seed_inputs()
        nstat = 1
        mix = 0
        self.mix_idx = {}
        for u in self.m.units:
            if any(self.k.nodes[n].op in MIXING for n in u.canonical_nodes_1):
                mix += 1
            self.mix_idx[u.id] = mix
            for b in u.outputs:
                self.unit_of[b.value_1] = u
                nstat += int(np.prod(self.k.values[b.value_1].shape) or 1)
        if self.prob:
            # Hoeffding for each bounded statement, union bound over all of them
            self.kappa = math.sqrt(2 * math.log(2 * nstat / self.lam))
            self.assumptions.add(
                "PROBABILISTIC: rounding errors are independent, mean-zero and bounded "
                "(Higham & Mary 2019 model); all bounds hold jointly with probability >= "
                f"1 - {self.lam:g} (kappa = {self.kappa:.2f} over {nstat} statements)")
        out_vals = {b.value_1 for b, _ in self.m.output_pairs}
        with torch.no_grad():
            for u in self.m.units:
                self._unit(u, out_vals)
        triples = []
        for b, _ in self.m.output_pairs:
            v = b.value_1
            self._ensure(v)
            B = self.bnd[v]
            plo, phi = self.prog_range(v)
            lo = float(plo.min())
            hi = float(phi.max())
            d = float(B[KD].max())
            triples.append((lo, hi, d))
        self.stats["seconds"] = round(time.time() - t0, 2)
        self.stats["last_backsub_D_sources"] = {k_: float(f"{v:.3g}") for k_, v in
                                                self._contrib.most_common(6)}
        self.assumptions.add("IEEE 754 binary32/64, round to nearest even, gradual underflow")
        if self.exact_ref:
            self.assumptions.add("reference = exact real arithmetic on program 1's graph "
                                 "(reference GELU/erf/tanh definitions); the bound is "
                                 "|program 2 - exact|, and program 1's own rounding is not "
                                 "part of it")
        if self.cdt != torch.float64:
            self.assumptions.add(f"back-substitution in float32 with relative inflation "
                                 f"{self.tau:g} per accumulation (analysis arithmetic)")
        return triples

    # ---- seeds ----

    def _seed_inputs(self):
        torch = _torch()
        k = self.k
        for b in self.m.input_pairs:
            v = k.values[b.value_1]
            idx = v.input_index
            ex = self.examples[idx]
            shape = tuple(v.shape)
            lo = torch.as_tensor(self.domain.lower[idx], dtype=torch.float64).to(self.dev)
            hi = torch.as_tensor(self.domain.upper[idx], dtype=torch.float64).to(self.dev)
            lo, hi = lo.expand(shape).clone(), hi.expand(shape).clone()
            if np.dtype(v.dtype).kind == "f":
                c = (lo + hi) / 2
                B = torch.zeros((5,) + shape, dtype=torch.float64, device=self.dev)
                B[KV] = (hi - lo) / 2
                self.c0[b.value_1] = c
                self.bnd[b.value_1] = B
                self.rng[b.value_1] = (lo, hi)
            else:
                if not torch.equal(lo, hi):
                    # integer inputs over a range: only usable through gather (as a source)
                    self.c0[b.value_1] = torch.floor((lo + hi) / 2).long()
                    self.rng[b.value_1] = (lo, hi)
                    self.cut[b.value_1] = True
                else:
                    self.c0[b.value_1] = lo.long() if np.dtype(v.dtype).kind != "b" else lo.bool()
                    self.rng[b.value_1] = (lo, hi)
                self.bnd[b.value_1] = None

    def _const(self, vid):
        torch = _torch()
        if vid in self.c0:
            return
        v = self.k.values[vid] if vid in self.k.values else None
        self._const_value(v, vid, self.c0, self.rng, self.bnd)

    def _const_value(self, v, vid, c0, rng, bnd):
        torch = _torch()
        d = np.asarray(v.data)
        kind = np.dtype(v.dtype).kind
        if kind == "f":
            x = torch.as_tensor(d.astype(np.float64), device=self.dev)
            if v.exact is not None:
                x = torch.full_like(x, float(v.exact))
            c0[vid] = x
            rng[vid] = (x, x)
            B = torch.zeros((5,) + tuple(x.shape), dtype=torch.float64, device=self.dev)
            if v.exact is not None:
                B[K64] = fpm.FORMATS["float64"].u * x.abs()
            bnd[vid] = B
        else:
            x = torch.as_tensor(d, device=self.dev)
            c0[vid] = x.bool() if kind == "b" else x.long()
            rng[vid] = (x.double(), x.double())
            bnd[vid] = None

    def _ensure(self, vid):
        if vid not in self.c0:
            v = self.k.values[vid]
            if v.kind == "const":
                self._const(vid)
            else:
                raise _unsupported_op(f"value {vid} was never computed")

    def prog_range(self, vid):
        """Range enclosing the exact value and both programs' values (per element)."""
        B = self.bnd.get(vid)
        if B is None:
            return self.rng[vid]
        c = self.c0[vid]
        e = B[KE1].maximum(B[KE2])
        s = B[KV] + B[K64] + e
        lo, hi = c - s, c + s
        r = self.rng.get(vid)
        if r is not None and vid in self.unit_of:
            # interval range of the exact function, widened by the program errors
            lo = lo.maximum(r[0] - e - B[K64])
            hi = hi.minimum(r[1] + e + B[K64])
        return lo, hi

    # ---- one unit ----

    def _unit(self, u, out_vals):
        torch = _torch()
        k = self.k
        for b in u.inputs:
            self._ensure(b.value_1)
        # centers and ranges of the exact unit function (internal values: IBP)
        for nid in u.canonical_nodes_1:
            n = k.nodes[nid]
            for i in n.inputs:
                if i is not None:
                    self._ensure(i)
            ins_c = [self.c0[i] if i is not None else None for i in n.inputs]
            ins_r = [self._range_any(i) if i is not None else None for i in n.inputs]
            c = self.eval64(k, n, ins_c)
            self.c0[n.outputs[0]] = c
            self.rng[n.outputs[0]] = self.ibp(k, n, ins_r, ins_c, c)
            if n.op == "gather" and self._gather_is_source(k, n):
                self.cut[n.outputs[0]] = True
        # local injections at the unit outputs
        t_inj = time.time()
        inj = self._injections(u)
        self.stats["inject_seconds"] = self.stats.get("inject_seconds", 0.0) + time.time() - t_inj
        for b in u.outputs:
            self.inj[b.value_1] = inj[b.value_1]
        # forward bounds through the unit
        fwd = self._forward_unit(u, inj)
        mixing = any(k.nodes[n].op in MIXING for n in u.canonical_nodes_1)
        rep = {"unit": u.id, "ops_1": list(u.ops_1), "ops_2": list(u.ops_2), "outputs": []}
        for b in u.outputs:
            v = b.value_1
            if fwd.get(v) is None:
                self.bnd[v] = None
                continue
            B = fwd[v]
            self.bnd[v] = B
            if self.cut.get(v):
                continue
            if mixing or v in out_vals:
                B = self._tighten(v, B, final=v in out_vals)
                self.bnd[v] = B
            c = self.c0[v].double()
            rep["outputs"].append({"value": v, "D": float(B[KD].max()),
                                   "E1": float(B[KE1].max()), "E2": float(B[KE2].max()),
                                   "V": float(B[KV].max()), "R64": float(B[K64].max()),
                                   "inj_D": float(inj[v][KD].max()),
                                   "center_mag": float(c.abs().max())})
        self.reports.append(rep)

    def _range_any(self, vid):
        """Range of the exact value and both programs' values: from the bounds where a value
        has them (inputs, constants, unit outputs), else its interval (unit internals)."""
        if self.bnd.get(vid) is not None:
            return self.prog_range(vid)
        return self.rng[vid]

    def _gather_is_source(self, k, n):
        idx = n.inputs[1]
        return self.cut.get(idx, False)

    # ---- forward rule (cheap, |J| forward) ----

    def _forward_unit(self, u, inj):
        torch = _torch()
        k = self.k
        fb = {}
        for b in u.inputs:
            fb[b.value_1] = self.bnd.get(b.value_1)
        for nid in u.canonical_nodes_1:
            n = k.nodes[nid]
            ins = []
            for i in n.inputs:
                if i is None:
                    ins.append(None)
                    continue
                if i not in fb:
                    self._ensure(i)
                    fb[i] = self.bnd.get(i)
                ins.append(fb[i])
            o = n.outputs[0]
            if np.dtype(k.values[o].dtype).kind != "f":
                self._check_discrete(k, n, ins)
                fb[o] = None
                continue
            if self.cut.get(o):
                lo, hi = self.rng[o]
                B = torch.zeros((5,) + tuple(lo.shape), dtype=torch.float64, device=self.dev)
                B[KV] = (hi - lo) / 2
                self.c0[o] = (lo + hi) / 2
                fb[o] = B
                continue
            ranges = [self._range_any(i) if i is not None else None for i in n.inputs]
            cvals = [self.c0.get(i) if i is not None else None for i in n.inputs]
            fb[o] = self.fwd(k, n, ins, ranges, self.rng[o], cvals)
        outs = {b.value_1 for b in u.outputs}
        for v, B in fb.items():
            if v not in outs and B is not None and v not in self.bnd:
                self.ibnd[v] = B
        out = {}
        for b in u.outputs:
            v = b.value_1
            if fb.get(v) is None:
                out[v] = None
                continue
            out[v] = fb[v] + self.kappa * inj[v]
        return out

    def _check_discrete(self, k, n, ins):
        """A boolean or integer result must be the same in both programs and over the
        domain (it selects branches and indices)."""
        torch = _torch()
        if n.op in ("lt", "le", "gt", "ge", "eq", "ne") and \
                all(i is not None and np.dtype(k.values[i].dtype).kind == "f" for i in n.inputs):
            (al, ah), (bl, bh) = [self._range_any(i) for i in n.inputs]
            fixed = (al == ah) & (bl == bh)
            decided = (ah < bl) | (bh < al) | fixed
            if not bool(decided.all()):
                raise _unsupported_path(f"comparison {n.op} of computed values may differ "
                                        "between the programs or over the domain")
            return
        for x, i in zip(ins, n.inputs):
            if i is None:
                continue
            if x is not None and bool((x > 0).any()):
                raise _unsupported_path(f"{n.op}: a boolean or integer result may differ "
                                        "between the programs or over the domain")
            lo, hi = self._range_any(i)
            if not torch.equal(lo, hi):
                raise _unsupported_path(f"{n.op}: a boolean or integer value varies over the "
                                        "domain")

    # ---- back-substitution ----

    def _tighten(self, vid, B, final):
        """Back-substitute rows of `vid` whose forward bound is loose; keep the min."""
        torch = _torch()
        c = self.c0[vid].double().abs()
        loose = (B[KV] + B[KE1].maximum(B[KE2])) > self.tol * c
        if final:
            loose = torch.ones_like(loose)
        rows = loose.reshape(-1).nonzero().reshape(-1)
        if rows.numel() == 0:
            return B
        self._contrib.clear()
        t0 = time.time()
        bs = self.backsub(vid, rows, window=None if final else (self.window or None))
        flat = B.reshape(5, -1).clone()
        flat[:, rows] = torch.minimum(flat[:, rows], bs)
        self.stats["backsub_rows"] += int(rows.numel())
        self.stats["backsub_seconds"] += time.time() - t0
        return flat.reshape(B.shape)

    def _ancestors(self, vid, window=None):
        """Nodes to back-substitute through, in reverse topological order. With a window,
        stop at outputs of units more than `window` mixing units upstream: their own
        (already computed) bounds are used there."""
        k = self.k
        lim = None
        if window:
            u = self.unit_of.get(vid)
            if u is not None:
                lim = self.mix_idx[u.id] - window
        self._stops = set()
        seen, stack = set(), [vid]
        while stack:
            v = stack.pop()
            if v in seen:
                continue
            seen.add(v)
            if self.cut.get(v):
                continue
            if lim is not None and v != vid and v in self.unit_of and \
                    self.mix_idx[self.unit_of[v].id] <= lim and self.bnd.get(v) is not None:
                self._stops.add(v)
                continue
            p = k.producer(v)
            if p is None:
                continue
            for i in p.inputs:
                if i is not None:
                    stack.append(i)
        pos = {n: i for i, n in enumerate(k.nodes)}
        nodes = sorted({k.values[v].producer for v in seen if k.values[v].producer is not None
                        and not self.cut.get(v) and v not in self._stops},
                       key=pos.get, reverse=True)
        return nodes, seen

    def backsub(self, vid, rows, window=None):
        """Bounds (5 kinds) for the given flat element indices of `vid`.

        Coefficients are point vectors (the midpoint Jacobian of every step). What a step's
        Jacobian range adds beyond its midpoint is bounded once per node by `residual(n)`,
        using the bounds already known at the node's inputs, so it is never multiplied by
        later steps."""
        torch = _torch()
        k = self.k
        nodes, seen = self._ancestors(vid, window)
        shape = tuple(k.values[vid].shape)
        numel = int(np.prod(shape)) if shape else 1
        biggest = max([int(np.prod(k.values[v].shape) or 1) for v in seen] + [numel])
        budget = self.max_rows or max(1, int(float(getattr(self.cfg, "backsub_memory", 4e9))
                                              // (biggest * 4 * 6)))
        out = torch.zeros((5, rows.numel()), dtype=torch.float64, device=self.dev)
        for s in range(0, rows.numel(), budget):
            r = rows[s:s + budget]
            Bn = r.numel()
            c = torch.zeros((Bn, numel), dtype=self.cdt, device=self.dev)
            c[torch.arange(Bn, device=self.dev), r] = 1
            coef = {vid: c.reshape((Bn,) + shape)}
            acc = torch.zeros((5, Bn), dtype=torch.float64, device=self.dev)
            accv = torch.zeros((5, Bn), dtype=torch.float64, device=self.dev)
            for nid in nodes:
                n = k.nodes[nid]
                o = n.outputs[0]
                if o not in coef:
                    continue
                co = coef.pop(o)
                w = co.abs().reshape(Bn, -1).T
                I = self.inj.get(o)
                if I is not None:
                    if self.prob:
                        t_ = self._dot(I.reshape(5, -1) ** 2, w ** 2)
                        accv += t_
                        self._contrib["injection(var)"] += float(t_[KD].max())
                    else:
                        t_ = self._dot(I.reshape(5, -1), w)
                        acc += t_
                        self._contrib["injection"] += float(t_[KD].max())
                R = self.residual(k, n)
                if R is not None:
                    t_ = self._dot(R.reshape(5, -1), w)
                    acc += t_
                    self._contrib[f"residual:{n.op}"] += float(t_[KD].max())
                for i, gi in zip(n.inputs, self.pvjp(k, n, co)):
                    if i is None or gi is None:
                        continue
                    coef[i] = coef[i] + gi if i in coef else gi
            for v, cv in coef.items():                  # inputs, constants, cuts, window stops
                B = self._bounds_of(v)
                if B is not None:
                    acc += self._dot(B.reshape(5, -1), cv.abs().reshape(Bn, -1).T)
            if self.prob:
                self._contrib["kappa*sqrt(var)"] = float((self.kappa * accv[KD].sqrt()).max())
                acc = acc + self.kappa * torch.sqrt(accv)
            out[:, s:s + Bn] = acc * (1 + 1e-6)
        return out

    def _dot(self, A, w):
        """A [5, n] (float64, >= 0) times w [n, B] (>= 0) as an upper bound in float64.
        In float32 every product and partial sum of nonnegative terms is within gamma_n,
        so dividing by (1 - gamma_n) gives an upper bound."""
        torch = _torch()
        if w.dtype == torch.float64:
            return A @ w
        n = A.shape[1]
        g = 2 * n * 2.0 ** -24
        if g >= 0.5:
            return A @ w.double()
        hi = torch.finfo(torch.float32).max / max(n, 1)
        if bool((A > hi).any()):
            return A @ w.double()
        return (A.float() @ w).double() / (1 - g) * (1 + 2.0 ** -20)

    def _bounds_of(self, vid):
        B = self.bnd.get(vid)
        return B if B is not None else self.ibnd.get(vid)

    # ================================================================================ #
    # Op semantics                                                                     #
    # ================================================================================ #

    def eval64(self, g, n, ins):
        """float64 (or exact integer) value of the exact operation at the centers."""
        torch = _torch()
        F = torch.nn.functional
        op, a = n.op, n.attrs
        x = ins[0] if ins else None
        out_dt = g.values[n.outputs[0]].dtype
        if op == "opaque":
            raise _unsupported_op(f"opaque {a.get('source')}")
        if op == "add":
            r = ins[0]
            for y in ins[1:]:
                r = r + y
            return r
        if op == "mul":
            r = ins[0]
            for y in ins[1:]:
                r = r * y
            return r
        simple = {"sub": lambda: ins[0] - ins[1], "div": lambda: ins[0] / ins[1],
                  "max": lambda: torch.maximum(ins[0], ins[1]),
                  "min": lambda: torch.minimum(ins[0], ins[1]),
                  "eq": lambda: ins[0] == ins[1], "ne": lambda: ins[0] != ins[1],
                  "lt": lambda: ins[0] < ins[1], "le": lambda: ins[0] <= ins[1],
                  "gt": lambda: ins[0] > ins[1], "ge": lambda: ins[0] >= ins[1],
                  "and": lambda: ins[0] & ins[1], "or": lambda: ins[0] | ins[1],
                  "not": lambda: ~x, "relu": lambda: torch.relu(x), "neg": lambda: -x,
                  "abs": lambda: x.abs(), "exp": lambda: torch.exp(x),
                  "log": lambda: torch.log(x), "sqrt": lambda: torch.sqrt(x),
                  "rsqrt": lambda: torch.rsqrt(x), "reciprocal": lambda: 1 / x,
                  "tanh": lambda: torch.tanh(x), "sigmoid": lambda: torch.sigmoid(x),
                  "erf": lambda: torch.erf(x), "sin": lambda: torch.sin(x),
                  "cos": lambda: torch.cos(x), "silu": lambda: F.silu(x),
                  "pow": lambda: torch.pow(ins[0], ins[1]),
                  "where": lambda: torch.where(ins[0].bool(), ins[1], ins[2])}
        if op in simple:
            return simple[op]()
        if op == "gelu":
            return F.gelu(x, approximate="tanh" if a.get("approximate") == "tanh" else "none")
        if op == "reshape":
            return x.reshape(a["shape"])
        if op == "permute":
            return x.permute(*a["perm"])
        if op == "broadcast_to":
            return x.expand(a["shape"])
        if op == "slice":
            return self._slice(x, a["spec"])
        if op == "concat":
            return torch.cat(ins, dim=a["axis"])
        if op == "pad":
            return self._pad(x, a["pads"], float(ins[1].reshape(-1)[0]))
        if op == "gather":
            return self._gather(x, ins[1], a["axis"])
        if op == "cast":
            kind = np.dtype(a["to"]).kind
            if kind == "f":
                return x.double()
            if kind == "b":
                return x != 0
            return x.long() if not x.is_floating_point() else torch.trunc(x).long()
        if op == "matmul":
            return torch.matmul(x, ins[1])
        if op == "conv":
            return self._conv(x, ins[1], a)
        if op in ("reduce_sum", "reduce_mean", "reduce_max", "reduce_min"):
            axes = a.get("axes") or tuple(range(x.dim()))
            f = {"reduce_sum": torch.sum, "reduce_mean": torch.mean}.get(op)
            if f is not None:
                return f(x, dim=axes, keepdim=a.get("keepdim", False))
            r = x
            for ax in sorted(axes, reverse=True):
                r = (r.amax if op == "reduce_max" else r.amin)(dim=ax, keepdim=True)
            if not a.get("keepdim", False):
                r = r.squeeze(tuple(axes)) if axes else r
            return r
        if op == "softmax":
            return torch.softmax(x, dim=a["axis"])
        if op == "log_softmax":
            return torch.log_softmax(x, dim=a["axis"])
        if op == "layer_norm":
            axes = a["axes"]
            mu = x.mean(dim=axes, keepdim=True)
            var = ((x - mu) ** 2).mean(dim=axes, keepdim=True)
            y = (x - mu) / torch.sqrt(var + a["eps"])
            if len(ins) > 1 and ins[1] is not None:
                y = y * ins[1]
            if len(ins) > 2 and ins[2] is not None:
                y = y + ins[2]
            return y
        if op == "rms_norm":
            axes = a["axes"]
            y = x / torch.sqrt((x * x).mean(dim=axes, keepdim=True) + a["eps"])
            return y * ins[1] if len(ins) > 1 and ins[1] is not None else y
        if op == "batch_norm":
            w, b, mean, var = ins[1:5]
            shp = (1, -1) + (1,) * (x.dim() - 2)
            s = (w / torch.sqrt(var + a["eps"])).reshape(shp)
            return (x - mean.reshape(shp)) * s + b.reshape(shp)
        if op == "max_pool":
            p, k_, st, dil = self._pool_args(a)
            xp = F.pad(x, p, value=-math.inf)
            return F.max_pool2d(xp, k_, st, 0, dil, ceil_mode=a.get("ceil_mode", False))
        if op == "avg_pool":
            p, k_, st, _ = self._pool_args(a)
            if p[0] != p[1] or p[2] != p[3]:
                raise _unsupported_op("asymmetric avg_pool padding")
            return F.avg_pool2d(x, k_, st, (p[2], p[0]), ceil_mode=a.get("ceil_mode", False),
                                count_include_pad=a.get("count_include_pad", True))
        raise _unsupported_op(f"no evaluation rule for {op}")

    # ---- small evaluation helpers ----

    @staticmethod
    def _slice(x, spec):
        idx = [slice(None)] * x.dim()
        for axis, s, e, st in spec:
            idx[axis] = slice(s, e, st)
        return x[tuple(idx)]

    @staticmethod
    def _pad(x, pads, value):
        F = _torch().nn.functional
        r = x.dim()
        b, e = pads[:r], pads[r:]
        pp = []
        for d in reversed(range(r)):
            pp += [b[d], e[d]]
        return F.pad(x, pp, value=value)

    @staticmethod
    def _gather(x, idx, axis):
        torch = _torch()
        flat = idx.reshape(-1).long() % x.shape[axis]
        r = torch.index_select(x, axis, flat)
        return r.reshape(x.shape[:axis] + idx.shape + x.shape[axis + 1:])

    @staticmethod
    def _pool_args(a):
        k_ = tuple(a["kernel"])
        if len(k_) != 2:
            raise _unsupported_op("only 2-D pooling")
        st = tuple(a.get("stride") or k_)
        pads = tuple(a.get("pads") or (0, 0, 0, 0))
        dil = tuple(a.get("dilation") or (1, 1))
        # F.pad order: (w_begin, w_end, h_begin, h_end)
        return (pads[1], pads[3], pads[0], pads[2]), k_, st, dil

    def _conv(self, x, w, a, weight_abs=False):
        torch = _torch()
        F = torch.nn.functional
        nsp = x.dim() - 2
        pads = a["pads"]
        b, e = pads[:nsp], pads[nsp:]
        if tuple(b) != tuple(e):
            pp = []
            for d in reversed(range(nsp)):
                pp += [b[d], e[d]]
            x = F.pad(x, pp)
            pad = 0
        else:
            pad = tuple(b)
        f = F.conv2d if nsp == 2 else F.conv1d if nsp == 1 else None
        if f is None:
            raise _unsupported_op("conv of rank other than 1-D or 2-D")
        return f(x, w, None, a["stride"], pad, a["dilation"], a["groups"])

    # ================================================================================ #
    # IBP (ranges of the exact function over input ranges)                             #
    # ================================================================================ #

    def ibp(self, g, n, ins, ins_c, center):
        torch = _torch()
        op, a = n.op, n.attrs
        out = g.values[n.outputs[0]]
        if np.dtype(out.dtype).kind != "f":
            c = center.double()
            if all(r is None or torch.equal(r[0], r[1]) for r in ins):
                return c, c
            if op in ("reshape", "permute", "broadcast_to", "slice", "concat", "cast"):
                return self._apply_lin(g, n, [r[0] for r in ins], ins_c), \
                    self._apply_lin(g, n, [r[1] for r in ins], ins_c)
            return c.clone().fill_(-math.inf), c.clone().fill_(math.inf)
        lo = [r[0] if r is not None else None for r in ins]
        hi = [r[1] if r is not None else None for r in ins]
        if op in ("reshape", "permute", "broadcast_to", "slice", "cast", "relu", "max_pool",
                  "reduce_max", "reduce_min", "avg_pool", "reduce_mean", "reduce_sum",
                  "exp", "tanh", "sigmoid", "erf", "sqrt", "log"):
            f = lambda v: self.eval64(g, n, [v] + [c for c in ins_c[1:]])  # noqa: E731
            if op in ("sqrt", "log") and bool((lo[0] < 0).any() if op == "sqrt" else
                                              (lo[0] <= 0).any()):
                raise _exceptional(f"{op} of a range that reaches {'0' if op == 'log' else '<0'}")
            return f(lo[0]), f(hi[0])
        if op in ("concat",):
            return torch.cat(lo, dim=a["axis"]), torch.cat(hi, dim=a["axis"])
        if op == "pad":
            v = float(ins_c[1].reshape(-1)[0])
            return self._pad(lo[0], a["pads"], v), self._pad(hi[0], a["pads"], v)
        if op == "gather":
            if torch.equal(ins[1][0], ins[1][1]):
                return self._gather(lo[0], ins_c[1], a["axis"]), \
                    self._gather(hi[0], ins_c[1], a["axis"])
            # index range: hull of the table rows it can select
            il, ih = ins[1]
            tab_lo, tab_hi = lo[0], hi[0]
            ax = a["axis"]
            res_lo, res_hi = [], []
            for l_, h_ in zip(il.reshape(-1).tolist(), ih.reshape(-1).tolist()):
                sel = torch.arange(int(l_), int(h_) + 1, device=self.dev)
                res_lo.append(torch.index_select(tab_lo, ax, sel).amin(dim=ax, keepdim=True))
                res_hi.append(torch.index_select(tab_hi, ax, sel).amax(dim=ax, keepdim=True))
            L_ = torch.cat(res_lo, dim=ax)
            H_ = torch.cat(res_hi, dim=ax)
            shp = tab_lo.shape[:ax] + il.shape + tab_lo.shape[ax + 1:]
            return L_.reshape(shp), H_.reshape(shp)
        if op == "neg":
            return -hi[0], -lo[0]
        if op == "abs":
            return _min_abs(lo[0], hi[0]), _mag(lo[0], hi[0])
        if op == "add":
            L_, H_ = lo[0], hi[0]
            for l2, h2 in zip(lo[1:], hi[1:]):
                L_, H_ = L_ + l2, H_ + h2
            return L_, H_
        if op == "sub":
            return lo[0] - hi[1], hi[0] - lo[1]
        if op == "mul":
            L_, H_ = lo[0], hi[0]
            for l2, h2 in zip(lo[1:], hi[1:]):
                L_, H_ = _iv_mul(L_, H_, l2, h2)
            return L_, H_
        if op == "div":
            if bool(((lo[1] <= 0) & (hi[1] >= 0)).any()):
                raise _exceptional("division by a range that contains 0")
            return _iv_mul(lo[0], hi[0], 1 / hi[1], 1 / lo[1])
        if op == "reciprocal":
            if bool(((lo[0] <= 0) & (hi[0] >= 0)).any()):
                raise _exceptional("reciprocal of a range that contains 0")
            return 1 / hi[0], 1 / lo[0]
        if op == "rsqrt":
            if bool((lo[0] <= 0).any()):
                raise _exceptional("rsqrt of a range that reaches 0")
            return torch.rsqrt(hi[0]), torch.rsqrt(lo[0])
        if op in ("max", "min"):
            f = torch.maximum if op == "max" else torch.minimum
            return f(lo[0], lo[1]), f(hi[0], hi[1])
        if op == "where":
            c = ins_c[0].bool()
            if not torch.equal(ins[0][0], ins[0][1]):
                return torch.minimum(lo[1], lo[2]), torch.maximum(hi[1], hi[2])
            return torch.where(c, lo[1], lo[2]), torch.where(c, hi[1], hi[2])
        if op in ("sin", "cos"):
            return torch.full_like(lo[0], -1.0), torch.full_like(lo[0], 1.0)
        if op == "gelu" or op == "silu":
            f = lambda v: self.eval64(g, n, [v])  # noqa: E731
            yl, yh = f(lo[0]), f(hi[0])
            L_, H_ = yl.minimum(yh), yl.maximum(yh)
            if op == "gelu":
                crit = (lo[0] <= -0.7517) & (hi[0] >= -0.7526)
                L_ = torch.where(crit, torch.full_like(L_, GELU_MIN), L_)
            else:
                crit = (lo[0] <= -1.2784) & (hi[0] >= -1.2786)
                L_ = torch.where(crit, torch.full_like(L_, SILU_MIN), L_)
            return L_ - 1e-12 * (1 + L_.abs()), H_ + 1e-12 * (1 + H_.abs())
        if op == "pow":
            p = float(ins_c[1].reshape(-1)[0])
            pts = [lo[0], hi[0]]
            vals = [torch.pow(t, p) for t in pts]
            L_ = torch.minimum(*vals)
            H_ = torch.maximum(*vals)
            if p > 0 and float(p).is_integer() and int(p) % 2 == 0:
                L_ = torch.where((lo[0] < 0) & (hi[0] > 0), torch.zeros_like(L_), L_)
            if torch.isnan(L_).any() or torch.isnan(H_).any():
                raise _exceptional("pow of a negative base")
            return L_, H_
        if op in ("matmul", "conv"):
            cm_a, cr_a = (lo[0] + hi[0]) / 2, (hi[0] - lo[0]) / 2
            cm_b, cr_b = (lo[1] + hi[1]) / 2, (hi[1] - lo[1]) / 2
            if op == "matmul":
                mm = torch.matmul
                c = mm(cm_a, cm_b)
                r = mm(cr_a, cm_b.abs()) + mm(cm_a.abs(), cr_b) + mm(cr_a, cr_b)
            else:
                c = self._conv(cm_a, cm_b, a)
                r = self._conv(cr_a, cm_b.abs(), a) + self._conv(cm_a.abs(), cr_b, a) \
                    + self._conv(cr_a, cr_b, a)
            return c - r, c + r
        if op in ("softmax", "log_softmax"):
            ax = a["axis"]
            el, eh = lo[0], hi[0]
            m = eh.amax(dim=ax, keepdim=True)
            Sl = torch.exp(el - m).sum(dim=ax, keepdim=True)
            Sh = torch.exp(eh - m).sum(dim=ax, keepdim=True)
            # y_i = e_i / (e_i + rest_i); increasing in x_i, decreasing in the others
            rest_h = Sh - torch.exp(eh - m)
            rest_l = Sl - torch.exp(el - m)
            if op == "softmax":
                yl = torch.exp(el - m) / (torch.exp(el - m) + rest_h.clamp(min=0))
                yh = torch.exp(eh - m) / (torch.exp(eh - m) + rest_l.clamp(min=0))
                return yl.clamp(0, 1), yh.clamp(0, 1)
            yl = (el - m) - torch.log(torch.exp(el - m) + rest_h.clamp(min=0))
            yh = (eh - m) - torch.log(torch.exp(eh - m) + rest_l.clamp(min=0))
            return yl, yh.clamp(max=0)
        if op in ("layer_norm", "rms_norm"):
            return self._norm_ibp(n, lo, hi, ins_c)[:2]
        if op == "batch_norm":
            w, b, mean, var = ins_c[1:5]
            shp = (1, -1) + (1,) * (lo[0].dim() - 2)
            s = (w / torch.sqrt(var + a["eps"])).reshape(shp)
            t = b.reshape(shp) - mean.reshape(shp) * s
            y1, y2 = lo[0] * s + t, hi[0] * s + t
            return y1.minimum(y2), y1.maximum(y2)
        raise _unsupported_op(f"no range rule for {op}")

    def _apply_lin(self, g, n, ins, ins_c):
        return self.eval64(g, n, ins)

    def _norm_ibp(self, n, lo, hi, ins_c):
        """Ranges of layer_norm / rms_norm, plus (sigma_lo, sigma_hi, zl, zh) per row."""
        torch = _torch()
        a = n.attrs
        axes = a["axes"]
        x_l, x_h = lo[0], hi[0]
        eps = float(a["eps"])
        if n.op == "layer_norm":
            ml = x_l.mean(dim=axes, keepdim=True)
            mh = x_h.mean(dim=axes, keepdim=True)
            dl, dh = x_l - mh, x_h - ml                     # x_i - mean
            sq_l = torch.where((dl <= 0) & (dh >= 0), torch.zeros_like(dl),
                               dl.abs().minimum(dh.abs()) ** 2)
            sq_h = _mag(dl, dh) ** 2
            var_l = sq_l.mean(dim=axes, keepdim=True)
            var_h = sq_h.mean(dim=axes, keepdim=True)
        else:
            dl, dh = x_l, x_h
            var_l = (_min_abs(x_l, x_h) ** 2).mean(dim=axes, keepdim=True)
            var_h = (_mag(x_l, x_h) ** 2).mean(dim=axes, keepdim=True)
        sl, sh = torch.sqrt(var_l + eps), torch.sqrt(var_h + eps)
        zl, zh = _iv_mul(dl, dh, 1 / sh, 1 / sl)
        if n.op == "layer_norm":
            nt = int(np.prod([x_l.shape[d] for d in axes]))
            cap = math.sqrt(max(nt - 1, 1))
            zl, zh = zl.clamp(min=-cap), zh.clamp(max=cap)
        yl, yh = zl, zh
        w = ins_c[1] if len(ins_c) > 1 else None
        b = ins_c[2] if len(ins_c) > 2 else None
        if w is not None:
            yl, yh = _iv_mul(yl, yh, w, w)
        if b is not None:
            yl, yh = yl + b, yh + b
        return yl, yh, sl, sh, zl, zh

    # ================================================================================ #
    # Slopes (derivative ranges of the exact operation over a range)                    #
    # ================================================================================ #

    def _slope(self, op, n, lo, hi, ins_c=None):
        torch = _torch()
        if op == "relu":
            sl = (lo > 0).double()
            sh = (hi > 0).double()
            return sl, sh
        if op == "exp":
            return torch.exp(lo), torch.exp(hi)
        if op == "log":
            return 1 / hi, 1 / lo
        if op == "sqrt":
            if bool((lo <= 0).any()):
                raise _unsupported_path("sqrt near 0: unbounded slope")
            return 0.5 / torch.sqrt(hi), 0.5 / torch.sqrt(lo)
        if op == "rsqrt":
            return -0.5 * lo ** -1.5, -0.5 * hi ** -1.5
        if op == "reciprocal":
            m, M = _min_abs(lo, hi), _mag(lo, hi)
            return -1 / (m * m), -1 / (M * M)
        amin, amax = _min_abs(lo, hi), _mag(lo, hi)
        if op == "tanh":
            return 1 - torch.tanh(amax) ** 2, 1 - torch.tanh(amin) ** 2
        if op == "sigmoid":
            s1, s2 = torch.sigmoid(amax), torch.sigmoid(amin)
            return s1 * (1 - s1), s2 * (1 - s2)
        if op == "erf":
            k_ = 2 / SQRT_PI
            return k_ * torch.exp(-amax ** 2), k_ * torch.exp(-amin ** 2)
        if op in ("sin", "cos"):
            return torch.full_like(lo, -1.0), torch.full_like(lo, 1.0)
        if op in ("gelu", "silu"):
            c = (lo + hi) / 2
            r = (hi - lo) / 2
            x = c.clone().requires_grad_(True)
            with torch.enable_grad():
                if op == "gelu":
                    y = torch.nn.functional.gelu(
                        x, approximate="tanh" if n.attrs.get("approximate") == "tanh" else "none")
                else:
                    y = torch.nn.functional.silu(x)
                (d,) = torch.autograd.grad(y.sum(), x)
            d = d.detach()
            L2, glo, ghi = (GELU_D2, -0.1291, GELU_LIP) if op == "gelu" else \
                (SILU_D2, -0.0999, SILU_LIP)
            return (d - L2 * r - 1e-9).clamp(min=glo), (d + L2 * r + 1e-9).clamp(max=ghi)
        if op == "pow":
            p = float(ins_c[1].reshape(-1)[0])
            v1, v2 = p * torch.pow(lo, p - 1), p * torch.pow(hi, p - 1)
            sl, sh = v1.minimum(v2), v1.maximum(v2)
            if p - 1 > 0:
                z = torch.zeros_like(sl)
                span = (lo < 0) & (hi > 0)
                sl = torch.where(span, sl.minimum(z), sl)
                sh = torch.where(span, sh.maximum(z), sh)
            return sl, sh
        raise _unsupported_op(f"no slope rule for {op}")

    # ================================================================================ #
    # Forward |J| rule for the bound kinds                                              #
    # ================================================================================ #

    def fwd(self, g, n, ins, ranges, out_r, cvals):
        """|J|-forward rule: bounds (5 kinds) of n's output from bounds of its inputs.
        `ranges` / `out_r` enclose the exact and both programs' values; `cvals` are centers."""
        torch = _torch()
        F = torch.nn.functional
        op, a = n.op, n.attrs
        x = ins[0] if ins else None
        out_shape = tuple(g.values[n.outputs[0]].shape)

        def zeros():
            return torch.zeros((5,) + out_shape, dtype=torch.float64, device=self.dev)

        def lin(f, b):
            return torch.stack([f(b[i]) for i in range(5)])

        def R(i):
            return ranges[i]

        def bc(b):
            """[5, *s] -> [5, *broadcast to out_shape]"""
            extra = len(out_shape) - (b.dim() - 1)
            return b.reshape((5,) + (1,) * extra + tuple(b.shape[1:])).expand((5,) + out_shape)

        if op in ("reshape", "permute", "broadcast_to", "slice", "cast"):
            if x is None:
                return zeros()
            return lin(lambda t: self.eval64(g, n, [t] + [None] * (len(n.inputs) - 1)), x)
        if op == "concat":
            parts = [b if b is not None else torch.zeros((5,) + tuple(g.values[i].shape),
                                                         dtype=torch.float64, device=self.dev)
                     for b, i in zip(ins, n.inputs)]
            return torch.cat(parts, dim=a["axis"] + 1)
        if op == "pad":
            return lin(lambda t: self._pad(t, a["pads"], 0.0), x) if x is not None else zeros()
        if op == "gather":
            if x is None:
                return zeros()
            idx = cvals[1]
            return lin(lambda t: self._gather(t, idx, a["axis"]), x)
        if op in ("add", "sub"):
            r = zeros()
            for b in ins:
                if b is not None:
                    r = r + bc(b)
            return r
        if op == "neg" or op == "abs":
            return x if x is not None else zeros()
        if op == "mul":
            r = zeros()
            mags = [_mag(*R(i)).expand(out_shape) for i in range(len(ins))]
            for i, b in enumerate(ins):
                if b is None:
                    continue
                f = bc(b)
                for j, mg in enumerate(mags):
                    if j != i:
                        f = f * mg
                r = r + f
            return r
        if op == "div":
            (al, ah), (bl, bh) = R(0), R(1)
            m = _min_abs(bl, bh).expand(out_shape)
            r = zeros()
            if ins[0] is not None:
                r = r + bc(ins[0]) / m
            if ins[1] is not None:
                r = r + bc(ins[1]) * _mag(al, ah).expand(out_shape) / (m * m)
            return r
        if op in ("max", "min"):
            r = zeros()
            for b in ins:
                if b is not None:
                    r = torch.maximum(r, bc(b))
            return r
        if op == "where":
            r = zeros()
            for b in ins[1:]:
                if b is not None:
                    r = torch.maximum(r, bc(b))
            return r
        if op in ("relu", "exp", "log", "sqrt", "rsqrt", "reciprocal", "tanh", "sigmoid", "erf",
                  "sin", "cos", "gelu", "silu", "pow"):
            if x is None:
                return zeros()
            lo, hi = R(0)
            sl, sh = self._slope(op, n, lo, hi, cvals)
            r = x * _mag(sl, sh)
            if op == "sqrt":
                r = torch.minimum(r, torch.sqrt(x))
            return r
        if op == "matmul":
            r = zeros()
            (al, ah), (bl, bh) = R(0), R(1)
            if ins[0] is not None:
                Bm = _mag(bl, bh)
                r = r + torch.stack([torch.matmul(ins[0][i], Bm) for i in range(5)])
            if ins[1] is not None:
                Am = _mag(al, ah)
                r = r + torch.stack([torch.matmul(Am, ins[1][i]) for i in range(5)])
            return r
        if op == "conv":
            r = zeros()
            (xl, xh), (wl, wh) = R(0), R(1)
            if ins[0] is not None:
                Wm = _mag(wl, wh)
                r = r + torch.stack([self._conv(ins[0][i], Wm, a) for i in range(5)])
            if ins[1] is not None and bool((ins[1] > 0).any()):
                raise _unsupported_op("convolution with a variable kernel")
            return r
        if op in ("reduce_sum", "reduce_mean", "avg_pool"):
            return lin(lambda t: self.eval64(g, n, [t]), x) if x is not None else zeros()
        if op in ("reduce_max", "reduce_min"):
            return lin(lambda t: self._reduce_max_abs(g, n, t), x) if x is not None else zeros()
        if op == "max_pool":
            if x is None:
                return zeros()
            p, k_, st, dil = self._pool_args(a)
            return lin(lambda t: F.max_pool2d(F.pad(t, p, value=0.0), k_, st, 0, dil,
                                              ceil_mode=a.get("ceil_mode", False)), x)
        if op in ("softmax", "log_softmax"):
            if x is None:
                return zeros()
            ax = a["axis"] + 1
            yl, yh = out_r
            if op == "softmax":
                yh5 = yh.unsqueeze(0)
                s = (yh5 * x).sum(dim=ax, keepdim=True)
                return yh5 * (x + s)
            p = torch.exp(yh).unsqueeze(0)
            return x + (p * x).sum(dim=ax, keepdim=True)
        if op in ("layer_norm", "rms_norm"):
            if x is None:
                return zeros()
            lo, hi = R(0)
            ins_c = cvals
            _, _, sl, sh, zl, zh = self._norm_ibp(n, [lo], [hi], [None] + list(ins_c[1:]))
            axes = tuple(d + 1 for d in a["axes"])
            w = ins_c[1].abs() if len(ins_c) > 1 and ins_c[1] is not None else 1.0
            zm = _mag(zl, zh).unsqueeze(0)
            if op == "layer_norm":
                core = x + x.mean(dim=axes, keepdim=True) + \
                    zm * (zm * x).mean(dim=axes, keepdim=True)
            else:
                core = x + zm * (zm * x).mean(dim=axes, keepdim=True)
            return w * core / sl.unsqueeze(0)
        if op == "batch_norm":
            w, b, mean, var = cvals[1:5]
            shp = (1, 1, -1) + (1,) * (x.dim() - 3)
            s = (w / torch.sqrt(var + a["eps"])).abs().reshape(shp)
            return x * s
        raise _unsupported_op(f"no forward rule for {op}")

    def _reduce_max_abs(self, g, n, t):
        torch = _torch()
        a = n.attrs
        axes = a.get("axes") or tuple(range(t.dim()))
        r = t
        for ax in sorted(axes, reverse=True):
            r = r.amax(dim=ax, keepdim=True)
        if not a.get("keepdim", False):
            r = r.squeeze(tuple(axes))
        return r

    # ================================================================================ #
    # Back-substitution steps: point (midpoint) VJP and per-node residual               #
    # ================================================================================ #

    def _inb(self, g, n, i):
        """Bounds [5, *shape] of input i of n (zeros for constants, None for non-floats)."""
        torch = _torch()
        v = n.inputs[i]
        if v is None or np.dtype(g.values[v].dtype).kind != "f":
            return None
        B = self._bounds_of(v)
        if B is None:
            B = torch.zeros((5,) + tuple(g.values[v].shape), dtype=torch.float64,
                            device=self.dev)
        return B

    def _jac(self, g, n):
        """Midpoint and radius data of n's Jacobian over the input ranges (cached)."""
        torch = _torch()
        hit = self._jcache.get(n.id)
        if hit is not None:
            return hit
        op, a = n.op, n.attrs
        J = {}
        R = lambda i: self._range_any(n.inputs[i])  # noqa: E731
        if op in ("relu", "exp", "log", "sqrt", "rsqrt", "reciprocal", "tanh", "sigmoid", "erf",
                  "sin", "cos", "gelu", "silu", "pow"):
            lo, hi = R(0)
            sl, sh = self._slope(op, n, lo, hi, [self.c0.get(i) for i in n.inputs])
            J["m"], J["r"] = (sl + sh) / 2, (sh - sl) / 2
            if op == "relu" and getattr(self.cfg, "relu_point_slope", True):
                # An unstable ReLU whose range lies mostly on one side: take the slope of that
                # side (1 or 0). For a, a' in [lo, hi],
                #   slope 1: |relu(a) - relu(a') - (a - a')| = |min(a,0) - min(a',0)|
                #            <= min(|a - a'|, -lo)
                #   slope 0: |relu(a) - relu(a')| <= min(|a - a'|, hi)
                # which beats the midpoint slope's |a - a'| / 2 when the short side is short.
                pos = (lo + hi) > 0
                gap = torch.where(pos, (-lo).clamp(min=0), hi.clamp(min=0))
                use = (sl != sh) & (gap < 0.25 * (hi - lo))
                J["m"] = torch.where(use, pos.double(), J["m"])
                J["r"] = torch.where(use, torch.zeros_like(J["r"]), J["r"])
                J["gap"] = torch.where(use, gap, torch.full_like(gap, math.inf))
        elif op in ("mul", "div", "matmul", "max", "min", "abs"):
            rs = [R(i) for i in range(len(n.inputs))]
            J["mid"] = [(lo + hi) / 2 for lo, hi in rs]
            J["rad"] = [(hi - lo) / 2 for lo, hi in rs]
            if op == "div":
                (al, ah), (bl, bh) = rs
                if bool(((bl <= 0) & (bh >= 0)).any()):
                    raise _exceptional("division by a range that contains 0")
                il, ih = 1 / bh, 1 / bl
                J["inv"] = ((il + ih) / 2, (ih - il) / 2)
                ql, qh = _iv_mul(al, ah, il, ih)
                ql, qh = _iv_mul(ql, qh, il, ih)
                J["q"] = (-(ql + qh) / 2, (qh - ql) / 2)        # d(a/b)/db = -a/b^2
            if op in ("max", "min"):
                (al, ah), (bl, bh) = rs
                if op == "max":
                    a_dom, b_dom = al > bh, bl > ah
                else:
                    a_dom, b_dom = ah < bl, bh < al
                sl = a_dom.double()
                sh = (~b_dom).double()
                J["s"] = ((sl + sh) / 2, (sh - sl) / 2)
            if op == "abs":
                lo, hi = rs[0]
                sl = torch.where(lo > 0, 1.0, -1.0).double()
                sh = torch.where(hi < 0, -1.0, 1.0).double()
                J["s"] = ((sl + sh) / 2, (sh - sl) / 2)
        self._jcache[n.id] = J
        return J

    TAU_SMALL = 2.0 ** -20

    def residual(self, g, n):
        """R [5, *out]: for every coefficient row c at n's output, the part of
        c . Delta_out that the midpoint step c . J0 . Delta_in misses is at most |c| . R.
        It covers the Jacobian range (unstable ReLUs, slope ranges, products of two computed
        values, softmax/norm Jacobians away from the center) and, for float32 analysis, the
        rounding of the step itself."""
        torch = _torch()
        hit = self._res.get(n.id)
        if hit is not None or n.id in self._res:
            return hit
        op, a = n.op, n.attrs
        out_shape = tuple(g.values[n.outputs[0]].shape)
        b = [self._inb(g, n, i) for i in range(len(n.inputs))]
        fl = self.cdt != torch.float64
        heavy = op in ("matmul", "conv", "reduce_sum", "reduce_mean", "avg_pool", "softmax",
                       "log_softmax", "layer_norm", "rms_norm")
        tau = (self._step_tau(g, n) if heavy else self.TAU_SMALL) if fl else 0.0
        R = None
        if all(x is None for x in b):
            self._res[n.id] = None
            return None

        def bc(x):
            extra = len(out_shape) - (x.dim() - 1)
            return x.reshape((5,) + (1,) * extra + tuple(x.shape[1:])).expand((5,) + out_shape)

        def ex(t):
            return t.expand(out_shape) if t.shape != out_shape else t

        if op in ("relu", "exp", "log", "sqrt", "rsqrt", "reciprocal", "tanh", "sigmoid", "erf",
                  "sin", "cos", "gelu", "silu", "pow"):
            J = self._jac(g, n)
            R = b[0] * J["r"]
            if "gap" in J:
                gp = J["gap"].unsqueeze(0)
                R = R + torch.where(torch.isfinite(gp), torch.minimum(b[0], gp),
                                    torch.zeros_like(R))
        elif op == "mul":
            J = self._jac(g, n)
            R = 0
            for i in range(len(n.inputs)):
                if b[i] is None:
                    continue
                rad_other = None
                for j in range(len(n.inputs)):
                    if j == i:
                        continue
                    m_, r_ = J["mid"][j], J["rad"][j]
                    if rad_other is None:
                        mo, ro = m_, r_
                        rad_other = True
                    else:
                        mo, ro = _imul(mo, ro, m_, r_)
                R = R + bc(b[i]) * ex(ro)
        elif op == "div":
            J = self._jac(g, n)
            R = 0
            if b[0] is not None:
                R = R + bc(b[0]) * ex(J["inv"][1])
            if b[1] is not None:
                R = R + bc(b[1]) * ex(J["q"][1])
        elif op in ("max", "min"):
            sr = self._jac(g, n)["s"][1]
            R = 0
            for x in b:
                if x is not None:
                    R = R + bc(x) * ex(sr)
        elif op == "abs":
            R = b[0] * self._jac(g, n)["s"][1]
        elif op == "matmul":
            J = self._jac(g, n)
            R = 0
            (Ar, Br) = J["rad"]
            if b[0] is not None and bool((Br > 0).any()):
                R = R + torch.stack([torch.matmul(b[0][i], Br) for i in range(5)])
            if b[1] is not None and bool((Ar > 0).any()):
                R = R + torch.stack([torch.matmul(Ar, b[1][i]) for i in range(5)])
        elif op in ("reduce_max", "reduce_min", "max_pool"):
            R = self._res_max(g, n, b[0])
        elif op in ("softmax", "log_softmax"):
            R = self._res_softmax(g, n, b[0])
        elif op in ("layer_norm", "rms_norm"):
            R = self._res_norm(g, n, b[0])
        if tau:
            ranges = [self._range_any(i) if i is not None else None for i in n.inputs]
            cvals = [self.c0.get(i) if i is not None else None for i in n.inputs]
            fwd = self.fwd(g, n, b, ranges, self.rng[n.outputs[0]], cvals)
            R = fwd * tau if R is None or (not torch.is_tensor(R)) else R + fwd * tau
        if R is not None and not torch.is_tensor(R):
            R = None
        self._res[n.id] = R
        return R

    def _step_tau(self, g, n):
        """Relative error of the float32 VJP of n: each coefficient is a sum of at most T
        products, so its rounding is at most gamma_T times the sum of magnitudes (with the
        same no-Winograd assumption as for the programs), doubled for safety."""
        op = n.op
        if op == "conv":
            w = g.values[n.inputs[1]].shape
            T = int(w[0] * np.prod(w[2:]))
        elif op == "matmul":
            sb = g.values[n.inputs[1]].shape
            sa = g.values[n.inputs[0]].shape
            T = max(sb[-1] if len(sb) > 1 else 1, sa[-2] if len(sa) > 1 else 1)
        elif op in ("reduce_sum", "reduce_mean", "avg_pool"):
            T = 1
        else:
            T = int(g.values[n.inputs[0]].shape[-1])
        u = 2.0 ** -24
        return min(2.0 ** -12, 2 * (T + 2) * u / (1 - (T + 2) * u))

    def _res_max(self, g, n, b):
        """max = convex combination of the candidates; the point step routes to the argmax at
        the center, so the rest is at most 2 * max over the candidates of the bound."""
        torch = _torch()
        F = torch.nn.functional
        if b is None:
            return None
        a = n.attrs
        lo, hi = self._range_any(n.inputs[0])
        if n.op == "max_pool":
            p, k_, st, dil = self._pool_args(a)
            lp = F.pad(lo, p, value=-math.inf)
            hp = F.pad(hi, p, value=-math.inf)
            mx_lo = F.max_pool2d(lp, k_, st, 0, dil, ceil_mode=a.get("ceil_mode", False))
            # a window is unstable if some element other than the max can reach max_lo
            out_shape = tuple(g.values[n.outputs[0]].shape)
            N, C, H, W = lp.shape
            ul = F.unfold(lp.reshape(N * C, 1, H, W), k_, dil, 0, st)
            uh = F.unfold(hp.reshape(N * C, 1, H, W), k_, dil, 0, st)
            L = int(np.prod(out_shape[2:]))
            if ul.shape[-1] != L:
                raise _unsupported_op("max_pool window layout (ceil_mode/padding) not supported")
            best = ul.amax(1, keepdim=True)
            unstable = ((uh >= best).sum(1) > 1).reshape(out_shape)
            bm = torch.stack([F.max_pool2d(F.pad(b[i], p, value=0.0), k_, st, 0, dil,
                                           ceil_mode=a.get("ceil_mode", False))
                              for i in range(5)])
            del mx_lo
            return 2 * bm * unstable.double()
        axes = a.get("axes") or tuple(range(lo.dim()))
        if n.op == "reduce_min":
            lo, hi = -hi, -lo
        best = lo
        for ax in axes:
            best = best.amax(dim=ax, keepdim=True)
        cnt = (hi >= best).double()
        bm = b
        for ax in axes:
            cnt = cnt.sum(dim=ax, keepdim=True)
            bm = bm.amax(dim=ax + 1, keepdim=True)
        unstable = (cnt > 1).double()
        R = 2 * bm * unstable.unsqueeze(0)
        if not a.get("keepdim", False):
            R = R.squeeze(tuple(ax + 1 for ax in axes))
        return R

    def _res_softmax(self, g, n, b):
        torch = _torch()
        if b is None:
            return None
        ax = n.attrs["axis"] + 1
        y0 = self.c0[n.outputs[0]]
        yl, yh = self.rng[n.outputs[0]]
        if n.op == "softmax":
            ry = (yh - y0).maximum(y0 - yl).clamp(min=0).unsqueeze(0)
            y0_, yh_ = y0.unsqueeze(0), yh.unsqueeze(0)
            return ry * (b + (yh_ * b).sum(ax, keepdim=True)) + y0_ * (ry * b).sum(ax, keepdim=True)
        p0, pl, ph = torch.exp(y0), torch.exp(yl), torch.exp(yh)
        rp = (ph - p0).maximum(p0 - pl).clamp(min=0).unsqueeze(0)
        return (rp * b).sum(ax, keepdim=True).expand_as(b).clone()

    def _norm_center(self, n):
        """sigma and z at the input center (per row)."""
        torch = _torch()
        x = self.c0[n.inputs[0]].double()
        axes = n.attrs["axes"]
        eps = float(n.attrs["eps"])
        if n.op == "layer_norm":
            mu = x.mean(dim=axes, keepdim=True)
            d = x - mu
        else:
            d = x
        s = torch.sqrt((d * d).mean(dim=axes, keepdim=True) + eps)
        return s, d / s

    def _res_norm(self, g, n, b):
        torch = _torch()
        if b is None:
            return None
        a = n.attrs
        axes = tuple(d + 1 for d in a["axes"])
        lo, hi = self._range_any(n.inputs[0])
        ins_c = [self.c0.get(i) if i is not None else None for i in n.inputs]
        _, _, sl, sh, zl, zh = self._norm_ibp(n, [lo], [hi], [None] + ins_c[1:])
        s0, z0 = self._norm_center(n)
        w = ins_c[1].abs() if len(ins_c) > 1 and ins_c[1] is not None else 1.0
        d_inv = (1 / sl - 1 / s0).abs().maximum((1 / s0 - 1 / sh).abs()).unsqueeze(0)
        zm = _mag(zl, zh).unsqueeze(0)
        rz = (zh - z0).maximum(z0 - zl).clamp(min=0).unsqueeze(0)
        z0a = z0.abs().unsqueeze(0)
        inv0 = (1 / s0).unsqueeze(0)
        mzb = (zm * b).mean(dim=axes, keepdim=True)
        if n.op == "layer_norm":
            full = b + b.mean(dim=axes, keepdim=True) + zm * mzb
        else:
            full = b + zm * mzb
        cross = rz * mzb + z0a * (rz * b).mean(dim=axes, keepdim=True)
        return w * (d_inv * full + inv0 * cross)

    def pvjp(self, g, n, c):
        """Coefficients for n's inputs from the point coefficient c at its output, using the
        midpoint Jacobian."""
        torch = _torch()
        F = torch.nn.functional
        op, a = n.op, n.attrs
        Bn = c.shape[0]
        lead = (Bn,)
        ins = n.inputs
        shapes = [tuple(g.values[i].shape) if i is not None else None for i in ins]
        floats = [i is not None and np.dtype(g.values[i].dtype).kind == "f" and
                  g.values[i].kind != "const" for i in ins]
        cdt = self.cdt
        res = [None] * len(ins)

        def to(t):
            return t.to(cdt)

        if op in ("reshape", "permute", "broadcast_to", "slice", "pad", "cast"):
            if floats[0]:
                res[0] = self._lin_vjp(op, a, shapes[0], c.shape[1:])(c)
            return res
        if op == "concat":
            sizes = [s[a["axis"]] for s in shapes]
            return [p if f else None for p, f in zip(c.split(sizes, dim=a["axis"] + 1), floats)]
        if op == "gather":
            if floats[0]:
                raise _unsupported_op("gather from a computed table")
            return res
        if op == "add":
            return [_sum_to(c, s, lead) if f else None for s, f in zip(shapes, floats)]
        if op == "sub":
            if floats[0]:
                res[0] = _sum_to(c, shapes[0], lead)
            if floats[1]:
                res[1] = _sum_to(-c, shapes[1], lead)
            return res
        if op == "neg":
            return [-c] if floats[0] else res
        if op == "where":
            cond = self.c0[ins[0]].bool()
            for i, mask in ((1, cond), (2, ~cond)):
                if floats[i]:
                    res[i] = _sum_to(c * mask.to(cdt), shapes[i], lead)
            return res
        if op in ("relu", "exp", "log", "sqrt", "rsqrt", "reciprocal", "tanh", "sigmoid", "erf",
                  "sin", "cos", "gelu", "silu", "pow"):
            if floats[0]:
                res[0] = c * to(self._jac(g, n)["m"])
            return res
        if op == "mul":
            J = self._jac(g, n)
            for i in range(len(ins)):
                if not floats[i]:
                    continue
                p = None
                for j in range(len(ins)):
                    if j != i:
                        p = J["mid"][j] if p is None else p * J["mid"][j]
                res[i] = _sum_to(c * to(p), shapes[i], lead)
            return res
        if op == "div":
            J = self._jac(g, n)
            if floats[0]:
                res[0] = _sum_to(c * to(J["inv"][0]), shapes[0], lead)
            if floats[1]:
                res[1] = _sum_to(c * to(J["q"][0]), shapes[1], lead)
            return res
        if op in ("max", "min"):
            sm = to(self._jac(g, n)["s"][0])
            if floats[0]:
                res[0] = _sum_to(c * sm, shapes[0], lead)
            if floats[1]:
                res[1] = _sum_to(c * (1 - sm), shapes[1], lead)
            return res
        if op == "abs":
            return [c * to(self._jac(g, n)["s"][0])]
        if op == "matmul":
            J = self._jac(g, n)
            Am, Bm = [to(t) for t in J["mid"]]
            sa, sb = shapes
            A1, B1 = len(sa) == 1, len(sb) == 1
            c2 = c.unsqueeze(-1) if B1 else c
            if A1:
                c2 = c2.unsqueeze(-2)
                Am = Am.unsqueeze(0)
            if B1:
                Bm = Bm.unsqueeze(-1)
            if floats[0]:
                gA = torch.matmul(c2, Bm.transpose(-1, -2))
                if A1:
                    gA = gA.squeeze(-2)
                res[0] = _sum_to(gA, sa, lead)
            if floats[1]:
                gB = torch.matmul(Am.transpose(-1, -2), c2)
                if B1:
                    gB = gB.squeeze(-1)
                res[1] = _sum_to(gB, sb, lead)
            return res
        if op == "conv":
            if floats[1]:
                raise _unsupported_op("convolution with a variable kernel")
            if floats[0]:
                res[0] = self._conv_vjp(shapes[0], to(self.c0[ins[1]]), c, a)
            return res
        if op in ("reduce_sum", "reduce_mean", "avg_pool"):
            res[0] = self._lin_vjp(op, a, shapes[0], c.shape[1:], g=g, n=n)(c)
            return res
        if op in ("reduce_max", "reduce_min", "max_pool"):
            x0 = to(self.c0[ins[0]])
            xb = x0.unsqueeze(0).expand((Bn,) + tuple(x0.shape)).reshape(
                (Bn * x0.shape[0],) + tuple(x0.shape[1:])).clone().requires_grad_(True)
            with torch.enable_grad():
                y = self.eval64(g, n, [xb])
                (gx,) = torch.autograd.grad(y, xb, c.reshape(y.shape))
            res[0] = gx.reshape((Bn,) + tuple(x0.shape))
            return res
        if op == "softmax":
            ax = a["axis"] + 1
            y0 = to(self.c0[n.outputs[0]]).unsqueeze(0)
            res[0] = y0 * (c - (c * y0).sum(ax, keepdim=True))
            return res
        if op == "log_softmax":
            ax = a["axis"] + 1
            p0 = torch.exp(to(self.c0[n.outputs[0]])).unsqueeze(0)
            res[0] = c - p0 * c.sum(ax, keepdim=True)
            return res
        if op in ("layer_norm", "rms_norm"):
            axes = tuple(d + 1 for d in a["axes"])
            s0, z0 = self._norm_center(n)
            s0, z0 = to(s0).unsqueeze(0), to(z0).unsqueeze(0)
            w = self.c0.get(ins[1]) if len(ins) > 1 and ins[1] is not None else None
            t = c * to(w) if w is not None else c
            d = t - z0 * (t * z0).mean(dim=axes, keepdim=True)
            if op == "layer_norm":
                d = d - t.mean(dim=axes, keepdim=True)
            res[0] = d / s0
            return res
        if op == "batch_norm":
            w, b, mean, var = [self.c0[i] for i in ins[1:5]]
            shp = (1, 1, -1) + (1,) * (c.dim() - 3)
            res[0] = c * to(w / torch.sqrt(var + a["eps"])).reshape(shp)
            return res
        if op in ("eq", "ne", "lt", "le", "gt", "ge", "and", "or", "not"):
            return res
        raise _unsupported_op(f"no back-substitution rule for {op}")

    def _lin_vjp(self, op, a, in_shape, out_shape, g=None, n=None):
        """VJP of a linear, sign-free operation (applied separately to centers and radii)."""
        torch = _torch()
        F = torch.nn.functional

        def f(c):
            Bn = c.shape[0]
            if op == "reshape" or op == "cast":
                return c.reshape((Bn,) + tuple(in_shape))
            if op == "permute":
                inv = np.argsort(a["perm"])
                return c.permute(0, *[int(i) + 1 for i in inv])
            if op == "broadcast_to":
                return _sum_to(c, in_shape, (Bn,))
            if op == "slice":
                z = torch.zeros((Bn,) + tuple(in_shape), dtype=c.dtype, device=c.device)
                idx = [slice(None)] * (len(in_shape) + 1)
                for axis, s, e, st in a["spec"]:
                    idx[axis + 1] = slice(s, e, st)
                z[tuple(idx)] = c
                return z
            if op == "pad":
                r = len(in_shape)
                b = a["pads"][:r]
                idx = [slice(None)] + [slice(b[d], b[d] + in_shape[d]) for d in range(r)]
                return c[tuple(idx)]
            if op in ("reduce_sum", "reduce_mean"):
                axes = a.get("axes") or tuple(range(len(in_shape)))
                if not a.get("keepdim", False):
                    for ax in sorted(axes):
                        c = c.unsqueeze(ax + 1)
                c = c.expand((Bn,) + tuple(in_shape))
                if op == "reduce_mean":
                    c = c / float(np.prod([in_shape[d] for d in axes]))
                return c
            if op == "avg_pool":
                x = torch.zeros((Bn * in_shape[0],) + tuple(in_shape[1:]), dtype=c.dtype,
                                device=c.device, requires_grad=True)
                p, k_, st, _ = self._pool_args(a)
                with torch.enable_grad():
                    y = F.avg_pool2d(x, k_, st, (p[2], p[0]), ceil_mode=a.get("ceil_mode", False),
                                     count_include_pad=a.get("count_include_pad", True))
                    (gx,) = torch.autograd.grad(y, x, c.reshape(y.shape))
                return gx.reshape((Bn,) + tuple(in_shape))
            raise _unsupported_op(f"no linear VJP for {op}")
        return f

    def _conv_vjp(self, in_shape, w, c, a):
        torch = _torch()
        F = torch.nn.functional
        Bn = c.shape[0]
        nsp = len(in_shape) - 2
        pads = a["pads"]
        b, e = pads[:nsp], pads[nsp:]
        if tuple(b) != tuple(e):
            raise _unsupported_op("asymmetric convolution padding")
        cc = c.reshape((Bn * c.shape[1],) + tuple(c.shape[2:]))
        shp = (Bn * in_shape[0],) + tuple(in_shape[1:])
        f = torch.nn.grad.conv2d_input if nsp == 2 else torch.nn.grad.conv1d_input
        gx = f(shp, w, cc, a["stride"], tuple(b), a["dilation"], a["groups"])
        return gx.reshape((Bn,) + tuple(in_shape))

    # ================================================================================ #
    # Local injections                                                                  #
    # ================================================================================ #

    def _injections(self, u):
        """Per-element local bounds at the unit outputs: [0, rho1, rho2, D, rho64]."""
        torch = _torch()
        if self.exact_ref:
            # program 1 only defines the real-valued function: it contributes no rounding and
            # nothing cancels against it, so D is program 2 versus exact arithmetic
            import collections
            rho = {1: collections.defaultdict(float), 2: self._side_rho(2, u)}
            zero_D = False
        else:
            rho = {j: self._side_rho(j, u) for j in (1, 2)}
            zero_D = self._relational_zero(u)
        r64 = self._ref_rho64(u)
        gap = self._idiom_gaps(u)
        out = {}
        for b in u.outputs:
            v = b.value_1
            shp = tuple(self.k.values[v].shape)
            I = torch.zeros((5,) + shp, dtype=torch.float64, device=self.dev)
            if np.dtype(self.k.values[v].dtype).kind != "f":
                out[v] = I
                continue
            g1, g2 = gap.get(v, (0.0, 0.0))
            if self.exact_ref:
                g1 = 0.0
            I[KE1] = rho[1][v] + g1
            I[KE2] = rho[2][v] + g2
            if zero_D:
                I[KD] = 0.0
            elif self.prob:
                I[KD] = torch.sqrt(I[KE1] ** 2 + I[KE2] ** 2)
            else:
                I[KD] = I[KE1] + I[KE2]
            I[K64] = r64[v]
            out[v] = I
        return out

    def _idiom_gaps(self, u):
        """Distance between a GELU written with approximate constants and the reference GELU,
        per element of the GELU input, for each side that uses such an idiom."""
        torch = _torch()
        k1, k2 = self.m.canonical_1, self.m.canonical_2
        g1 = [k1.nodes[n] for n in u.canonical_nodes_1 if k1.nodes[n].op == "gelu"]
        g2 = [k2.nodes[n] for n in u.canonical_nodes_2 if k2.nodes[n].op == "gelu"]
        if not any(n.meta.get("idiom_constants") for n in g1 + g2):
            return {}
        if len(g1) != 1 or len(u.outputs) != 1 or g1[0].outputs[0] != u.outputs[0].value_1:
            raise _unsupported_op("approximate GELU idiom inside a larger unit")
        n = g1[0]
        lo, hi = self._range_any(n.inputs[0])
        X = _mag(lo, hi)

        def gap(node):
            c = node.meta.get("idiom_constants")
            if not c:
                return torch.zeros_like(X)
            if node.attrs.get("approximate") == "tanh":
                s_, a_ = float(c["scale"]), float(c["cubic"])
                gp = 0.5 * X * (abs(s_ - SQRT_2_OVER_PI) * (X + 0.044715 * X ** 3)
                                + s_ * abs(a_ - 0.044715) * X ** 3)
            else:
                gp = X * X * abs(float(c["scale"]) - INV_SQRT2) / SQRT_PI
            return gp * fpm.SAFETY + 1e-16 * (X ** 4 + X)
        gap2 = torch.zeros_like(X)
        for m in g2:
            gap2 = torch.maximum(gap2, gap(m))
        return {u.outputs[0].value_1: (gap(n), gap2)}

    def _relational_zero(self, u):
        """Both programs produce bitwise-equal unit outputs: equal inputs, identical operations,
        identical known accumulation trees (Strategy 1)."""
        torch = _torch()
        for b in u.inputs:
            B = self.bnd.get(b.value_1)
            if B is not None and bool((B[KD] != 0).any()):
                return False
        st = self.st
        try:
            from .absint import FloatErrorInterpreter
            helper = FloatErrorInterpreter()
            helper._cur, helper._cfg = st, self.cfg
            g1, n1, i1, _ = helper._raw_unit(1, u, st)
            g2, n2, i2, _ = helper._raw_unit(2, u, st)
            if len(n1) != len(n2):
                return False
            fake = [None] * len(u.inputs)

            class _T:
                min_val = max_val = max_error = 0.0
            fake = [_T()] * len(u.inputs)
            _, idx1 = helper._bind_inputs(1, u, st, g1, i1, fake, True)
            _, idx2 = helper._bind_inputs(2, u, st, g2, i2, fake, True)
            k1, _ = helper._signatures(1, u, st, g1, n1, idx1)
            k2, _ = helper._signatures(2, u, st, g2, n2, idx2)
        except (InconclusiveError, KeyError):
            return False
        if sorted(k1.values()) != sorted(k2.values()):
            return False
        inv2 = {v: nid for nid, v in k2.items()}
        det = getattr(self.cfg, "assume_deterministic_kernels", True)
        same_kernel = False
        for nid in n1:
            a1, a2 = g1.nodes[nid], g2.nodes[inv2[k1[nid]]]
            if a1.op in EXACT_OPS or a1.op == "cast":
                continue
            ek = self._eager_kernel(1, a1) if det else None
            if ek is not None and ek == self._eager_kernel(2, a2):
                same_kernel = True       # one kernel, bitwise-equal inputs, equal attributes
                continue
            if a1.op in ("matmul", "conv"):
                o1 = self._order(1, a1, g1)
                o2 = self._order(2, a2, g2)
                if not (isinstance(o1, fpm.Known) and isinstance(o2, fpm.Known) and
                        o1.tree == o2.tree and o1.fp == o2.fp and o1.fp != "either"):
                    return False
                continue
            if a1.op in CORRECTLY_ROUNDED and self._separate(1, a1) and self._separate(2, a2):
                continue
            return False
        if same_kernel:
            self.assumptions.add(
                "PyTorch eager kernels are deterministic: the same operator on bitwise-equal "
                "inputs (same attributes, shapes, device and process) returns bitwise-equal "
                "results")
        return True

    def _eager_kernel(self, j, node):
        """The source operators behind a raw node when PyTorch's own kernels run all of them
        (the model itself, or the operators a Triton/TileLang path leaves in place); else
        None."""
        st = self.st
        if st.src[j].frontend != "torch.export":
            return None
        ops = []
        for o in sorted(node.meta.get("origins", ())):
            src = st.src[j].nodes.get(o)
            if src is None:
                return None
            impl = src.meta.get("impl") or st.src[j].meta.get("impl") or {}
            if impl.get("backend") != "eager":
                return None
            ops.append(src.op)
        return tuple(sorted(ops)) or None

    def _separate(self, j, node):
        st = self.st
        for o in node.meta.get("origins", ()):
            src = st.src[j].nodes.get(o)
            impl = src.meta.get("impl", st.src[j].meta.get("impl", {})) if src else {}
            if st.compute_count[j][o] == 1:
                continue
            if impl.get("backend") in DSL_BACKENDS and src is not None and \
                    _base_op(src.op) in SWAPPED_SOURCE and node.op == "add":
                continue
            return False
        return True

    def _impl(self, j, node):
        st = self.st
        o = sorted(node.meta.get("origins", ()))
        src = st.src[j].nodes.get(o[0]) if o else None
        impl = (src.meta.get("impl") if src is not None else None) or \
            st.src[j].meta.get("impl") or {"backend": st.src[j].frontend}
        return impl, (src.op if src is not None else "")

    def _order(self, j, node, g):
        impl, sop = self._impl(j, node)
        if node.op == "matmul":
            K = g.values[node.inputs[0]].shape[-1]
        elif node.op == "conv":
            K = int(np.prod(g.values[node.inputs[1]].shape[1:]))
        else:
            K = 0
        key = None
        if node.op == "matmul":
            # the exact kernel call: operand shapes, source operator, and whether its bias add
            # belongs to the same call (fprev.linear_key)
            key = (impl.get("backend", "unknown"), "matmul",
                   tuple(g.values[node.inputs[0]].shape), tuple(g.values[node.inputs[1]].shape),
                   _base_op(sop), self._bias_leaf(g, node) is not None)
        return self.oracle.order(impl, sop, node.op, K, key)

    # ---- per-side rounding inside a unit ----

    def _side_rho(self, j, u):
        """rho_j at the unit outputs: program j's rounding inside the unit, measured against
        the exact unit function at program j's own unit inputs."""
        torch = _torch()
        from .absint import FloatErrorInterpreter
        st = self.st
        helper = FloatErrorInterpreter()
        helper._cur, helper._cfg = st, self.cfg
        g, nodes, raw_ins, produced = helper._raw_unit(j, u, st)
        canon = st.canon[j]
        by_origin = {}
        for idx, b in enumerate(u.inputs):
            cv = canon.values[b.value_1 if j == 1 else b.value_2]
            if cv.origin is not None:
                by_origin.setdefault(cv.origin, idx)
        c0, rng, r = {}, {}, {}
        for vid in raw_ins:
            v = g.values[vid]
            if v.kind == "const":
                self._const_value(v, vid, c0, rng, {})
                r[vid] = 0.0
                continue
            idx = by_origin.get(v.origin)
            if idx is None:
                free = [i for i in range(len(u.inputs)) if u.inputs[i].kind != "const"]
                if len(free) == 1:
                    idx = free[0]
                else:
                    raise _unsupported_op(f"unit {u.id}: cannot bind raw input {vid}")
            cv = u.inputs[idx].value_1
            lo, hi = self._range_side(cv, j)
            c0[vid] = self.c0[cv]
            rng[vid] = (lo, hi)
            r[vid] = 0.0
        for nid in nodes:
            n = g.nodes[nid]
            ins_c = [c0.get(i) if i is not None else None for i in n.inputs]
            ins_r = [rng.get(i) if i is not None else None for i in n.inputs]
            o = n.outputs[0]
            c = self.eval64(g, n, ins_c)
            lo, hi = self.ibp(g, n, ins_r, ins_c, c)
            c0[o] = c
            if np.dtype(g.values[o].dtype).kind != "f":
                rng[o] = (lo, hi)
                r[o] = 0.0
                continue
            prop = self._prop_side(g, n, ins_r, r, ins_c, (lo, hi))
            loc = self._rho(g, n, ins_r, (lo, hi), ins_c, side=j)
            tot = prop + loc
            rng[o] = (lo - tot, hi + tot)
            r[o] = tot
        out = {}
        inside = set(nodes)
        boundary = [o for nid in nodes for o in g.nodes[nid].outputs
                    if o in g.outputs or any(c.id not in inside for c in g.consumers(o))]
        for b in u.outputs:
            cv = canon.values[b.value_1 if j == 1 else b.value_2]
            rv = st.by_origin[j].get(cv.origin) if cv.origin is not None else None
            if rv is None or rv not in r:
                if len(u.outputs) == 1 and len(boundary) == 1:
                    rv = boundary[0]
                elif rv is None and len(nodes) == 0:
                    rv = None
                else:
                    raise _unsupported_op(f"unit {u.id}: output {b.value_1} not found in "
                                          f"program {j}")
            shp = tuple(self.k.values[b.value_1].shape)
            val = r.get(rv, 0.0) if rv is not None else 0.0
            if not torch.is_tensor(val):
                val = torch.full(shp, float(val), dtype=torch.float64, device=self.dev)
            out[b.value_1] = val.reshape(shp) if tuple(val.shape) != shp else val
        return out

    def _range_side(self, vid, j):
        B = self.bnd.get(vid)
        if B is None:
            return self.rng[vid]
        c = self.c0[vid]
        s = B[KV] + B[K64] + (B[KE1] if j == 1 else B[KE2])
        return c - s, c + s

    def _prop_side(self, g, n, ins_r, r, ins_c, out_r):
        """Forward propagation of the within-unit rounding of earlier operations."""
        torch = _torch()
        vals = [r.get(i, 0.0) if i is not None else 0.0 for i in n.inputs]
        if all((not torch.is_tensor(v)) and v == 0.0 for v in vals):
            return 0.0
        b = []
        for v, i in zip(vals, n.inputs):
            if i is None or np.dtype(g.values[i].dtype).kind != "f":
                b.append(None)
                continue
            shp = tuple(g.values[i].shape)
            if not torch.is_tensor(v):
                v = torch.full(shp, float(v), dtype=torch.float64, device=self.dev)
            B = torch.zeros((5,) + shp, dtype=torch.float64, device=self.dev)
            B[KD] = v
            b.append(B)
        return self.fwd(g, n, b, ins_r, out_r, ins_c)[KD]

    def _ref_rho64(self, u):
        """Rounding of the float64 reference evaluation (any order) at the unit outputs."""
        torch = _torch()
        k = self.k
        f64 = fpm.FORMATS["float64"]
        r = {}
        for b in u.inputs:
            r[b.value_1] = 0.0
        for nid in u.canonical_nodes_1:
            n = k.nodes[nid]
            ins_r = [self._range_any(i) if i is not None else None for i in n.inputs]
            ins_c = [self.c0.get(i) if i is not None else None for i in n.inputs]
            o = n.outputs[0]
            if np.dtype(k.values[o].dtype).kind != "f":
                r[o] = 0.0
                continue
            lo, hi = self.rng[o]
            prop = self._prop_side(k, n, ins_r, r, ins_c, (lo, hi))
            loc = self._rho(k, n, ins_r, (lo, hi), ins_c, side=0, fmt=f64)
            r[o] = prop + loc
        out = {}
        for b in u.outputs:
            v = r.get(b.value_1, 0.0)
            shp = tuple(k.values[b.value_1].shape)
            out[b.value_1] = v if torch.is_tensor(v) else torch.full(
                shp, float(v), dtype=torch.float64, device=self.dev)
        self.assumptions.add("float64 reference: torch float64 kernels, any summation order, "
                             "elementary functions within the ULP table")
        return out

    # ---- local rounding of one operation, per element ----

    def _rho(self, g, n, ins_r, out_r, ins_c, side, fmt=None):
        torch = _torch()
        op, a = n.op, n.attrs
        out = g.values[n.outputs[0]]
        if fmt is None:
            if out.dtype not in fpm.FORMATS:
                raise _unsupported_path(f"{op}: no rounding model for {out.dtype}")
            fmt = fpm.FORMATS[out.dtype]
        u, eta = fmt.u, fmt.eta
        lo, hi = out_r
        Y = _mag(lo, hi)
        ulp = lambda y: 2 * u * y + 2 * eta  # noqa: E731
        if op in EXACT_OPS or op in ("reshape", "permute", "broadcast_to", "slice", "concat",
                                      "pad", "gather"):
            return 0.0
        if op == "cast":
            src = g.values[n.inputs[0]].dtype
            sf = fpm.FORMATS.get(src)
            if sf is not None and sf.p <= fmt.p:
                return 0.0
            if sf is None and float(Y.max()) <= 2.0 ** fmt.p:
                return 0.0
            return u * Y
        if op in ("add", "sub"):
            if len(n.inputs) != 2:
                k_ = len(n.inputs)
                return (k_ - 1) * u * (1 + fmt.gamma(k_)) * \
                    sum(_mag(*r) for r in ins_r)
            return u * Y
        if op == "mul":
            if len(n.inputs) != 2:
                return (len(n.inputs) - 1) * (u * Y * (1 + fmt.gamma(len(n.inputs))) + eta)
            return u * Y + eta
        if op == "div":
            return self._ulp("div") * ulp(Y)
        if op == "pow":
            return self._ulp("pow") * ulp(Y)
        if op in ("exp", "log", "tanh", "erf", "sqrt", "rsqrt", "reciprocal", "sin", "cos"):
            return self._ulp(op) * ulp(Y)
        if op == "sigmoid":
            q = self._ulp("exp")
            return (2 * q + 3.02) * u * Y + fmt.tiny
        if op == "silu":
            q = self._ulp("exp")
            return (2 * q + 4.02) * u * Y + fmt.tiny
        if op == "gelu":
            xl, xh = ins_r[0]
            X, m = _mag(xl, xh), _min_abs(xl, xh)
            tail = X * u + 2.02 * u * Y + 6 * eta
            if a.get("approximate") == "tanh":
                q = self._ulp("tanh")
                c, a_ = SQRT_2_OVER_PI, 0.044715
                inc = 0.5 * X * 5.05 * u * c * (X + a_ * X ** 3)
                dec = torch.clamp(4 * torch.exp(-2 * c * (m + a_ * m ** 3) * 0.999), max=1.0)
                return (inc * dec + 0.5 * X * q * (2 * u + 2 * eta) + tail) * fpm.SAFETY
            q = self._ulp("erf")
            arg = (1 / SQRT_PI) * INV_SQRT2 * u * (2 / math.e) * 1.01
            return (arg + 0.5 * X * q * (2 * u + 2 * eta) + tail) * fpm.SAFETY
        if op in ("matmul", "conv", "reduce_sum", "reduce_mean", "avg_pool"):
            return self._rho_accum(g, n, ins_r, out_r, side, fmt)
        if op in ("softmax", "log_softmax"):
            return self._rho_softmax(n, ins_r[0], out_r, fmt)
        if op in ("layer_norm", "rms_norm"):
            return self._rho_norm(n, ins_r, ins_c, fmt)
        if op == "batch_norm":
            x = _mag(*ins_r[0])
            w, b, mean, var = ins_c[1:5]
            shp = (1, -1) + (1,) * (x.dim() - 2)
            s = (w / torch.sqrt(var + a["eps"])).abs().reshape(shp)
            q = self._ulp("rsqrt")
            mag = x * s + (mean.reshape(shp) * s).abs() + b.abs().reshape(shp)
            return ((2 * q + 6) * u * mag + 4 * eta) * fpm.SAFETY
        raise _unsupported_op(f"no rounding rule for {op}")

    def _rho_softmax(self, n, xr, out_r, fmt):
        torch = _torch()
        u, eta = fmt.u, fmt.eta
        ax = n.attrs["axis"]
        xl, xh = xr
        nt = xl.shape[ax]
        q = self._ulp("exp")
        m_hi = xh.amax(dim=ax, keepdim=True)
        m_lo = xl.amax(dim=ax, keepdim=True)
        d_lo, d_hi = xl - m_hi, xh - m_lo                        # x_i - max
        under = d_hi < EXP_UNDERFLOW
        dmag = torch.where(under, torch.zeros_like(d_lo), d_lo.abs().maximum(d_hi.abs()))
        e_i = (1 + 2 * q * u) * torch.exp(u * dmag) - 1           # relative, per term
        tiny_abs = torch.where(under, torch.exp(d_hi.clamp(max=0)) + 2 * q * eta,
                               torch.full_like(d_hi, 2 * q * eta))
        e_max = e_i.amax(dim=ax, keepdim=True)
        g_n = fmt.gamma(nt)
        e_S = (1 + e_max) * (1 + (nt - 1) * u * (1 + g_n)) - 1 + \
            tiny_abs.sum(dim=ax, keepdim=True) / (1 - e_max)
        if bool((e_S >= 0.5).any()):
            raise InconclusiveError(Inconclusive.BOUND_ABOVE_EPSILON,
                                    "softmax rounding bound diverges (input range too wide)")
        self.assumptions.add("softmax evaluated as exp(x - max) / sum, any summation order")
        yl, yh = out_r
        if n.op == "softmax":
            e_y = (1 + e_i) * (1 + u) ** 2 / (1 - e_S) - 1
            return (e_y * yh + tiny_abs + 2 * eta) * fpm.SAFETY
        ln = math.log(nt)
        ql = self._ulp("log")
        return (u * dmag - torch.log1p(-e_S) + ql * (2 * u * (ln + 1) + 2 * eta)
                + u * (dmag + ln) * 1.01 + eta) * fpm.SAFETY

    def _rho_norm(self, n, ins_r, ins_c, fmt):
        torch = _torch()
        u, eta = fmt.u, fmt.eta
        a = n.attrs
        axes = a["axes"]
        xl, xh = ins_r[0]
        nt = int(np.prod([xl.shape[d] for d in axes]))
        eps = float(a["eps"])
        q = self._ulp("rsqrt")
        g_n = fmt.gamma(nt)
        tree = lambda leaf: u * (1 + g_n) * leaf * (nt * (nt + 1) / 2 - 1)  # noqa: E731
        _, _, sl, sh, zl, zh = self._norm_ibp(n, [xl], [xh], [None] + ins_c[1:])
        X = _mag(xl, xh).amax(dim=axes, keepdim=True)
        z = _mag(zl, zh)
        w = ins_c[1].abs() if len(ins_c) > 1 and ins_c[1] is not None else 1.0
        b = ins_c[2].abs() if len(ins_c) > 2 and ins_c[2] is not None else 0.0
        var_lo = sl ** 2 - eps                                      # lower bound of var / ms
        denom = var_lo.clamp(min=0) + eps
        if n.op == "layer_norm":
            width = (xh.amax(dim=axes, keepdim=True) - xl.amin(dim=axes, keepdim=True))
            d_mu = tree(X) / nt + 2 * u * X + eta
            D = width + d_mu
            V = sh ** 2 - eps
            S_tree = tree(D * D * (1 + 3 * u))
            e_var = d_mu ** 2 + 3.01 * u * D * D + S_tree / nt
            e_var = e_var + 2 * u * (V + e_var) + eta
            rho_a = (e_var + u * (V + e_var + eps)) / denom
            self.assumptions.add("layer_norm: two-pass or Welford variance (bounded as two-pass)"
                                 ", rsqrt(var + eps), any summation order")
        else:
            d_mu = torch.zeros_like(X)
            rel = (1 + u) * (1 + (nt - 1) * u * (1 + g_n)) * (1 + 2 * u) - 1
            rho_a = torch.full_like(X, rel + u)
            self.assumptions.add("rms_norm: x * rsqrt(mean(x^2) + eps), any summation order")
        if bool((rho_a >= 0.5).any()):
            raise InconclusiveError(Inconclusive.BOUND_ABOVE_EPSILON,
                                    f"{n.op} rounding bound diverges (variance too small "
                                    "relative to the rounding of its computation)")
        se = torch.sqrt(denom)
        e_r = (1 + 2 * q * u) / torch.sqrt(1 - rho_a) - 1
        e_y = d_mu / se * (1 + e_r) + z * e_r + 2.01 * u * (z + d_mu / se) * (1 + e_r) + 2 * eta
        return (w * e_y + u * (2 * (z + e_y) * w + b) * 1.01 + 2 * eta) * fpm.SAFETY

    # ---- accumulations per element ----

    def _rho_accum(self, g, n, ins_r, out_r, side, fmt):
        torch = _torch()
        op, a = n.op, n.attrs
        u, eta = fmt.u, fmt.eta
        if side == 0:
            # float64 reference: any order; every node weight is at most the total, so
            # sum_k M_k <= (K - 1) W, plus leaf roundings. No sort needed at u = 2^-53.
            if op == "matmul":
                X = self._matmul_leaf_fn(g, n, ins_r)
            elif op == "conv":
                X = self._conv_leaf_fn(g, n, ins_r)
            else:
                X = self._reduce_leaf_fn(g, n, ins_r)
            K = X("count")
            W = X("sum")
            rho = u * (1 + fmt.gamma(K + 1)) * (K * W + K * K * eta) + 2 * K * eta
            if op in ("reduce_mean", "avg_pool"):
                rho = rho / K * (1 + u) + 2.01 * u * _mag(*out_r) + eta
            return rho * fpm.SAFETY
        else:
            order = self._order(side, n, g) if op in ("matmul", "conv") else \
                self.oracle.order(self._impl(side, n)[0], self._impl(side, n)[1], "reduce_sum",
                                  self._nterms(g, n))
        products = op in ("matmul", "conv")
        bias = self._bias_leaf(g, n) if side != 0 else None
        if op == "matmul":
            X = self._matmul_leaf_fn(g, n, ins_r)
        elif op == "conv":
            X = self._conv_leaf_fn(g, n, ins_r)
        else:
            X = self._reduce_leaf_fn(g, n, ins_r)
        rho = self._order_sigma(order, X, fmt, products) if self.prob else \
            self._order_rho(order, X, fmt, products)
        if bias is not None and not isinstance(order, (fpm.Known, fpm.Chunks)):
            # The bias may be a leaf anywhere in the tree: it lies under at most K nodes, and
            # its parent repeats the weight of its sibling subtree (at most the total W).
            K = X("count")
            rho = rho + u * (1 + fmt.gamma(K + 1)) * (K * bias + X("sum"))
        if op in ("reduce_mean", "avg_pool"):
            nterm = X("count")
            Y = _mag(*out_r)
            rho = rho / nterm * (1 + u) + 2.01 * u * Y + eta
        return rho * fpm.SAFETY

    def _nterms(self, g, n):
        shp = g.values[n.inputs[0]].shape
        if n.op == "avg_pool":
            return int(np.prod(n.attrs["kernel"]))
        axes = n.attrs.get("axes") or tuple(range(len(shp)))
        return int(np.prod([shp[d] for d in axes]))

    def _bias_leaf(self, g, n):
        """|bias| per output element when a bias add from the same source op follows."""
        torch = _torch()
        out = n.outputs[0]
        cons = g.consumers(out)
        if len(cons) != 1 or cons[0].op != "add":
            return None
        c = cons[0]
        if set(c.meta.get("origins", ())) != set(n.meta.get("origins", ())):
            return None
        other = c.inputs[1] if c.inputs[0] == out else c.inputs[0]
        if other is None or not g.is_const(other):
            return None
        b = torch.as_tensor(np.abs(np.asarray(g.values[other].data, dtype=np.float64)),
                            device=self.dev)
        return b.expand(tuple(g.values[c.outputs[0]].shape)) if b.dim() else b

    @staticmethod
    def _leaf_stats(L, ranges):
        """Second-moment statistics of leaf magnitudes L [rows, K, cols] along K:
        ssq   sum_{j=1}^{K-1} (W - P_{j-1})^2 over ascending sorted leaves: by the lemma, the
              largest sum of squared node weights of any tree (squares are monotone);
        seqsq sum_{m=1}^{K} P_m^2 in natural order (a sequential chain from 0);
        sumsq sum_i X_i^2; and per chunk range (s, e): (sum P_m^2 within it, chunk sum)."""
        torch = _torch()
        K = L.shape[1]
        out = {}
        s_, _ = torch.sort(L, dim=1)
        P = torch.cumsum(s_, dim=1)
        W = P[:, -1:]
        prev = torch.cat([torch.zeros_like(P[:, :1]), P[:, :K - 2]], dim=1) if K > 1 else P[:, :0]
        out["ssq"] = ((W - prev) ** 2).sum(1)
        Pn = torch.cumsum(L, dim=1)
        out["seqsq"] = (Pn ** 2).sum(1)
        out["sumsq"] = (L ** 2).sum(1)
        for (a, b) in ranges:
            if b <= a:
                continue
            Pr = torch.cumsum(L[:, a:b], dim=1)
            out[("rsq", a, b)] = (Pr ** 2).sum(1)
            out[("rw", a, b)] = Pr[:, -1]
        return out

    # ---- cancellation-aware node magnitudes --------------------------------------------- #
    #
    # A node of a summation tree rounds the exact sum z of its two computed operands, and
    # |fl(z) - z| <= half an ulp of z = u * pow2floor(|z|). With s the exact sum of the leaves
    # under the node and W the sum of their magnitudes,
    #
    #     |z| <= |s| + (rounding accumulated below the node) <= |s| + gamma * W.
    #
    # The plain bound replaces |s| by W, which ignores that leaves of opposite sign cancel.
    # Here |s| is bounded from the leaf *intervals* [lo_i, hi_i]:
    #   known sequential chain: s is a prefix sum, so |s| <= max(|sum lo|, |sum hi|);
    #   any tree: the leaves under a node are an unknown subset, so
    #       |s| <= min(W_node, G),  G = max(sum_i max(hi_i, 0), sum_i max(-lo_i, 0)),
    #   and by the lemma the j-th largest W_node of any tree is at most the sum of the
    #   K - j + 1 largest leaf magnitudes. u * pow2floor(min(w, G) + gamma w) is nondecreasing
    #   in w, so the sum over the nodes is largest for that tree.

    @staticmethod
    def _pow2floor(M):
        """Largest power of two <= M, exactly (0 where M <= 0)."""
        torch = _torch()
        _, ex = torch.frexp(M)
        return torch.where(M > 0, torch.ldexp(torch.ones_like(M), ex - 1), torch.zeros_like(M))

    def _signed_core(self, lo, hi, spec):
        """Sums over the tree nodes of pow2floor(node magnitude bound) for leaf intervals
        lo, hi [rows, K, cols] (summed along K). `spec` = (mode, zero_init, gamma, keta,
        ranges). Returns per (rows, cols):
          "leaf"  sum_i pow2floor(|x_i|)                   (separately rounded products)
          "seq"   sequential chain in natural order        (mode "seq")
          "any"   the largest sum over all binary trees    (mode "any")
          "tree"  a known binary tree                      (mode ("tree", SumTree))
          per range (a, b): ("seq", a, b), ("leaf", a, b) for a chain from 0 over leaves
          a..b-1, and its exact sum's enclosure ("lo", a, b), ("hi", a, b), ("abs", a, b).
        gamma multiplies the magnitude sum W of the node (accumulated rounding below it, plus
        the rounding of this float64 evaluation); keta is added to every node magnitude."""
        torch = _torch()
        mode, zero, g_, keta, ranges = spec
        K = lo.shape[1]
        A = lo.abs().maximum(hi.abs())
        up = 1 + 2.0 ** -40
        p2 = lambda M: self._pow2floor(M * up)       # noqa: E731
        sq = bool(self.prob)                           # second moments (probabilistic model)
        la = p2(A)
        out = {"leaf": la.sum(1)}
        if sq:
            out["leaf2"] = (la * la).sum(1)
        del la
        if mode == "seq":
            PA = torch.cumsum(A, dim=1)
            M = _mag(torch.cumsum(lo, dim=1), torch.cumsum(hi, dim=1))
            M = torch.minimum(M, PA) + keta + g_ * (PA + keta)
            M = p2(M if zero else M[:, 1:])
            out["seq"] = M.sum(1)
            if sq:
                out["seq2"] = (M * M).sum(1)
        elif mode == "any":
            G = torch.maximum(hi.clamp(min=0).sum(1, keepdim=True),
                              (-lo).clamp(min=0).sum(1, keepdim=True))
            top = torch.cumsum(torch.sort(A, dim=1, descending=True).values, dim=1)[:, 1:]
            M = torch.minimum(top, G * (1 + K * 2.0 ** -50)) + keta + g_ * (top + keta)
            M = p2(M)
            out["any"] = M.sum(1)
            if sq:
                out["any2"] = (M * M).sum(1)
        elif isinstance(mode, tuple) and mode[0] == "tree":
            # a known binary tree: the exact sum under a node is the sum of its leaves'
            # intervals, computed bottom-up level by level
            t = mode[1]
            nt, T = t.n_total, t.n_total + len(t.children)

            def full(v):
                z = torch.zeros((v.shape[0], T, v.shape[2]), dtype=v.dtype, device=v.device)
                z[:, :K] = v
                return z
            NL, NH, NA = full(lo), full(hi), full(A)
            for nodes, left, right in t.levels():
                nodes, left, right = [torch.as_tensor(i, device=lo.device)
                                      for i in (nodes, left, right)]
                NL[:, nodes] = NL[:, left] + NL[:, right]
                NH[:, nodes] = NH[:, left] + NH[:, right]
                NA[:, nodes] = NA[:, left] + NA[:, right]
            M = torch.minimum(_mag(NL[:, nt:], NH[:, nt:]), NA[:, nt:]) + keta \
                + g_ * (NA[:, nt:] + keta)
            M = p2(M)
            out["tree"] = M.sum(1)
            if sq:
                out["tree2"] = (M * M).sum(1)
            del NL, NH, NA
        for (a, b) in ranges:
            if b <= a:
                continue
            PA = torch.cumsum(A[:, a:b], dim=1)
            PL, PH = torch.cumsum(lo[:, a:b], dim=1), torch.cumsum(hi[:, a:b], dim=1)
            M = torch.minimum(_mag(PL, PH), PA) + keta + g_ * (PA + keta)
            M = p2(M)
            out[("seq", a, b)] = M.sum(1)
            la = p2(A[:, a:b])
            out[("leaf", a, b)] = la.sum(1)
            if sq:
                out[("seq2", a, b)] = (M * M).sum(1)
                out[("leaf2", a, b)] = (la * la).sum(1)
            out[("lo", a, b)], out[("hi", a, b)], out[("abs", a, b)] = \
                PL[:, -1], PH[:, -1], PA[:, -1]
        return out

    def _mm_signed(self, al, ah, bl, bh, spec):
        """_signed_core over the leaves A[..., m, k] * B[..., k, n] of a matmul."""
        torch = _torch()
        K = al.shape[-1]
        lead = torch.broadcast_shapes(al.shape[:-2], bl.shape[:-2])
        M, N = al.shape[-2], bl.shape[-1]
        al, ah = [t.expand(lead + (M, K)).reshape(-1, M, K) for t in (al, ah)]
        bl, bh = [t.expand(lead + (K, N)).reshape(-1, K, N) for t in (bl, bh)]
        a_point, b_point = torch.equal(al, ah), torch.equal(bl, bh)
        res = {}
        chunk = max(1, int(2e7 // max(1, K * N)))
        for bi in range(al.shape[0]):
            for m0 in range(0, M, chunk):
                xl, xh = al[bi, m0:m0 + chunk, :, None], ah[bi, m0:m0 + chunk, :, None]
                yl, yh = bl[bi, None, :, :], bh[bi, None, :, :]
                if a_point or b_point:
                    p1, p2 = (xl * yl, xl * yh) if a_point else (xl * yl, xh * yl)
                    lo, hi = torch.minimum(p1, p2), torch.maximum(p1, p2)
                else:
                    p1, p2, p3, p4 = xl * yl, xl * yh, xh * yl, xh * yh
                    lo = torch.minimum(torch.minimum(p1, p2), torch.minimum(p3, p4))
                    hi = torch.maximum(torch.maximum(p1, p2), torch.maximum(p3, p4))
                    del p3, p4
                del p1, p2
                for k_, v in self._signed_core(lo, hi, spec).items():
                    if k_ not in res:
                        res[k_] = torch.empty((al.shape[0], M, N), dtype=torch.float64,
                                              device=self.dev)
                    res[k_][bi, m0:m0 + chunk] = v
        return {k_: v.reshape(lead + (M, N)) for k_, v in res.items()}

    # Leaf functions: X(kind) returns
    #   "count"  -> number of leaves K
    #   "sum"    -> sum_i X_i per output element
    #   "ramp"   -> sum_i (K - i) X_i per output element (natural order, i = 0..K-1)
    #   ("range", a, b) -> (sum_{a<=i<b} X_i, sum_{a<=i<b} (b - i) X_i) per output element
    #   "sorted_weighted" -> sum_i s_(i) min(i, K-1) over ascending sorted leaves (1-based)
    #   "max2" -> max(X_0, X_1) (for an FMA over two products)

    def _matmul_leaf_fn(self, g, n, ins_r):
        torch = _torch()
        (al, ah), (bl, bh) = ins_r
        A, B = _mag(al, ah), _mag(bl, bh)
        K = A.shape[-1]
        A1 = A.dim() == 1
        B1 = B.dim() == 1
        Am = A.unsqueeze(0) if A1 else A
        Bm = B.unsqueeze(-1) if B1 else B
        out_shape = tuple(g.values[n.outputs[0]].shape)
        ramp = torch.arange(K, 0, -1, dtype=torch.float64, device=self.dev)

        def fix(t):
            return t.reshape(out_shape)

        def f(kind):
            if kind == "count":
                return K
            if kind == "sum":
                return fix(torch.matmul(Am, Bm))
            if kind == "ramp":
                return fix(torch.matmul(Am * ramp, Bm))
            if isinstance(kind, tuple) and kind[0] == "range":
                _, s, e = kind
                sub_r = torch.arange(e - s, 0, -1, dtype=torch.float64, device=self.dev)
                return fix(torch.matmul(Am[..., s:e], Bm[..., s:e, :])), \
                    fix(torch.matmul(Am[..., s:e] * sub_r, Bm[..., s:e, :]))
            if kind == "max2":
                return fix(torch.maximum(Am[..., :, :1] * Bm[..., :1, :],
                                         Am[..., :, 1:2] * Bm[..., 1:2, :]))
            if kind == "sorted_weighted":
                return fix(self._sorted_weighted_mm(Am, Bm))
            if isinstance(kind, tuple) and kind[0] == "stats":
                return {k_: fix(v) for k_, v in self._mm_stats(Am, Bm, kind[1]).items()}
            if isinstance(kind, tuple) and kind[0] == "signed":
                lead1 = lambda t: t.unsqueeze(0) if A1 else t        # noqa: E731
                last1 = lambda t: t.unsqueeze(-1) if B1 else t       # noqa: E731
                st = self._mm_signed(lead1(al), lead1(ah), last1(bl), last1(bh), kind[1])
                return {k_: fix(v) for k_, v in st.items()}
            raise ValueError(kind)
        return f

    def _mm_stats(self, Am, Bm, ranges):
        torch = _torch()
        K = Am.shape[-1]
        lead = torch.broadcast_shapes(Am.shape[:-2], Bm.shape[:-2])
        Am = Am.expand(lead + Am.shape[-2:])
        Bm = Bm.expand(lead + Bm.shape[-2:])
        M, N = Am.shape[-2], Bm.shape[-1]
        Af = Am.reshape(-1, M, K)
        Bf = Bm.reshape(-1, K, N)
        res = {}
        chunk = max(1, int(5e7 // max(1, K * N)))
        for bi in range(Af.shape[0]):
            for m0 in range(0, M, chunk):
                leaves = Af[bi, m0:m0 + chunk, :, None] * Bf[bi, None, :, :]
                st = self._leaf_stats(leaves, ranges)
                for k_, v in st.items():
                    if k_ not in res:
                        res[k_] = torch.empty(lead + (M, N), dtype=torch.float64,
                                              device=self.dev).reshape(-1, M, N)
                    res[k_][bi, m0:m0 + chunk] = v
        return {k_: v.reshape(lead + (M, N)) for k_, v in res.items()}

    def _sorted_weighted_mm(self, Am, Bm):
        """sum over sorted leaves of the products A[..., m, i] * B[..., i, n]."""
        torch = _torch()
        K = Am.shape[-1]
        coef = torch.arange(1, K + 1, dtype=torch.float64, device=self.dev).clamp(max=K - 1)
        lead = torch.broadcast_shapes(Am.shape[:-2], Bm.shape[:-2])
        Am = Am.expand(lead + Am.shape[-2:])
        Bm = Bm.expand(lead + Bm.shape[-2:])
        M, N = Am.shape[-2], Bm.shape[-1]
        out = torch.empty(lead + (M, N), dtype=torch.float64, device=self.dev)
        Af = Am.reshape(-1, M, K)
        Bf = Bm.reshape(-1, K, N)
        of = out.reshape(-1, M, N)
        chunk = max(1, int(2e8 // max(1, K * N)))
        for bi in range(Af.shape[0]):
            for m0 in range(0, M, chunk):
                leaves = Af[bi, m0:m0 + chunk, :, None] * Bf[bi, None, :, :]   # [m, K, N]
                s, _ = torch.sort(leaves, dim=1)
                of[bi, m0:m0 + chunk] = (s * coef[None, :, None]).sum(1)
        return out

    def _conv_leaf_fn(self, g, n, ins_r):
        torch = _torch()
        F = torch.nn.functional
        a = n.attrs
        (xl, xh), (wl, wh) = ins_r
        Xm, Wm = _mag(xl, xh), _mag(wl, wh)
        if not torch.equal(wl, wh):
            raise _unsupported_op("convolution with a variable kernel")
        K = int(np.prod(Wm.shape[1:]))
        ramp = torch.arange(K, 0, -1, dtype=torch.float64, device=self.dev).reshape(Wm.shape[1:])

        def f(kind):
            if kind == "count":
                return K
            if kind == "sum":
                return self._conv(Xm, Wm, a)
            if kind == "ramp":
                return self._conv(Xm, Wm * ramp, a)
            if isinstance(kind, tuple) and kind[0] == "range":
                _, s, e = kind
                flat = Wm.reshape(Wm.shape[0], -1)
                mask = torch.zeros_like(flat)
                mask[:, s:e] = 1
                rr = torch.zeros(K, dtype=torch.float64, device=self.dev)
                rr[s:e] = torch.arange(e - s, 0, -1, dtype=torch.float64, device=self.dev)
                return self._conv(Xm, (flat * mask).reshape(Wm.shape), a), \
                    self._conv(Xm, (flat * rr).reshape(Wm.shape), a)
            if kind == "max2":
                flat = Wm.reshape(Wm.shape[0], -1)
                m0 = torch.zeros_like(flat)
                m0[:, 0] = 1
                m1 = torch.zeros_like(flat)
                m1[:, 1] = 1
                return torch.maximum(self._conv(Xm, (flat * m0).reshape(Wm.shape), a),
                                     self._conv(Xm, (flat * m1).reshape(Wm.shape), a))
            if kind == "sorted_weighted":
                return self._sorted_weighted_conv(Xm, Wm, a)
            if isinstance(kind, tuple) and kind[0] == "stats":
                return self._sorted_weighted_conv(Xm, Wm, a, stats_ranges=kind[1])
            if isinstance(kind, tuple) and kind[0] == "signed":
                return self._sorted_weighted_conv(Xm, Wm, a, signed=(xl, xh, wl, kind[1]))
            raise ValueError(kind)
        return f

    def _sorted_weighted_conv(self, Xm, Wm, a, stats_ranges=None, signed=None):
        torch = _torch()
        F = torch.nn.functional
        if signed is not None:
            xl, xh, w, spec = signed
            if xl.dim() != 4:
                xl, xh, w = xl.unsqueeze(2), xh.unsqueeze(2), w.unsqueeze(2)
        if Xm.dim() != 4:
            Xm, Wm = Xm.unsqueeze(2), Wm.unsqueeze(2)
            a = dict(a, stride=(1,) + tuple(a["stride"]), dilation=(1,) + tuple(a["dilation"]),
                     pads=(0, a["pads"][0], 0, a["pads"][1]))
            squeeze = True
        else:
            squeeze = False
        nsp = 2
        pads = a["pads"]
        b, e = pads[:nsp], pads[nsp:]
        xp = F.pad(Xm, (b[1], e[1], b[0], e[0]))
        G = a["groups"]
        N, C, H, W = xp.shape
        O = Wm.shape[0]
        kh, kw = Wm.shape[2:]
        cols = F.unfold(xp, (kh, kw), a["dilation"], 0, a["stride"])        # [N, C*kh*kw, L]
        L = cols.shape[-1]
        Cg = C // G
        K = Cg * kh * kw
        cols = cols.reshape(N, G, K, L)
        Wf = Wm.reshape(G, O // G, K)
        coef = torch.arange(1, K + 1, dtype=torch.float64, device=self.dev).clamp(max=K - 1)
        out = torch.empty((N, O, L), dtype=torch.float64, device=self.dev)
        stats = {}
        og = O // G
        chunk_o = max(1, int((2e7 if signed is not None else 5e7) // max(1, K * L)))
        if signed is not None:
            del cols
            pad_ = (b[1], e[1], b[0], e[0])
            cl = F.unfold(F.pad(xl, pad_), (kh, kw), a["dilation"], 0, a["stride"])
            ch = F.unfold(F.pad(xh, pad_), (kh, kw), a["dilation"], 0, a["stride"])
            cl, ch = cl.reshape(N, G, K, L), ch.reshape(N, G, K, L)
            Ws = w.reshape(G, O // G, K)
            stats_ranges = ()
        for ni in range(N):
            for gi in range(G):
                for o0 in range(0, og, chunk_o):
                    if signed is not None:
                        wv = Ws[gi, o0:o0 + chunk_o, :, None]
                        p1, p2 = wv * cl[ni, gi, None, :, :], wv * ch[ni, gi, None, :, :]
                        st_ = self._signed_core(torch.minimum(p1, p2), torch.maximum(p1, p2),
                                                spec)
                        del p1, p2
                        sl_ = slice(gi * og + o0, gi * og + o0 + wv.shape[0])
                        for k_, v in st_.items():
                            if k_ not in stats:
                                stats[k_] = torch.empty((N, O, L), dtype=torch.float64,
                                                        device=self.dev)
                            stats[k_][ni, sl_] = v
                        continue
                    leaves = Wf[gi, o0:o0 + chunk_o, :, None] * cols[ni, gi, None, :, :]
                    sl_ = slice(gi * og + o0, gi * og + o0 + leaves.shape[0])
                    if stats_ranges is None:
                        s, _ = torch.sort(leaves, dim=1)
                        out[ni, sl_] = (s * coef[None, :, None]).sum(1)
                        continue
                    for k_, v in self._leaf_stats(leaves, stats_ranges).items():
                        if k_ not in stats:
                            stats[k_] = torch.empty((N, O, L), dtype=torch.float64,
                                                    device=self.dev)
                        stats[k_][ni, sl_] = v
        oh = (H - (a["dilation"][0] * (kh - 1) + 1)) // a["stride"][0] + 1
        ow = (W - (a["dilation"][1] * (kw - 1) + 1)) // a["stride"][1] + 1

        def shape(t):
            t = t.reshape(N, O, oh, ow)
            return t.squeeze(2) if squeeze else t
        if stats_ranges is not None:
            return {k_: shape(v) for k_, v in stats.items()}
        return shape(out)

    def _reduce_leaf_fn(self, g, n, ins_r):
        torch = _torch()
        F = torch.nn.functional
        a = n.attrs
        xl, xh = ins_r[0]
        Xm = _mag(xl, xh)
        if n.op == "avg_pool":
            p, k_, st, _ = self._pool_args(a)
            if a.get("count_include_pad", True):
                xp = F.pad(Xm, p)
            else:
                xp = F.pad(Xm, p)
            N, C, H, W = xp.shape
            cols = F.unfold(xp.reshape(N * C, 1, H, W), k_, 1, 0, st)   # [NC, K, L]
            out_shape = tuple(g.values[n.outputs[0]].shape)
            Lout = int(np.prod(out_shape[2:]))
            cols = cols[..., :Lout]
            if cols.shape[-1] < Lout:
                raise _unsupported_op("avg_pool with ceil_mode windows")
            leaves = cols.permute(0, 2, 1)                                  # [NC, L, K]
            K = leaves.shape[-1]

            def view(t):
                return t.reshape(out_shape)

            def as_leaves(t):
                tp = F.pad(t, p)
                c_ = F.unfold(tp.reshape(N * C, 1, H, W), k_, 1, 0, st)[..., :Lout]
                return c_.permute(0, 2, 1)
        else:
            axes = a.get("axes") or tuple(range(Xm.dim()))
            keep = [d for d in range(Xm.dim()) if d not in axes]
            perm = keep + list(axes)
            K = int(np.prod([Xm.shape[d] for d in axes]))
            leaves = Xm.permute(*perm).reshape(-1, K)
            out_shape = tuple(g.values[n.outputs[0]].shape)

            def view(t):
                return t.reshape(out_shape)

            def as_leaves(t):
                return t.expand(Xm.shape).permute(*perm).reshape(-1, K)
        coef = torch.arange(1, K + 1, dtype=torch.float64, device=self.dev).clamp(max=K - 1)
        ramp = torch.arange(K, 0, -1, dtype=torch.float64, device=self.dev)

        def f(kind):
            if kind == "count":
                return K
            if kind == "sum":
                return view(leaves.sum(-1))
            if kind == "ramp":
                return view((leaves * ramp).sum(-1))
            if isinstance(kind, tuple) and kind[0] == "range":
                _, s, e = kind
                sub = leaves[..., s:e]
                rr = torch.arange(e - s, 0, -1, dtype=torch.float64, device=self.dev)
                return view(sub.sum(-1)), view((sub * rr).sum(-1))
            if kind == "max2":
                return view(leaves[..., :2].amax(-1))
            if kind == "sorted_weighted":
                s, _ = torch.sort(leaves, dim=-1)
                return view((s * coef).sum(-1))
            if isinstance(kind, tuple) and kind[0] == "stats":
                st = self._leaf_stats(leaves.reshape(-1, K, 1), kind[1])
                return {k_: view(v.reshape(-1)) for k_, v in st.items()}
            if isinstance(kind, tuple) and kind[0] == "signed":
                st = self._signed_core(as_leaves(xl).reshape(-1, K, 1),
                                       as_leaves(xh).reshape(-1, K, 1), kind[1])
                return {k_: view(v.reshape(-1)) for k_, v in st.items()}
            raise ValueError(kind)
        return f

    def _order_sigma(self, order, X, fmt, products):
        """Sub-Gaussian scale of an accumulation's rounding error under the probabilistic
        model: the sum of independent mean-zero node errors |d_k| <= u M_k + eta has
        variance proxy sum_k (u M_k + eta)^2. For AnyTree the lemma bounds sum_k M_k^2 by
        the decreasing-sequential tree (squares are monotone)."""
        torch = _torch()
        u, eta = fmt.u, fmt.eta
        K = X("count")
        if K <= 1:
            return (u * X("sum") + eta) if products else 0.0 * X("sum")
        if getattr(self.cfg, "signed_accumulation", True):
            return self._order_sigma_signed(order, X, fmt, products)
        ranges = tuple(order.ranges) if isinstance(order, fpm.Chunks) else ()
        st = X(("stats", ranges))
        g_ = (1 + fmt.gamma(K + 1)) ** 2
        tail = 2 * K * (u * K * eta * 2 + 2 * eta) ** 2
        if isinstance(order, fpm.Known):
            t = order.tree
            if t.seq_order is None or list(t.seq_order) != list(range(t.n)) or t.n_zero != 1:
                raise _unsupported_path("known non-sequential trees are bounded per tensor "
                                        "only (use method='interval')")
            var = 2 * u * u * g_ * st["seqsq"] + tail
            if products and order.fp != "fma":
                var = var + 2 * u * u * st["sumsq"] + 2 * K * eta * eta
        elif isinstance(order, fpm.Chunks):
            var = 0.0
            W = 0.0
            for (a, b) in order.ranges:
                if b <= a:
                    continue
                var = var + 2 * u * u * g_ * st[("rsq", a, b)]
                W = W + st[("rw", a, b)]
            S = len(order.ranges)
            var = var + 2 * u * u * g_ * (S - 1) * W * W + tail
        else:
            var = 2 * u * u * g_ * st["ssq"] + tail
            if products:
                var = var + 2 * u * u * st["sumsq"] + 2 * K * eta * eta
        return torch.sqrt(var)

    def _order_sigma_signed(self, order, X, fmt, products):
        """_order_sigma with the node magnitudes of `_signed_core`: the variance proxy is
        sum_k (u pow2floor(M_k) + eta)^2 <= 2 u^2 sum_k pow2floor(M_k)^2 + tail."""
        torch = _torch()
        u, eta = fmt.u, fmt.eta
        K = X("count")
        g_ = fmt.gamma(K + 1) + (K + 2) * 2.0 ** -50
        keta = K * eta if products else 0.0
        tail = 2 * K * (u * K * eta * 2 + 2 * eta) ** 2
        if isinstance(order, fpm.Known) and (
                order.tree.seq_order is None
                or list(order.tree.seq_order) != list(range(order.tree.n))):
            st = X(("signed", (("tree", order.tree), True, g_, keta, ())))
            var = 2 * u * u * st["tree2"] + tail
            if products:
                var = var + 2 * u * u * st["leaf2"] + 2 * K * eta * eta
        elif isinstance(order, fpm.Known):
            t = order.tree
            if t.n_zero != 1:
                raise _unsupported_path("known non-sequential trees are bounded per tensor "
                                        "only (use method='interval')")
            st = X(("signed", ("seq", True, g_, keta, ())))
            var = 2 * u * u * st["seq2"] + tail
            if products and order.fp != "fma":
                var = var + 2 * u * u * st["leaf2"] + 2 * K * eta * eta
        elif isinstance(order, fpm.Chunks):
            ranges = tuple((s, e) for s, e in order.ranges if e > s)
            st = X(("signed", ("chunks", True, g_, keta, ranges)))
            var = tail
            mags, pos, neg = [], [], []
            for (s, e), inner in zip(order.ranges, order.inner):
                if e <= s:
                    continue
                var = var + 2 * u * u * st[("seq2", s, e)]
                if products and (not isinstance(inner, fpm.Known) or inner.fp != "fma"):
                    var = var + 2 * u * u * st[("leaf2", s, e)] + 2 * (e - s) * eta * eta
                err = g_ * (st[("abs", s, e)] + keta) + keta
                lo_c, hi_c = st[("lo", s, e)] - err, st[("hi", s, e)] + err
                mags.append(_mag(lo_c, hi_c))
                pos.append(hi_c.clamp(min=0))
                neg.append((-lo_c).clamp(min=0))
            S = len(mags)
            if S > 1:
                A = torch.stack(mags, -1)
                G = torch.maximum(torch.stack(pos, -1).sum(-1, keepdim=True),
                                  torch.stack(neg, -1).sum(-1, keepdim=True))
                top = torch.cumsum(torch.sort(A, dim=-1, descending=True).values, dim=-1)[..., 1:]
                gs = fmt.gamma(S) + S * 2.0 ** -50
                M = self._pow2floor((torch.minimum(top, G * (1 + S * 2.0 ** -50)) + gs * top)
                                    * (1 + 2.0 ** -40))
                var = var + 2 * u * u * (M * M).sum(-1)
        else:
            st = X(("signed", ("any", True, g_, keta, ())))
            var = 2 * u * u * st["any2"] + tail
            if products:
                var = var + 2 * u * u * st["leaf2"] + 2 * K * eta * eta
        return torch.sqrt(var)

    def _order_rho(self, order, X, fmt, products):
        """Per-element rounding bound of an accumulation under an order model."""
        if not getattr(self.cfg, "signed_accumulation", True):
            return self._order_rho_abs(order, X, fmt, products)
        torch = _torch()
        u, eta = fmt.u, fmt.eta
        K = X("count")
        if K <= 1:
            return (u * X("sum") + eta) if products else 0.0 * X("sum")
        # gamma: rounding accumulated below a node, plus the float64 rounding of the leaf
        # products and prefix sums of this analysis
        g_ = fmt.gamma(K + 1) + (K + 2) * 2.0 ** -50
        keta = K * eta if products else 0.0
        if isinstance(order, fpm.Known):
            t = order.tree
            if t.seq_order is None or list(t.seq_order) != list(range(t.n)):
                # a revealed tree (fprev): one half-ulp per node of its own partial sum. Every
                # product is charged its own rounding, which also covers a kernel that fuses
                # some of them into the accumulation (an FMA rounds once, at the node).
                st = X(("signed", (("tree", t), True, g_, keta, ())))
                b = u * st["tree"]
                if products:
                    b = b + len(t.children) * eta + u * st["leaf"] + K * eta
                return b
            zero = t.n_zero == 1
            nodes = K if zero else K - 1
            st = X(("signed", ("seq", zero, g_, keta, ())))
            b = u * st["seq"]
            if products:
                b = b + nodes * eta
                if order.fp != "fma":
                    b = b + u * st["leaf"] + K * eta
                elif not zero:
                    b = b + u * X("max2") + eta
            return b
        if isinstance(order, fpm.Chunks):
            ranges = tuple((s, e) for s, e in order.ranges if e > s)
            st = X(("signed", ("chunks", True, g_, keta, ranges)))
            total = 0.0
            mags, pos, neg = [], [], []
            for (s, e), inner in zip(order.ranges, order.inner):
                if e <= s:
                    continue
                total = total + u * st[("seq", s, e)]
                if products:
                    total = total + (e - s) * eta
                    if not isinstance(inner, fpm.Known) or inner.fp != "fma":
                        total = total + u * st[("leaf", s, e)] + (e - s) * eta
                # the computed chunk sum: its exact sum's enclosure widened by its rounding
                err = g_ * (st[("abs", s, e)] + keta) + keta
                lo_c, hi_c = st[("lo", s, e)] - err, st[("hi", s, e)] + err
                mags.append(_mag(lo_c, hi_c))
                pos.append(hi_c.clamp(min=0))
                neg.append((-lo_c).clamp(min=0))
            S = len(mags)
            if S > 1:
                A = torch.stack(mags, -1)
                G = torch.maximum(torch.stack(pos, -1).sum(-1, keepdim=True),
                                  torch.stack(neg, -1).sum(-1, keepdim=True))
                top = torch.cumsum(torch.sort(A, dim=-1, descending=True).values, dim=-1)[..., 1:]
                gs = fmt.gamma(S) + S * 2.0 ** -50
                M = torch.minimum(top, G * (1 + S * 2.0 ** -50)) + gs * top
                total = total + u * self._pow2floor(M * (1 + 2.0 ** -40)).sum(-1)
            return total
        # AnyTree (and Depth, bounded by AnyTree)
        st = X(("signed", ("any", True, g_, keta, ())))
        if products:
            return u * st["any"] + (K - 1) * eta + u * st["leaf"] + K * eta
        return u * st["any"]

    def _order_rho_abs(self, order, X, fmt, products):
        """The bound without cancellation: every node magnitude is the sum of the leaf
        magnitudes below it (`signed_accumulation=False`)."""
        u, eta = fmt.u, fmt.eta
        K = X("count")
        if K <= 1:
            return (u * X("sum") + eta) if products else 0.0 * X("sum")
        if isinstance(order, fpm.Known):
            t = order.tree
            if t.seq_order is None or list(t.seq_order) != list(range(t.n)):
                raise _unsupported_path("known non-sequential trees are bounded per tensor "
                                        "only (use method='interval')")
            g_ = fmt.gamma(K + 1)
            zero = t.n_zero == 1
            ramp = X("ramp")                       # sum_m P_m with P_m prefix sums (1..K)
            if not zero:
                ramp = ramp - X(("range", 0, 1))[0]   # nodes P_2..P_K
            nodes = K if zero else K - 1
            if products:
                b = u * (1 + g_) * (ramp + nodes * K * eta) + nodes * eta
                if order.fp != "fma":
                    b = b + u * X("sum") + K * eta
                elif not zero:
                    b = b + u * X("max2") + eta
            else:
                b = u * (1 + fmt.gamma(K)) * ramp
            return b
        if isinstance(order, fpm.Chunks):
            total = 0.0
            sums = []
            for (s, e), inner in zip(order.ranges, order.inner):
                if e <= s:
                    continue
                Ws, Rs = X(("range", s, e))
                g_ = fmt.gamma(e - s + 1)
                if products:
                    total = total + u * (1 + g_) * (Rs + (e - s) ** 2 * eta) + (e - s) * eta
                    if not isinstance(inner, fpm.Known) or inner.fp != "fma":
                        total = total + u * Ws + (e - s) * eta
                else:
                    total = total + u * (1 + fmt.gamma(e - s)) * Rs
                sums.append((1 + g_) * (Ws + (e - s) * eta))
            S = len(sums)
            if S > 1:
                import torch
                st = torch.stack(sums, -1)
                s_, _ = torch.sort(st, dim=-1)
                coef = torch.arange(1, S + 1, dtype=torch.float64,
                                    device=st.device).clamp(max=S - 1)
                total = total + u * (1 + fmt.gamma(S)) * (s_ * coef).sum(-1)
            return total
        # AnyTree (and Depth, bounded by AnyTree)
        sw = X("sorted_weighted")
        if products:
            g_ = fmt.gamma(K + 1)
            return u * (1 + g_) * (sw + (K - 1) * K * eta) + (K - 1) * eta + u * X("sum") + K * eta
        return u * (1 + fmt.gamma(K)) * sw
