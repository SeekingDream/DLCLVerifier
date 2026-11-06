"""Abstract interpretation of matched units: interval ranges plus a bound on the difference.

`FloatErrorInterpreter.transfer` maps one `[min_val, max_val, max_error]` triple per unit input
to one per unit output (docs/ABSTRACT_INTERPRETATION.md). For a unit computing the logical
function F, program j runs its own operations on its own inputs x^j:

    |out1 - out2| <= |f1^(x1) - f1(x1)| + |f1(x1) - f2(x2)| + |f2(x2) - f2^(x2)|
                       \\__ R1 (rounding) _/   \\___ P (exact) __/   \\__ R2 (rounding) _/

* P is computed on program 1's canonical nodes in exact real arithmetic (Lipschitz bounds over
  the input ranges). Where a program matched only through an idiom with approximate constants
  (a GELU written with fl32(1/sqrt 2)), the distance to the reference function is added.
* R_j is computed on program j's own operations: a "raw" lowering of its source graph with no
  rewrite that changes rounding (no reassociation, no scalar hoisting, no idiom folding).
  Every operation adds its local rounding bound; earlier errors propagate through Lipschitz
  bounds. Accumulations (matmul, conv, sums) get their bound from the accumulation order of
  that implementation (`OrderOracle`): a known tree (Strategy 1) or every admissible tree
  (Strategy 2). The two sides are bounded independently, so a known order on one side is kept
  even when the other is unknown.
* When both programs run the same operations (isomorphic raw graphs) a joint pass also bounds
  |out1 - out2| operation by operation. It can use cancellation: with bitwise-equal inputs, a
  correctly rounded operation that both programs run on its own gives bitwise-equal results,
  and two known accumulation trees cancel on their shared subtrees. The smaller bound wins.

Ranges always enclose the computed values of both programs, so Lipschitz constants and rounding
magnitudes are taken over sets that contain every point they are applied to.

Anything the model cannot cover raises `InconclusiveError` with an `outcomes.Inconclusive`
reason; the certifier then returns an inconclusive result instead of a bound.
"""
from __future__ import annotations

import collections
import hashlib
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import fpmodel as fpm
from .canonical import COMMUTATIVE, NARY, _attr_key, canonicalize
from .interpretation import ErrorTriple, UnitInterpreter
from .outcomes import Inconclusive

SQRT_PI = math.sqrt(math.pi)
INV_SQRT2 = 1 / math.sqrt(2)
SQRT_2_OVER_PI = math.sqrt(2 / math.pi)
GELU_LIP = 1.1291                 # sup |GELU'| (erf and tanh forms): 1.12900
GELU_MIN = -0.17005               # min GELU (erf: -0.169972 at -0.7518; tanh: -0.170041)
SILU_LIP = 1.0999                 # sup |SiLU'| = 1.09984
SILU_MIN = -0.27847               # min SiLU = -0.278465 at -1.2785
LAYOUT = {"reshape", "permute", "broadcast_to", "slice", "concat", "pad", "gather"}
EXACT_OPS = LAYOUT | {"relu", "neg", "abs", "max", "min", "where", "reduce_max", "reduce_min",
                      "max_pool", "eq", "ne", "lt", "le", "gt", "ge", "and", "or", "not"}
CORRECTLY_ROUNDED = {"add", "sub", "mul", "div", "sqrt", "cast"}
DSL_BACKENDS = {"triton", "tilelang"}
SWAPPED_SOURCE = {"aten.linear", "aten.matmul", "aten.mm", "aten.bmm", "aten.addmm",
                  "aten.conv1d", "aten.conv2d", "aten.convolution"}


def _base_op(source_op: str) -> str:
    """aten.conv2d.default -> aten.conv2d"""
    parts = source_op.split(".")
    return ".".join(parts[:2]) if source_op.startswith("aten.") else source_op


class InconclusiveError(Exception):
    def __init__(self, reason: Inconclusive, detail: str, unit: str = ""):
        super().__init__(f"{reason.value}: {detail}")
        self.reason, self.detail, self.unit = reason, detail, unit


def _unsupported_op(msg):
    return InconclusiveError(Inconclusive.UNSUPPORTED_OPERATOR, msg)


def _unsupported_path(msg):
    return InconclusiveError(Inconclusive.UNSUPPORTED_PATH, msg)


def _exceptional(msg):
    return InconclusiveError(Inconclusive.EXCEPTIONAL_UNRESOLVED, msg)


# --------------------------------------------------------------------------------------------
# Abstract values
# --------------------------------------------------------------------------------------------

@dataclass
class AV:
    """Interval [lo, hi] enclosing the values, and `dev`: a bound on the difference this pass
    tracks (between the programs, or between computed and exact). For booleans dev is 0 or 1
    (may differ)."""
    lo: float
    hi: float
    dev: float
    kind: str = "f"                 # f float | i integer | b bool
    dtype: str = "float32"

    @property
    def mag(self) -> float:
        return max(abs(self.lo), abs(self.hi))


def _kind(dtype: str) -> str:
    d = np.dtype(dtype)
    return {"f": "f", "i": "i", "u": "i", "b": "b"}.get(d.kind, "?")


def _av_const(v) -> AV:
    d = np.asarray(v.data)
    k = _kind(v.dtype)
    if d.size == 0:
        return AV(0.0, 0.0, 0.0, k, v.dtype)
    lo, hi = float(np.min(d)), float(np.max(d))
    if v.exact is not None:
        ex = float(v.exact)
        lo, hi = min(lo, ex), max(hi, ex)
    if not (math.isfinite(lo) and math.isfinite(hi)):
        if k == "f":
            # -inf/inf masks (attention): only used through where/max, keep exact infinities
            lo = -math.inf if not math.isfinite(lo) else lo
            hi = math.inf if not math.isfinite(hi) else hi
    return AV(lo, hi, 0.0, k, v.dtype)


def _hull(*avs) -> Tuple[float, float]:
    return min(a.lo for a in avs), max(a.hi for a in avs)


def _mul_iv(a_lo, a_hi, b_lo, b_hi):
    c = [a_lo * b_lo, a_lo * b_hi, a_hi * b_lo, a_hi * b_hi]
    c = [0.0 if math.isnan(x) else x for x in c]
    return min(c), max(c)


# --------------------------------------------------------------------------------------------
# Handler results
# --------------------------------------------------------------------------------------------

@dataclass
class Res:
    lo: float
    hi: float
    lip: float                                      # propagated difference (no new rounding)
    rho: Dict[int, float] = field(default_factory=dict)    # local rounding bound per side
    rel: Optional[float] = None                     # joint bound when all input devs are 0
    kind: str = "f"
    notes: List[str] = field(default_factory=list)


@dataclass
class Mode:
    exact: bool                                     # P pass: no rounding
    sides: Tuple[int, ...]                          # sides whose rounding is added
    joint: bool = False                             # dev compares the two programs
    nodes: Dict[int, object] = field(default_factory=dict)   # side -> raw node (joint/side)


@dataclass
class UnitReport:
    unit: str
    ops_1: List[str]
    ops_2: List[str]
    outputs: List[dict] = field(default_factory=list)
    accumulations: List[dict] = field(default_factory=list)
    strategy: str = ""


# --------------------------------------------------------------------------------------------
# Per-match state: raw lowerings that keep each program's rounding structure
# --------------------------------------------------------------------------------------------

class _MatchState:
    def __init__(self, match):
        self.match = match
        self.src = {1: match.graph_1, 2: match.graph_2}
        self.canon = {1: match.canonical_1, 2: match.canonical_2}
        self.raw = {j: canonicalize(self.src[j], fold_constants=True, normalize_arith=False,
                                    flatten=False, recognize_idioms=False) for j in (1, 2)}
        self.pos = {j: {nid: i for i, nid in enumerate(self.raw[j].nodes)} for j in (1, 2)}
        self.by_src = {j: collections.defaultdict(list) for j in (1, 2)}
        self.compute_count = {j: collections.Counter() for j in (1, 2)}
        for j in (1, 2):
            for n in self.raw[j].nodes.values():
                for o in n.meta.get("origins", ()):
                    self.by_src[j][o].append(n.id)
                    if n.op not in LAYOUT and n.op != "cast":
                        self.compute_count[j][o] += 1
        self.by_origin = {j: {} for j in (1, 2)}
        for j in (1, 2):
            for vid, v in self.raw[j].values.items():
                if v.origin is not None:
                    self.by_origin[j].setdefault(v.origin, vid)


# --------------------------------------------------------------------------------------------
# Accumulation orders
# --------------------------------------------------------------------------------------------

class OrderOracle:
    """Order model of one implementation's accumulation (docs sections 4.1, 5.1)."""

    def __init__(self, config, assumptions: set):
        self.cfg = config
        self.assumptions = assumptions
        self._seq = {}

    def _sequential(self, n):
        if n not in self._seq:
            self._seq[n] = fpm.SumTree.sequential(n, zero_init=True)
        return self._seq[n]

    def _evidence(self, backend, op, n):
        ev = (getattr(self.cfg, "order_evidence", None) or {}).get((backend, op))
        if ev is None or ev.n != n:
            return None
        if ev.level == "known" or (ev.level == "validated" and
                                   getattr(self.cfg, "accept_validated_orders", False)):
            for item in ev.digest_items():
                self.assumptions.add(item)
            return ev.order()
        return None

    def order(self, impl: dict, source_op: str, op: str, n: int):
        backend = impl.get("backend", "unknown")
        if op in ("matmul", "conv"):
            if backend in DSL_BACKENDS and _base_op(source_op) in SWAPPED_SOURCE and \
                    getattr(self.cfg, "trust_dsl_kernel_order", True):
                return self._dsl_order(backend, impl, n)
            ev = self._evidence(backend, op, n)
            if ev is not None:
                return ev
            if op == "conv" and not getattr(self.cfg, "assume_sum_of_products_conv", False):
                raise _unsupported_path(
                    f"{backend} convolution ({source_op}): the library may select Winograd or "
                    "FFT algorithms, which are not summation trees of the products; set "
                    "assume_sum_of_products_conv=True to assume a direct/implicit-GEMM "
                    "algorithm")
            if op == "conv":
                self.assumptions.add(f"{backend} convolution is a sum of products "
                                     "(direct or implicit GEMM; no Winograd/FFT)")
            return fpm.AnyTree("either")
        ev = self._evidence(backend, op, n)
        return ev if ev is not None else fpm.AnyTree("separate")

    def _dsl_order(self, backend, impl, K):
        S = int(impl.get("split_k", 1) or 1)
        self.assumptions.add(
            f"{backend} GEMM kernel (kernels.py): each output is a sequential fp32 FMA chain "
            "over k from a zero accumulator, read from the kernel source "
            "(Triton tl.dot with input_precision='ieee'; TileLang T.serial FMA loop)")
        if S == 1:
            return fpm.Known(self._sequential(K), "fma", evidence=f"{backend} kernel source")
        if backend == "triton":
            bk = int(impl.get("block_k", 32))
            per = -(-(-(-K // S)) // bk) * bk
        else:
            per = -(-K // S)
        ranges = tuple((s * per, min(K, (s + 1) * per)) for s in range(S))
        inner = tuple(fpm.Known(self._sequential(max(b - a, 1)), "fma") for a, b in ranges)
        self.assumptions.add(f"{backend} split_k partial sums are combined by torch.sum in an "
                             "unknown order")
        return fpm.Chunks(ranges, inner, fpm.AnyTree("separate"))


# --------------------------------------------------------------------------------------------
# The interpreter
# --------------------------------------------------------------------------------------------

class FloatErrorInterpreter(UnitInterpreter):
    """Sound transfer functions for matched units (both accumulation strategies)."""

    def __init__(self):
        self._states: Dict[int, _MatchState] = {}
        self.assumptions: set = set()
        self.reports: List[UnitReport] = []

    # ---- configuration helpers ----

    def _ulp(self, cfg, name: str) -> float:
        table = dict(fpm.CUDA_FP32_ULP)
        table.update(getattr(cfg, "elementary_ulp", None) or {})
        if name not in table:
            raise _unsupported_op(f"no accuracy assumption for {name}; add it to "
                                  "elementary_ulp")
        self.assumptions.add(f"{name} accurate to {table[name]} ulp")
        return float(table[name])

    def _state(self, match) -> _MatchState:
        st = self._states.get(id(match))
        if st is None or st.match is not match:
            st = self._states[id(match)] = _MatchState(match)
            self._check_precision(match)
        return st

    def _check_precision(self, match):
        backends = set()
        for g in (match.graph_1, match.graph_2):
            backends.add(g.meta.get("impl", {}).get("backend", g.frontend))
            for n in g.nodes.values():
                backends.add(n.meta.get("impl", {}).get("backend", g.frontend))
        dev = getattr(self._cfg, "device", None)
        if dev and str(dev).startswith("cuda") and ("eager" in backends or "torch" in backends):
            import torch
            if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
                raise _unsupported_path("TF32 is enabled for eager CUDA matmul/conv; fp32 "
                                        "rounding cannot be assumed")
            self.assumptions.add("eager CUDA: TF32 disabled for matmul and cuDNN (checked)")
        if "onnxruntime_gpu" in backends:
            self.assumptions.add("onnxruntime CUDA EP created with use_tf32=0 "
                                 "(compilers.compile_ort)")
        if "onnx" in backends:
            self.assumptions.add("ONNX program executed in fp32 without TF32 or fast math")
        self.assumptions.add("IEEE 754 binary32/64, round to nearest even, gradual underflow, "
                             "no fast-math contraction except FMA where stated")

    # ---- entry point ----

    def transfer(self, ctx, inputs):
        self._cfg = ctx.config
        st = self._state(ctx.match)
        self._cur = st
        u = ctx.unit
        if getattr(self, "_oracle_cfg", None) is not ctx.config:
            self._oracle = OrderOracle(ctx.config, self.assumptions)
            self._oracle_cfg = ctx.config
        report = UnitReport(u.id, list(u.ops_1), list(u.ops_2))
        try:
            P = self._exact_pass(u, ctx.match, inputs, report)
            R = {j: self._side_pass(j, u, st, inputs, report) for j in (1, 2)}
            J = self._joint_pass(u, st, inputs, report)
        except InconclusiveError as e:
            e.unit = u.id
            raise
        except fpm.UnsupportedFormat as e:
            raise InconclusiveError(Inconclusive.UNSUPPORTED_PATH, str(e), u.id)
        out = []
        for i, b in enumerate(u.outputs):
            p, r1, r2 = P[i], R[1][i], R[2][i]
            e = r1.dev + p.dev + r2.dev
            lo, hi = min(r1.lo, r2.lo), max(r1.hi, r2.hi)
            via = "P+R1+R2"
            if J is not None:
                j = J[i]
                if j.dev < e:
                    e, via = j.dev, "joint"
                lo, hi = max(lo, j.lo), min(hi, j.hi)
            if r1.kind == "b":
                lo, hi, e = 0.0, 1.0, min(e, 1.0)
            self._check_finite(lo, hi, e, b, u)
            report.outputs.append({"value": b.value_1, "P": p.dev, "R1": r1.dev, "R2": r2.dev,
                                   "joint": None if J is None else J[i].dev,
                                   "bound": e, "via": via, "range": [lo, hi]})
            out.append(ErrorTriple(float(lo), float(hi), float(e)))
        report.strategy = ", ".join(sorted({o["via"] for o in report.outputs}))
        self.reports.append(report)
        return out

    def _check_finite(self, lo, hi, e, b, u):
        fmt = fpm.FORMATS.get(b.dtype)
        if math.isnan(e) or math.isnan(lo) or math.isnan(hi):
            raise _exceptional(f"unit {u.id}: NaN in the analysis")
        if math.isinf(e):
            raise InconclusiveError(Inconclusive.BOUND_ABOVE_EPSILON,
                                    f"unit {u.id}: the error bound diverges", u.id)
        if fmt is not None and max(abs(lo), abs(hi)) > fmt.max and \
                not (math.isinf(lo) and math.isinf(hi)):
            raise _exceptional(f"unit {u.id}: values may overflow {b.dtype} "
                               f"(range [{lo:.3g}, {hi:.3g}])")

    # ---- P: exact difference on program 1's canonical nodes ----

    def _exact_pass(self, u, m, inputs, report):
        k = m.canonical_1
        env = {}
        for b, t in zip(u.inputs, inputs):
            env[b.value_1] = AV(t.min_val, t.max_val, t.max_error, _kind(b.dtype), b.dtype)
        idiom2 = [m.canonical_2.nodes[n] for n in u.canonical_nodes_2
                  if m.canonical_2.nodes[n].op == "gelu" and
                  m.canonical_2.nodes[n].meta.get("idiom_constants")]
        mode = Mode(exact=True, sides=())
        for nid in u.canonical_nodes_1:
            n = k.nodes[nid]
            ins = [self._get(k, env, i) for i in n.inputs]
            res = self._handle(k, n, ins, mode)
            dev = res.lip
            if n.op == "gelu":
                x = ins[0]
                dev += self._idiom_gap(n, x)
                dev += max((self._idiom_gap(n2, x) for n2 in idiom2), default=0.0)
            for o in n.outputs:
                env[o] = AV(res.lo, res.hi, dev, res.kind, k.values[o].dtype)
        return [self._get(k, env, b.value_1) for b in u.outputs]

    def _get(self, k, env, vid):
        if vid is None:
            return None
        if vid in env:
            return env[vid]
        v = k.values[vid]
        if v.kind == "const":
            env[vid] = _av_const(v)
            return env[vid]
        raise _unsupported_op(f"value {vid} has no abstract value")

    @staticmethod
    def _idiom_gap(n, x: AV) -> float:
        """|GELU written with these constants - reference GELU| over x's range."""
        c = n.meta.get("idiom_constants")
        if not c:
            return 0.0
        X = x.mag
        if n.attrs.get("approximate") == "tanh":
            s, a = float(c["scale"]), float(c["cubic"])
            gap = 0.5 * X * (abs(s - SQRT_2_OVER_PI) * (X + 0.044715 * X ** 3)
                             + s * abs(a - 0.044715) * X ** 3)
        else:
            gap = X * X * abs(float(c["scale"]) - INV_SQRT2) / SQRT_PI
        return gap * fpm.SAFETY + 1e-16 * (X ** 4 + X)

    # ---- R_j: one program's own rounding, on its raw lowering ----

    def _raw_unit(self, j, u, st):
        S = set(u.source_nodes_1 if j == 1 else u.source_nodes_2)
        nodes = sorted({nid for s in S for nid in st.by_src[j].get(s, ())},
                       key=st.pos[j].get)
        g = st.raw[j]
        for nid in nodes:
            if not set(g.nodes[nid].meta.get("origins", ())) <= S:
                raise _unsupported_op(f"raw node {nid} spans several units")
        produced = {o for nid in nodes for o in g.nodes[nid].outputs}
        ins = []
        for nid in nodes:
            for i in g.nodes[nid].inputs:
                if i is not None and i not in produced and i not in ins:
                    ins.append(i)
        return g, nodes, ins, produced

    def _bind_inputs(self, j, u, st, g, raw_ins, inputs, dev_from_triple):
        canon = st.canon[j]
        by_origin = {}
        for idx, b in enumerate(u.inputs):
            cv = canon.values[b.value_1 if j == 1 else b.value_2]
            if cv.origin is not None:
                by_origin.setdefault(cv.origin, idx)
        env, index = {}, {}
        unbound = []
        for r in raw_ins:
            v = g.values[r]
            if v.kind == "const":
                env[r] = _av_const(v)
                continue
            idx = by_origin.get(v.origin)
            if idx is None:
                unbound.append(r)
                continue
            index[r] = idx
        free = [i for i in range(len(u.inputs)) if i not in index.values()
                and u.inputs[i].kind != "const"]
        if unbound:
            if len(unbound) == 1 and len(free) == 1:
                index[unbound[0]] = free[0]
            else:
                raise _unsupported_op(f"unit {u.id}: cannot bind raw inputs {unbound} of "
                                      f"program {j} to the unit boundary")
        for r, idx in index.items():
            t, b = inputs[idx], u.inputs[idx]
            env[r] = AV(t.min_val, t.max_val, t.max_error if dev_from_triple else 0.0,
                        _kind(g.values[r].dtype), g.values[r].dtype)
        return env, index

    def _bind_outputs(self, j, u, st, g, nodes, produced, env, check_env=True):
        canon = st.canon[j]
        outs = []
        inside = set(nodes)
        boundary = [o for nid in nodes for o in g.nodes[nid].outputs
                    if o in g.outputs or any(c.id not in inside for c in g.consumers(o))]
        for b in u.outputs:
            cv = canon.values[b.value_1 if j == 1 else b.value_2]
            r = st.by_origin[j].get(cv.origin) if cv.origin is not None else None
            if r is None or (check_env and r not in env):
                if len(u.outputs) == 1 and len(boundary) == 1:
                    r = boundary[0]
                else:
                    raise _unsupported_op(f"unit {u.id}: output {b.value_1} has no "
                                          f"counterpart in program {j}'s operations")
            outs.append(r)
        return outs

    def _side_pass(self, j, u, st, inputs, report):
        g, nodes, raw_ins, produced = self._raw_unit(j, u, st)
        env, _ = self._bind_inputs(j, u, st, g, raw_ins, inputs, dev_from_triple=False)
        self._run(g, nodes, env, lambda n: Mode(False, (j,), False, {j: n}), report, j)
        outs = self._bind_outputs(j, u, st, g, nodes, produced, env)
        return [env[r] for r in outs]

    # ---- joint pass over isomorphic raw graphs ----

    def _signatures(self, j, u, st, g, nodes, index):
        sig = {}
        for r, idx in index.items():
            sig[r] = ("in", idx)

        def s(v):
            if v is None:
                return "none"
            if v in sig:
                return sig[v]
            val = g.values[v]
            if val.kind == "const":
                d = np.ascontiguousarray(val.data)
                sig[v] = ("c", val.dtype, val.shape, hashlib.sha1(d.tobytes()).hexdigest())
                return sig[v]
            raise KeyError(v)
        per_node = {}
        for nid in nodes:
            n = g.nodes[nid]
            ins = [s(i) for i in n.inputs]
            if n.op in COMMUTATIVE or n.op in NARY:
                ins = sorted(ins, key=repr)
            key = hashlib.sha1(repr((n.op, _attr_key(n.attrs),
                                     tuple(g.values[o].shape for o in n.outputs),
                                     tuple(g.values[o].dtype for o in n.outputs),
                                     tuple(ins))).encode()).hexdigest()
            per_node[nid] = key
            for slot, o in enumerate(n.outputs):
                sig[o] = (key, slot)
        return per_node, sig

    def _joint_pass(self, u, st, inputs, report):
        g1, n1, i1, p1 = self._raw_unit(1, u, st)
        g2, n2, i2, p2 = self._raw_unit(2, u, st)
        if len(n1) != len(n2) or not n1:
            return None
        try:
            env1, idx1 = self._bind_inputs(1, u, st, g1, i1, inputs, True)
            _, idx2 = self._bind_inputs(2, u, st, g2, i2, inputs, True)
            k1, s1 = self._signatures(1, u, st, g1, n1, idx1)
            k2, s2 = self._signatures(2, u, st, g2, n2, idx2)
        except (InconclusiveError, KeyError):
            return None
        inv2 = collections.defaultdict(list)
        for nid, key in k2.items():
            inv2[key].append(nid)
        if any(len(v) != 1 for v in inv2.values()) or sorted(k1.values()) != sorted(k2.values()):
            return None
        partner = {nid: inv2[k1[nid]][0] for nid in n1}
        self._run(g1, n1, env1, lambda n: Mode(False, (1, 2), True,
                                               {1: n, 2: g2.nodes[partner[n.id]]}),
                  report, 0, st=st)
        outs1 = self._bind_outputs(1, u, st, g1, n1, p1, env1)
        outs2 = self._bind_outputs(2, u, st, g2, n2, p2, {}, check_env=False)
        # the outputs must correspond under the isomorphism
        for a, b in zip(outs1, outs2):
            if s1.get(a) != s2.get(b):
                return None
        return [env1[r] for r in outs1]

    # ---- driver over a list of raw nodes ----

    def _run(self, g, nodes, env, mode_of, report, side, st=None):
        consumed = set()
        for nid in nodes:
            if nid in consumed:
                continue
            n = g.nodes[nid]
            mode = mode_of(n)
            ins = [self._get(g, env, i) for i in n.inputs]
            group = self._bias_group(g, n, set(nodes)) if n.op in ("matmul", "conv") else None
            if group is not None:
                bias_node, bias_vid = group
                if mode.joint and self._bias_group(self._cur.raw[2], mode.nodes[2], None) is None:
                    group = None
                if group is not None:
                    consumed.add(bias_node.id)
                    res = self._accumulate(g, n, ins, mode, report, bias=(g.values[bias_vid]))
                    self._store(env, g, bias_node, res, mode, ins)
                    continue
            res = self._handle(g, n, ins, mode, report)
            self._store(env, g, n, res, mode, ins)

    def _store(self, env, g, n, res, mode, ins):
        if mode.exact:
            dev = res.lip
            widen = 0.0
        else:
            rho = [res.rho.get(s, 0.0) for s in mode.sides]
            widen = max(rho) if rho else 0.0
            dev = res.lip + sum(rho)
            if mode.joint and all(a is None or a.dev == 0 for a in ins):
                if res.rel is not None:
                    dev = res.rel
                elif n.op in CORRECTLY_ROUNDED and self._separate_both(mode):
                    dev = 0.0
        out_dtype = g.values[n.outputs[0]].dtype
        lo, hi = res.lo - widen, res.hi + widen
        if res.kind == "b":
            lo, hi, dev = 0.0, 1.0, (0.0 if dev == 0 else 1.0)
        for o in n.outputs:
            env[o] = AV(lo, hi, dev, res.kind, out_dtype)

    def _separate_both(self, mode) -> bool:
        """Both programs run this correctly rounded operation as its own IEEE operation (no
        contraction with a neighbour), so equal inputs give bitwise-equal outputs."""
        st = self._cur
        for j in (1, 2):
            n = mode.nodes[j]
            for o in n.meta.get("origins", ()):
                src = st.src[j].nodes.get(o)
                impl = src.meta.get("impl", st.src[j].meta.get("impl", {})) if src else {}
                if st.compute_count[j][o] == 1:
                    continue
                if impl.get("backend") in DSL_BACKENDS and _base_op(src.op) in SWAPPED_SOURCE and \
                        n.op == "add":
                    continue                     # bias added by a separate torch op
                return False
        return True

    def _bias_group(self, g, n, inside):
        """matmul/conv followed by a bias add from the same source operation."""
        out = n.outputs[0]
        cons = g.consumers(out)
        if len(cons) != 1 or out in g.outputs:
            return None
        c = cons[0]
        if c.op != "add" or (inside is not None and c.id not in inside):
            return None
        if set(c.meta.get("origins", ())) != set(n.meta.get("origins", ())):
            return None
        other = c.inputs[1] if c.inputs[0] == out else c.inputs[0]
        if other is None or not g.is_const(other):
            return None
        return c, other

    # ---- per-side metadata ----

    def _impl(self, j, raw_node):
        st = self._cur
        o = sorted(raw_node.meta.get("origins", ()))
        src = st.src[j].nodes.get(o[0]) if o else None
        impl = (src.meta.get("impl") if src is not None else None) or \
            st.src[j].meta.get("impl") or {"backend": st.src[j].frontend}
        return impl, (src.op if src is not None else "")

    # ---- op handlers ----

    def _handle(self, g, n, ins, mode, report=None) -> Res:
        op = n.op
        out = g.values[n.outputs[0]]
        k = _kind(out.dtype)
        if k == "?":
            raise _unsupported_path(f"{op}: unsupported dtype {out.dtype}")
        if k == "f" and out.dtype not in fpm.FORMATS:
            raise _unsupported_path(f"{op}: no rounding model for {out.dtype}")
        fmt = fpm.FORMATS.get(out.dtype, fpm.FORMATS["float32"])
        sides = mode.sides

        def R(x):
            return {s: x for s in sides}

        if op == "opaque":
            raise _unsupported_op(f"opaque operation {n.attrs.get('source')}: "
                                  f"{n.meta.get('reason', '')}")
        if op in ("reshape", "permute", "broadcast_to", "slice"):
            a = ins[0]
            return Res(a.lo, a.hi, a.dev, kind=a.kind)
        if op == "concat":
            lo, hi = _hull(*ins)
            return Res(lo, hi, max(a.dev for a in ins), kind=ins[0].kind)
        if op == "pad":
            lo, hi = _hull(*[a for a in ins if a is not None])
            return Res(lo, hi, max(a.dev for a in ins if a is not None), kind=ins[0].kind)
        if op == "gather":
            t, idx = ins
            if idx.dev != 0:
                raise _unsupported_path("gather with indices that may differ")
            tv = g.values[n.inputs[0]]
            if tv.kind == "const" and g.is_const(n.inputs[1]):
                sel = np.take(tv.data, g.values[n.inputs[1]].data, axis=n.attrs["axis"])
                return Res(float(sel.min()), float(sel.max()), 0.0, kind=t.kind)
            return Res(t.lo, t.hi, t.dev, kind=t.kind)
        if op == "cast":
            return self._cast(g, n, ins[0], out, mode)
        if op in ("eq", "ne", "lt", "le", "gt", "ge", "and", "or", "not"):
            dev = 0.0 if all(a.dev == 0 for a in ins if a is not None) else 1.0
            return Res(0.0, 1.0, dev, kind="b")
        if op == "where":
            c, a, b = ins
            lo, hi = _hull(a, b)
            dev = max(a.dev, b.dev)
            if c.dev != 0:
                dev = max(dev, a.hi - b.lo, b.hi - a.lo)
            return Res(lo, hi, dev, kind=k)
        if k != "f":
            return self._int_op(n, ins, k)
        if op in ("add", "sub") and len(ins) == 2:
            a, b = ins
            lo, hi = (a.lo + b.lo, a.hi + b.hi) if op == "add" else (a.lo - b.hi, a.hi - b.lo)
            return Res(lo, hi, a.dev + b.dev, R(fmt.rho(max(abs(lo), abs(hi)), True)))
        if op == "add":                                      # n-ary (canonical, exact pass)
            return Res(sum(a.lo for a in ins), sum(a.hi for a in ins), sum(a.dev for a in ins))
        if op == "mul":
            acc = ins[0]
            lo, hi, dev = acc.lo, acc.hi, acc.dev
            for b in ins[1:]:
                A = max(abs(lo), abs(hi))
                dev = A * b.dev + b.mag * dev
                lo, hi = _mul_iv(lo, hi, b.lo, b.hi)
            return Res(lo, hi, dev, R(fmt.rho(max(abs(lo), abs(hi)))) if len(ins) == 2 else {})
        if op == "div":
            a, b = ins
            if b.lo <= 0 <= b.hi:
                raise _exceptional("division by a range that contains 0")
            m = min(abs(b.lo), abs(b.hi))
            lo, hi = _mul_iv(a.lo, a.hi, 1 / b.hi, 1 / b.lo)
            dev = a.dev / m + a.mag * b.dev / (m * m)
            return Res(lo, hi, dev, R(fmt.rho(max(abs(lo), abs(hi)))))
        if op in ("max", "min"):
            a, b = ins
            f = max if op == "max" else min
            return Res(f(a.lo, b.lo), f(a.hi, b.hi), max(a.dev, b.dev))
        if op == "relu":
            a = ins[0]
            return Res(max(a.lo, 0.0), max(a.hi, 0.0), a.dev)
        if op == "neg":
            a = ins[0]
            return Res(-a.hi, -a.lo, a.dev)
        if op == "abs":
            a = ins[0]
            lo = 0.0 if a.lo <= 0 <= a.hi else min(abs(a.lo), abs(a.hi))
            return Res(lo, a.mag, a.dev)
        if op == "pow":
            return self._pow(g, n, ins, fmt, R)
        if op in ("exp", "log", "sqrt", "rsqrt", "reciprocal", "tanh", "sigmoid", "erf", "sin",
                  "cos", "silu", "gelu"):
            return self._unary(op, n, ins[0], fmt, R)
        if op in ("matmul", "conv", "reduce_sum", "reduce_mean", "avg_pool"):
            return self._accumulate(g, n, ins, mode, report)
        if op in ("reduce_max", "reduce_min", "max_pool"):
            a = ins[0]
            return Res(a.lo, a.hi, a.dev)
        if op in ("softmax", "log_softmax"):
            return self._softmax(g, n, ins[0], fmt, R)
        if op in ("layer_norm", "rms_norm"):
            return self._norm(g, n, ins, fmt, R)
        if op == "batch_norm":
            return self._batch_norm(g, n, ins, fmt, R)
        raise _unsupported_op(f"no transfer function for {op}")

    def _int_op(self, n, ins, k):
        if any(a.dev for a in ins if a is not None):
            raise _unsupported_path(f"integer {n.op} on values that may differ")
        a = ins[0]
        if n.op in ("add", "sub", "mul") and len(ins) == 2:
            b = ins[1]
            if n.op == "add":
                lo, hi = a.lo + b.lo, a.hi + b.hi
            elif n.op == "sub":
                lo, hi = a.lo - b.hi, a.hi - b.lo
            else:
                lo, hi = _mul_iv(a.lo, a.hi, b.lo, b.hi)
            if max(abs(lo), abs(hi)) >= 2.0 ** 62:
                raise _exceptional("integer overflow possible")
            return Res(lo, hi, 0.0, kind=k)
        if n.op in ("max", "min", "relu", "neg", "abs", "reduce_max", "reduce_min"):
            lo, hi = _hull(*[x for x in ins if x is not None])
            m = max(abs(lo), abs(hi))
            return Res(-m, m, 0.0, kind=k)
        raise _unsupported_op(f"integer {n.op}")

    def _cast(self, g, n, a, out, mode):
        src_k, dst_k = a.kind, _kind(out.dtype)
        if dst_k == "f":
            fmt = fpm.fmt_of(out.dtype)
            if src_k == "f":
                sf = fpm.fmt_of(a.dtype)
                rho = 0.0 if fmt.p >= sf.p else fmt.rho(a.mag)
            else:
                rho = 0.0 if a.mag <= 2.0 ** fmt.p else fmt.rho(a.mag)
            return Res(a.lo, a.hi, a.dev, {s: rho for s in mode.sides})
        if a.kind == "f" and a.dev != 0:
            raise _unsupported_path(f"cast of a float that may differ to {out.dtype}")
        if dst_k == "b":
            return Res(0.0, 1.0, 0.0, kind="b")
        return Res(math.floor(a.lo), math.ceil(a.hi), 0.0, kind=dst_k)

    def _pow(self, g, n, ins, fmt, R):
        x, e = ins
        ev = g.values[n.inputs[1]]
        if ev.kind != "const" or ev.numel != 1:
            raise _unsupported_op("pow with a non-scalar or variable exponent")
        p = float(np.asarray(ev.data).reshape(()))
        integer = float(p).is_integer()
        if not integer and x.lo < 0:
            raise _exceptional("pow with a non-integer exponent of a possibly negative base")
        if p < 0 and x.lo <= 0 <= x.hi:
            raise _exceptional("negative power of a range that contains 0")
        if p < 1 and not integer and x.lo <= 0:
            raise _unsupported_op("fractional power at 0 (unbounded derivative)")
        pts = [x.lo, x.hi] + ([0.0] if x.lo < 0 < x.hi else [])
        vals = [math.copysign(abs(t) ** p, t) if integer and int(p) % 2 else abs(t) ** p
                for t in pts]
        lo, hi = min(vals), max(vals)
        if p >= 1:
            lip = abs(p) * x.mag ** (p - 1) * x.dev
        else:
            m = min(abs(x.lo), abs(x.hi))
            lip = abs(p) * m ** (p - 1) * x.dev
        q = self._ulp(self._cfg, "pow")
        return Res(lo, hi, lip, R(q * fmt.ulp_bound(max(abs(lo), abs(hi)))))

    def _unary(self, op, n, x, fmt, R):
        u = fmt.u
        if op == "exp":
            if x.hi > math.log(fmt.max):
                raise _exceptional(f"exp may overflow (input up to {x.hi:.3g})")
            lo, hi = math.exp(x.lo), math.exp(x.hi)
            q = self._ulp(self._cfg, "exp")
            return Res(lo, hi, hi * x.dev, R(q * fmt.ulp_bound(hi)))
        if op == "log":
            if x.lo <= 0:
                raise _exceptional("log of a range that reaches 0")
            lo, hi = math.log(x.lo), math.log(x.hi)
            q = self._ulp(self._cfg, "log")
            return Res(lo, hi, x.dev / x.lo, R(q * fmt.ulp_bound(max(abs(lo), abs(hi)))))
        if op == "sqrt":
            if x.lo < 0:
                raise _exceptional("sqrt of a range that contains negative values")
            lo, hi = math.sqrt(x.lo), math.sqrt(x.hi)
            dev = math.sqrt(x.dev)
            if x.lo > 0:
                dev = min(dev, x.dev / (2 * math.sqrt(x.lo)))
            q = self._ulp(self._cfg, "sqrt")
            return Res(lo, hi, dev, R(q * fmt.ulp_bound(hi)))
        if op == "rsqrt":
            if x.lo <= 0:
                raise _exceptional("rsqrt of a range that reaches 0")
            lo, hi = x.hi ** -0.5, x.lo ** -0.5
            q = self._ulp(self._cfg, "rsqrt")
            return Res(lo, hi, 0.5 * x.lo ** -1.5 * x.dev, R(q * fmt.ulp_bound(hi)))
        if op == "reciprocal":
            if x.lo <= 0 <= x.hi:
                raise _exceptional("reciprocal of a range that contains 0")
            m = min(abs(x.lo), abs(x.hi))
            lo, hi = sorted((1 / x.lo, 1 / x.hi))
            q = self._ulp(self._cfg, "reciprocal")
            return Res(lo, hi, x.dev / (m * m), R(q * fmt.ulp_bound(1 / m)))
        m = 0.0 if x.lo <= 0 <= x.hi else min(abs(x.lo), abs(x.hi))
        if op == "tanh":
            lo, hi = math.tanh(x.lo), math.tanh(x.hi)
            q = self._ulp(self._cfg, "tanh")
            return Res(lo, hi, (1 - math.tanh(m) ** 2) * x.dev,
                       R(q * fmt.ulp_bound(max(abs(lo), abs(hi)))))
        if op == "erf":
            lo, hi = math.erf(x.lo), math.erf(x.hi)
            q = self._ulp(self._cfg, "erf")
            return Res(lo, hi, 2 / SQRT_PI * math.exp(-m * m) * x.dev,
                       R(q * fmt.ulp_bound(max(abs(lo), abs(hi)))))
        if op in ("sin", "cos"):
            q = self._ulp(self._cfg, op)
            return Res(-1.0, 1.0, x.dev, R(q * fmt.ulp_bound(1.0)))
        if op == "sigmoid":
            sg = lambda t: 1 / (1 + math.exp(-t)) if t > -700 else 0.0  # noqa: E731
            lo, hi = sg(x.lo), sg(x.hi)
            s = sg(m)
            q = self._ulp(self._cfg, "exp")
            self.assumptions.add("sigmoid evaluated as 1/(1+exp(-x))")
            return Res(lo, hi, s * (1 - s) * x.dev, R((2 * q + 3.02) * u * hi + fmt.tiny))
        if op == "silu":
            sl = lambda t: t / (1 + math.exp(-t)) if t > -700 else 0.0  # noqa: E731
            vals = [sl(x.lo), sl(x.hi)]
            lo = min(vals + ([SILU_MIN] if x.lo <= -1.278 <= x.hi + 0.001 else []))
            hi = max(vals)
            q = self._ulp(self._cfg, "exp")
            self.assumptions.add("silu evaluated as x/(1+exp(-x)) or x*sigmoid(x)")
            Y = max(abs(lo), abs(hi))
            return Res(lo, hi, SILU_LIP * x.dev, R((2 * q + 4.02) * u * Y + fmt.tiny))
        if op == "gelu":
            return self._gelu(n, x, fmt, R)
        raise _unsupported_op(op)

    def _gelu(self, n, x, fmt, R):
        u, X = fmt.u, x.mag
        tanh_form = n.attrs.get("approximate") == "tanh"
        if tanh_form:
            G = lambda t: 0.5 * t * (1 + math.tanh(SQRT_2_OVER_PI * (t + 0.044715 * t ** 3)))  # noqa
        else:
            G = lambda t: 0.5 * t * (1 + math.erf(t * INV_SQRT2))  # noqa: E731
        vals = [G(x.lo), G(x.hi)]
        lo = min(vals + ([GELU_MIN] if x.lo <= -0.7517 and x.hi >= -0.7526 else []))
        hi = max(vals)
        lo, hi = lo - 1e-12 * (1 + abs(lo)), hi + 1e-12 * (1 + abs(hi))
        Y = max(abs(lo), abs(hi))
        tail = X * u + 2.02 * u * Y + 2 * fmt.eta * 3
        if tanh_form:
            q = self._ulp(self._cfg, "tanh")
            c, a = SQRT_2_OVER_PI, 0.044715
            inc = lambda t: 0.5 * t * 5.05 * u * c * (t + a * t ** 3)  # noqa: E731
            dec = lambda t: np.minimum(1.0, 4 * np.exp(-2 * c * (t + a * t ** 3) * 0.999))  # noqa
            arg = fpm.sup_increasing_times_decreasing(inc, dec, X)
            rho = arg + 0.5 * X * q * (2 * u + 2 * fmt.eta) + tail
            self.assumptions.add("tanh-GELU evaluated as 0.5x(1+tanh(c(x+a x^3)))")
        else:
            q = self._ulp(self._cfg, "erf")
            arg = (1 / SQRT_PI) * INV_SQRT2 * u * (2 / math.e) * 1.01
            rho = arg + 0.5 * X * q * (2 * u + 2 * fmt.eta) + tail
            self.assumptions.add("GELU evaluated as 0.5x(1+erf(x/sqrt 2))")
        return Res(lo, hi, GELU_LIP * x.dev, R(rho * fpm.SAFETY))

    # ---- accumulations (matmul, conv, sums) ----

    def _accumulate(self, g, n, ins, mode, report, bias=None) -> Res:
        op = n.op
        out = g.values[n.outputs[0]]
        fmt = fpm.fmt_of(out.dtype)
        x = ins[0]
        products = op in ("matmul", "conv")
        if op == "matmul":
            X, cols_lo, cols_hi, lip, nterm = self._matmul_leaves(g, n, ins)
        elif op == "conv":
            X, cols_lo, cols_hi, lip, nterm = self._conv_leaves(g, n, ins)
        else:
            shp = g.values[n.inputs[0]].shape
            if op == "avg_pool":
                nterm = int(np.prod(n.attrs["kernel"]))
                padded = any(n.attrs.get("pads", ())) and n.attrs.get("count_include_pad", True)
                lo_, hi_ = (min(x.lo, 0.0), max(x.hi, 0.0)) if padded else (x.lo, x.hi)
            else:
                axes = n.attrs.get("axes") or tuple(range(len(shp)))
                nterm = int(np.prod([shp[a] for a in axes])) if shp else 1
                lo_, hi_ = x.lo, x.hi
            X = np.full((max(nterm, 1), 1), max(abs(lo_), abs(hi_)))
            if op == "reduce_sum":
                cols_lo, cols_hi, lip = np.array([nterm * lo_]), np.array([nterm * hi_]), \
                    nterm * x.dev
            else:
                cols_lo, cols_hi, lip = np.array([lo_]), np.array([hi_]), x.dev
        bias_cols = None
        if bias is not None:
            bias_cols = self._bias_columns(bias, X.shape[1])
            cols_lo = cols_lo + bias_cols[0]
            cols_hi = cols_hi + bias_cols[1]
        lo, hi = float(np.min(cols_lo)), float(np.max(cols_hi))
        res = Res(lo, hi, lip)
        if mode.exact:
            return res
        acc_report = {"op": op, "terms": int(X.shape[0]), "columns": int(X.shape[1])}
        orders = {}
        for s in mode.sides:
            impl, sop = self._impl(s, mode.nodes[s])
            kind = op if op in ("matmul", "conv") else "reduce_sum"
            order = self._oracle.order(impl, sop, kind, X.shape[0])
            orders[s] = order
            res.rho[s] = self._order_bound(order, X, fmt, products, bias_cols, op, nterm, hi, lo)
            acc_report[f"order_{s}"] = fpm.describe(order)
            acc_report[f"R{s}"] = res.rho[s]
        if mode.joint and fpm.same_known(orders.get(1), orders.get(2)) and \
                all(a is None or a.dev == 0 for a in ins):
            if orders[1].tree.n == X.shape[0]:
                d, note = fpm.relational(orders[1], orders[2], X, fmt, products)
                rel = float(np.max(d))
                if bias_cols is not None and rel > 0:
                    rel += 2 * fmt.rho(max(abs(lo), abs(hi)), True)
                if op in ("reduce_mean", "avg_pool") and rel > 0:
                    rel = rel / nterm + 2 * (2.01 * fmt.u * max(abs(lo), abs(hi)) + fmt.eta)
                res.rel = rel
                acc_report["relational"] = note
                acc_report["D"] = rel
        if report is not None:
            report.accumulations.append(acc_report)
        return res

    def _order_bound(self, order, X, fmt, products, bias_cols, op, nterm, hi, lo):
        Y = max(abs(lo), abs(hi))
        if bias_cols is not None:
            if isinstance(order, (fpm.Known, fpm.Chunks)):
                b = float(np.max(fpm.bound(order, X, fmt, products)))
                b += fmt.rho(Y, True)                 # bias added by a separate rounding
            else:
                Xb = np.vstack([X, bias_cols[2][None, :]])   # bias is one more leaf
                b = float(np.max(fpm.bound(order, Xb, fmt, products)))
        else:
            b = float(np.max(fpm.bound(order, X, fmt, products)))
        if op in ("reduce_mean", "avg_pool"):
            b = b / nterm * (1 + fmt.u) + 2.01 * fmt.u * Y + fmt.eta
        return b

    def _bias_columns(self, bias, ncols):
        d = np.asarray(bias.data, dtype=np.float64)
        flat = d.reshape(-1)
        if flat.size == ncols and ncols > 1:
            return flat, flat, np.abs(flat)
        if flat.size == 1 or ncols == 1:
            return np.full(ncols, flat.min()), np.full(ncols, flat.max()), \
                np.full(ncols, np.abs(flat).max())
        return np.full(ncols, flat.min()), np.full(ncols, flat.max()), \
            np.full(ncols, np.abs(flat).max())

    def _matmul_leaves(self, g, n, ins):
        a, b = ins
        va, vb = g.values[n.inputs[0]], g.values[n.inputs[1]]
        K = va.shape[-1]
        if vb.kind == "const":
            w = np.asarray(vb.data, dtype=np.float64)
            w = w.reshape(K, -1) if w.ndim <= 2 else np.moveaxis(w.reshape(-1, K, w.shape[-1]),
                                                                  1, 0).reshape(K, -1)
            return self._weighted(w, a, b.dev)
        if va.kind == "const":
            w = np.asarray(va.data, dtype=np.float64)
            w = w.reshape(-1, K).T
            return self._weighted(w, b, a.dev)
        A, B = a.mag, b.mag
        lo, hi = _mul_iv(a.lo, a.hi, b.lo, b.hi)
        X = np.full((K, 1), A * B)
        lip = K * (A * b.dev + B * a.dev)
        return X, np.array([K * lo]), np.array([K * hi]), lip, K

    @staticmethod
    def _weighted(w, x: AV, wdev):
        """Leaves |x_i| |w_ik| for a constant weight matrix w [K, columns]."""
        pos, neg = np.clip(w, 0, None), np.clip(w, None, 0)
        sp, sn = pos.sum(0), neg.sum(0)
        cols_lo = x.lo * sp + x.hi * sn
        cols_hi = x.hi * sp + x.lo * sn
        absw = np.abs(w)
        lip = float(np.max(absw.sum(0))) * x.dev + x.mag * w.shape[0] * wdev
        return x.mag * absw, cols_lo, cols_hi, lip, w.shape[0]

    def _conv_leaves(self, g, n, ins):
        x, w = ins
        vw = g.values[n.inputs[1]]
        padded = any(n.attrs.get("pads", ()))
        xv = AV(min(x.lo, 0.0) if padded else x.lo, max(x.hi, 0.0) if padded else x.hi, x.dev)
        if vw.kind != "const":
            Kc = int(np.prod(vw.shape[1:]))
            lo, hi = _mul_iv(xv.lo, xv.hi, w.lo, w.hi)
            return np.full((Kc, 1), xv.mag * w.mag), np.array([Kc * lo]), \
                np.array([Kc * hi]), Kc * (xv.mag * w.dev + w.mag * x.dev), Kc
        d = np.asarray(vw.data, dtype=np.float64)
        wm = d.reshape(d.shape[0], -1).T                    # [Cg*kh*kw, Cout]
        return self._weighted(wm, xv, 0.0)

    # ---- composite ops ----

    def _softmax(self, g, n, x, fmt, R):
        u = fmt.u
        nt = g.values[n.inputs[0]].shape[n.attrs["axis"]]
        Rw = x.hi - x.lo
        if not math.isfinite(Rw):
            raise _exceptional("softmax over an unbounded range")
        q = self._ulp(self._cfg, "exp")
        e_e = (1 + 2 * q * u) * math.exp(u * Rw) - 1
        g_n = fmt.gamma(nt)
        e_S = (1 + e_e) * (1 + (nt - 1) * u * (1 + g_n)) - 1 + nt * q * 2 * fmt.eta / (1 - e_e)
        if e_S >= 1:
            return Res(0.0, 1.0, math.inf, R(math.inf))
        self.assumptions.add("softmax evaluated as exp(x - max) / sum(exp(x - max)), any "
                             "summation order")
        if n.op == "softmax":
            ymin = 1 / (1 + (nt - 1) * math.exp(Rw)) if Rw < 700 else 0.0
            ymax = 1 / (1 + (nt - 1) * math.exp(-Rw)) if Rw < 700 else 1.0
            e_y = (1 + e_e) * (1 + u) ** 2 / (1 - e_S) - 1
            return Res(ymin, ymax, 0.5 * x.dev, R(e_y * ymax + 2 * fmt.eta))
        ln = math.log(nt)
        lo = -Rw - ln
        hi = -math.log1p((nt - 1) * math.exp(-Rw)) if Rw < 700 else 0.0
        ql = self._ulp(self._cfg, "log")
        rho = u * Rw - math.log1p(-e_S) + ql * fmt.ulp_bound(ln + 1) + u * (Rw + ln) * 1.01 \
            + fmt.eta
        return Res(lo, hi, 2 * x.dev, R(rho))

    def _norm(self, g, n, ins, fmt, R):
        u = fmt.u
        x = ins[0]
        w = ins[1] if len(ins) > 1 else None
        b = ins[2] if len(ins) > 2 else None
        shp = g.values[n.inputs[0]].shape
        axes = n.attrs.get("axes") or (len(shp) - 1,)
        nt = int(np.prod([shp[a] for a in axes]))
        eps = float(n.attrs["eps"])
        if eps <= 0:
            raise _exceptional(f"{n.op} with eps = {eps}")
        se = math.sqrt(eps)
        X = x.mag
        Wm = w.mag if w is not None else 1.0
        Bl, Bh = (b.lo, b.hi) if b is not None else (0.0, 0.0)
        Bm = max(abs(Bl), abs(Bh))
        q = self._ulp(self._cfg, "rsqrt")
        g_n = fmt.gamma(nt)
        tree = lambda leaf: u * (1 + g_n) * leaf * (nt * (nt + 1) / 2 - 1)  # noqa: E731
        if n.op == "layer_norm":
            width = x.hi - x.lo
            z = min(math.sqrt(max(nt - 1, 0)), width / se)
            L = (2 + z) / se
            d_mu = tree(X) / nt + 2 * u * X + fmt.eta
            D = width + d_mu
            V = (width / 2) ** 2
            S_tree = tree(D * D * (1 + 3 * u))
            e_var = d_mu ** 2 + 3.01 * u * D * D + S_tree / nt
            e_var += 2 * u * (V + e_var) + fmt.eta
            rho_a = (e_var + u * (V + e_var + eps)) / eps
            self.assumptions.add("layer_norm: mean and variance by two-pass or Welford "
                                 "(bounded as two-pass), rsqrt(var + eps), any summation order")
        else:
            z = min(math.sqrt(nt), X / se)
            L = (1 + z) / se
            d_mu = 0.0
            rel = (1 + u) * (1 + (nt - 1) * u * (1 + g_n)) * (1 + 2 * u) - 1
            rho_a = rel + u
            self.assumptions.add("rms_norm: x * rsqrt(mean(x^2) + eps), any summation order")
        if rho_a >= 0.5:
            rho = math.inf
        else:
            e_r = (1 + 2 * q * u) / math.sqrt(1 - rho_a) - 1
            e_y = d_mu / se * (1 + e_r) + z * e_r + 2.01 * u * (z + d_mu / se) * (1 + e_r) \
                + 2 * fmt.eta
            rho = Wm * e_y + u * (2 * (z + e_y) * Wm + Bm) * 1.01 + 2 * fmt.eta
        lo, hi = -z * Wm + Bl, z * Wm + Bh
        lip = Wm * L * x.dev
        if w is not None and w.dev:
            lip += z * w.dev
        if b is not None and b.dev:
            lip += b.dev
        return Res(lo, hi, lip, R(rho * fpm.SAFETY))

    def _batch_norm(self, g, n, ins, fmt, R):
        x = ins[0]
        vals = [g.values[i] for i in n.inputs[1:5]]
        if any(v is None or v.kind != "const" for v in vals):
            raise _unsupported_op("batch_norm with non-constant statistics")
        w, b, mean, var = [np.asarray(v.data, dtype=np.float64) for v in vals]
        eps = float(n.attrs["eps"])
        s = w / np.sqrt(var + eps)
        t = b - mean * s
        lo = float(np.min(np.minimum(x.lo * s, x.hi * s) + t))
        hi = float(np.max(np.maximum(x.lo * s, x.hi * s) + t))
        q = self._ulp(self._cfg, "rsqrt")
        mag = float(np.max(x.mag * np.abs(s) + np.abs(mean * s) + np.abs(b)))
        rho = (2 * q + 6) * fmt.u * mag + 4 * fmt.eta
        self.assumptions.add("batch_norm (inference) evaluated as (x - mean) * scale + b or "
                             "x * scale + shift with fp32 scale = w * rsqrt(var + eps)")
        return Res(lo, hi, float(np.max(np.abs(s))) * x.dev, R(rho * fpm.SAFETY))
