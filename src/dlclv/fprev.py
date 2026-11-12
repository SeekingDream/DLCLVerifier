"""Reveal and validate the accumulation order of a summation kernel (FPRev, arXiv:2411.00442).

FPRev calls the kernel on masked inputs: +M at position i, -M at position j, 1 elsewhere, with
M large enough that M + (any partial sum of ones) rounds back to M. The ones summed into a
subtree containing i or j before i and j cancel are absorbed, so the kernel returns
n - l(i, j), where l(i, j) is the number of leaves under the lowest common ancestor of i and j.
The tree is rebuilt by merging leaf groups in increasing order of l. Groups that merge at the
same l and form one subtree with more than two children are a fused multi-term node.

FPRev's correctness assumes the order is a fixed function of the implementation and the
hardware: it excludes randomized and value-dependent orders. It does not check that assumption.
`collect_evidence` therefore adds the checks of docs/ABSTRACT_INTERPRETATION.md section 4.2
that can be automated:

* determinism: repeated calls on random inputs give bitwise-equal results;
* data independence (dynamic corroboration): an exact IEEE emulator of the revealed tree
  reproduces the kernel bit for bit on random and adversarial inputs (cancellation, wide
  exponent spread, subnormals).

That yields level "validated". Level "known" also needs static evidence (the kernel source or
SASS shows a data-independent reduction, and which FP operations it uses), which a person
records in `OrderEvidence.static_evidence`.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .fpmodel import FORMATS, Fmt, Known, SumTree, round_to_format


class RevealError(RuntimeError):
    pass


def mask_value(fmt: Fmt, n: int) -> float:
    """M such that fl(M + k) = M for 0 <= k <= n: any M >= 2^(p+1) * n works; FPRev uses the
    largest power of two."""
    M = 2.0 ** fmt.emax
    if M * 2.0 ** -fmt.p < n + 1:
        raise RevealError(f"n = {n} is too large for {fmt.name} masking")
    return M


def _lca_size(sum_fn, n, i, j, M, dtype):
    x = np.ones(n, dtype=dtype)
    x[i], x[j] = M, -M
    s = float(sum_fn(x))
    l = n - s
    if not float(l).is_integer() or not 2 <= l <= n:
        raise RevealError(f"kernel returned {s} for the masked pair ({i}, {j}); "
                          "the order is not a fixed summation tree over these inputs")
    return int(l)


def reveal_order(sum_fn: Callable[[np.ndarray], float], n: int, dtype: str = "float32") -> SumTree:
    """Rebuild the summation tree of `sum_fn` on n inputs (basic FPRev, n(n-1)/2 calls)."""
    fmt = FORMATS[dtype]
    if n == 1:
        return SumTree(1, [])
    M = mask_value(fmt, n)
    L = {}
    for i in range(n):
        for j in range(i + 1, n):
            L[i, j] = _lca_size(sum_fn, n, i, j, M, dtype)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    node_of = {i: i for i in range(n)}       # union-find root -> tree id of that group
    size = {i: 1 for i in range(n)}
    children: List[tuple] = []
    by_l = collections.defaultdict(list)
    for (i, j), l in L.items():
        by_l[l].append((i, j))
    for l in sorted(by_l):
        groups = collections.defaultdict(set)   # new component -> old roots it merges
        merged_parent = {}
        for i, j in by_l[l]:
            ri, rj = find(i), find(j)
            if ri == rj:
                continue
            merged_parent.setdefault(ri, ri)
            merged_parent.setdefault(rj, rj)
            # merge into a temporary union keyed by representative
            a, b = _tmp_find(merged_parent, ri), _tmp_find(merged_parent, rj)
            if a != b:
                merged_parent[b] = a
        for r in merged_parent:
            groups[_tmp_find(merged_parent, r)].add(r)
        for rep, roots in groups.items():
            roots = sorted(roots, key=lambda r: node_of[r])
            total = sum(size[r] for r in roots)
            if total != l:
                raise RevealError(f"inconsistent LCA sizes: merging {len(roots)} groups of "
                                  f"{total} leaves at l = {l}")
            ids = tuple(node_of[r] for r in roots)
            children.append(ids)
            new_root = roots[0]
            for r in roots[1:]:
                parent[r] = new_root
            node_of[new_root] = n + len(children) - 1
            size[new_root] = total
    if len(children) == 0 or size[find(0)] != n:
        raise RevealError("revealed structure is not a single tree")
    return SumTree(n, children, name=f"fprev({n})")


def _tmp_find(p, a):
    while p[a] != a:
        p[a] = p[p[a]]
        a = p[a]
    return a


# --------------------------------------------------------------------------------------------
# Exact emulation of a tree
# --------------------------------------------------------------------------------------------

def emulate(tree: SumTree, x: Sequence[float], dtype: str = "float32",
            products: Optional[Sequence[tuple]] = None, fp: str = "separate") -> float:
    """Value an IEEE implementation of `tree` returns, computed exactly and rounded once per
    node. Plain sums use `x`. Dot products pass `products=[(a_i, b_i), ...]`; with fp="fma"
    a node takes one product unrounded (the lowest-numbered real leaf child)."""
    fmt = FORMATS[dtype]
    rnd = lambda v: round_to_format(v, fmt)  # noqa: E731
    n, total = tree.n, tree.n_total
    val: Dict[int, Fraction] = {}
    exact_prod: Dict[int, Fraction] = {}
    for i in range(n):
        if products is None:
            val[i] = Fraction(float(x[i]))
        else:
            a, b = products[i]
            exact_prod[i] = Fraction(float(a)) * Fraction(float(b))
            val[i] = rnd(exact_prod[i])
    for z in range(n, total):
        val[z] = Fraction(0)
    for k, ch in enumerate(tree.children):
        terms = [val[c] for c in ch]
        if products is not None and fp == "fma":
            real = [c for c in ch if c < n]
            if real:
                terms[ch.index(real[0])] = exact_prod[real[0]]
        val[total + k] = rnd(sum(terms, Fraction(0)))
    root = total + len(tree.children) - 1 if tree.children else 0
    return float(val[root])


# --------------------------------------------------------------------------------------------
# Evidence
# --------------------------------------------------------------------------------------------

@dataclass
class OrderEvidence:
    """What justifies treating an order as fixed (docs section 4.2)."""
    tree: SumTree
    fp: str = "separate"                      # fma | separate | either
    level: str = "validated"                  # validated (dynamic only) | known
    n: int = 0
    kernel: str = ""                          # what was probed: library, kernel, shapes
    pinned: Dict[str, str] = field(default_factory=dict)   # device, versions, algorithm, ...
    deterministic_runs: int = 0
    emulator_trials: int = 0
    static_evidence: str = ""                 # required for level "known"
    notes: List[str] = field(default_factory=list)

    def order(self) -> Known:
        return Known(self.tree, self.fp, evidence=f"{self.level}: {self.kernel}")

    def digest_items(self) -> List[str]:
        return [f"order[{self.kernel}] level={self.level} fp={self.fp} n={self.n} "
                f"tree={self.tree.name} runs={self.deterministic_runs} "
                f"emulated={self.emulator_trials} pinned={sorted(self.pinned.items())} "
                f"static={self.static_evidence!r}"]


def adversarial_inputs(n: int, trials: int, rng: np.random.Generator, dtype="float32"):
    """Random inputs plus the cases where different trees disagree most."""
    fmt = FORMATS[dtype]
    outs = []
    for t in range(trials):
        kind = t % 5
        if kind == 0:
            x = rng.standard_normal(n)
        elif kind == 1:                       # wide exponent spread
            x = rng.standard_normal(n) * 2.0 ** rng.integers(-40, 40, n)
        elif kind == 2:                       # cancellation
            x = rng.standard_normal(n)
            x[: n // 2] = -x[n // 2: 2 * (n // 2)] * (1 + 2.0 ** -20)
        elif kind == 3:                       # subnormals
            x = rng.standard_normal(n) * fmt.tiny * 2.0 ** rng.integers(-20, 2, n)
        else:                                 # near overflow, but the sum stays finite
            x = rng.standard_normal(n) * fmt.max / (4 * n)
        outs.append(x.astype(dtype))
    return outs


def collect_evidence(sum_fn: Callable[[np.ndarray], float], n: int, dtype: str = "float32",
                     kernel: str = "", pinned: Optional[Dict[str, str]] = None,
                     runs: int = 5, trials: int = 50, seed: int = 0) -> OrderEvidence:
    """Reveal the order of a plain summation kernel and check it dynamically.

    Raises RevealError if the kernel is not deterministic or the emulated tree does not
    reproduce it bit for bit: then the order must be treated as unknown (Strategy 2)."""
    tree = reveal_order(sum_fn, n, dtype)
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n).astype(dtype)
    first = np.asarray(sum_fn(x)).tobytes()
    for _ in range(runs - 1):
        if np.asarray(sum_fn(x)).tobytes() != first:
            raise RevealError("kernel is not deterministic across runs")
    if tree.multiterm:
        raise RevealError("revealed multi-term fused nodes: no IEEE model (unsupported path)")
    for xi in adversarial_inputs(n, trials, rng, dtype):
        got = np.float64(np.asarray(sum_fn(xi), dtype=dtype))
        want = emulate(tree, xi, dtype)
        if not (got == want or (np.isnan(got) and np.isnan(want))):
            raise RevealError(f"emulated tree gives {want!r}, kernel gives {got!r}: the order "
                              "is value dependent or the FP model differs")
    return OrderEvidence(tree, "separate", "validated", n, kernel, dict(pinned or {}), runs,
                         trials, notes=["plain summation; products not involved"])


# --------------------------------------------------------------------------------------------
# Dot-product kernels (matmul): reveal the order along K for one output element
# --------------------------------------------------------------------------------------------

def matmul_sum_fn(mm: Callable, K: int, M: int = 1, N: int = 1, row: int = 0, col: int = 0,
                  device: str = "cuda"):
    """A plain-summation view of `mm(A[M,K], B[K,N])` at output (row, col): A[row] = x,
    B[:, col] = 1, so the kernel sums x along its K accumulation order. With B = 1 every
    product is exact, so FMA and separate multiply-add give the same tree."""
    import torch

    def fn(x):
        A = torch.zeros((M, K), dtype=torch.float32, device=device)
        A[row] = torch.as_tensor(np.asarray(x, dtype=np.float32), device=device)
        B = torch.ones((K, N), dtype=torch.float32, device=device)
        return float(mm(A, B)[row, col].item())
    return fn


def detect_fma(mm: Callable, device: str = "cuda", trials: int = 200, seed: int = 0) -> str:
    """Probe a two-term dot product: with FMA, fl(a0*b0 + fl(a1*b1)) or fl(fl(a0*b0) + a1*b1)
    differs from fl(fl(a0*b0) + fl(a1*b1)) on suitable inputs. Returns fma, separate or either
    (inconsistent observations)."""
    import torch
    rng = np.random.default_rng(seed)
    fmt = FORMATS["float32"]
    seen = set()
    for _ in range(trials):
        a = rng.standard_normal(2).astype(np.float32)
        b = rng.standard_normal(2).astype(np.float32)
        b[1] = np.float32(-a[0] * b[0] / a[1] * (1 + rng.standard_normal() * 2 ** -12))
        A = torch.as_tensor(a.reshape(1, 2), device=device)
        B = torch.as_tensor(b.reshape(2, 1), device=device)
        got = float(mm(A, B)[0, 0].item())
        p = [Fraction(float(a[i])) * Fraction(float(b[i])) for i in range(2)]
        sep = float(round_to_format(round_to_format(p[0], fmt) + round_to_format(p[1], fmt), fmt))
        f0 = float(round_to_format(p[0] + round_to_format(p[1], fmt), fmt))
        f1 = float(round_to_format(round_to_format(p[0], fmt) + p[1], fmt))
        z0 = float(round_to_format(round_to_format(p[0], fmt) + p[1], fmt))
        ok_sep = got == sep
        ok_fma = got in (f0, f1, z0)
        if ok_sep and not ok_fma:
            seen.add("separate")
        elif ok_fma and not ok_sep:
            seen.add("fma")
        elif not ok_sep and not ok_fma:
            seen.add("other")
    if seen == {"fma"}:
        return "fma"
    if seen == {"separate"}:
        return "separate"
    return "either"


# --------------------------------------------------------------------------------------------
# Fast reveal: O(n log n) masked queries for balanced trees, n - 1 for a chain
# --------------------------------------------------------------------------------------------

def reveal_order_batched(lca_sizes: Callable[[List[tuple]], List[int]], n: int) -> SumTree:
    """Rebuild a summation tree from LCA sizes, asking only for the pairs it needs.

    `lca_sizes(pairs)` returns l(i, j) for each pair: the number of leaves under the lowest
    common ancestor of leaves i and j (FPRev's masked-input measurement). The sizes l(p, j) of a
    pivot leaf p against the other leaves of its subtree give the chain of p's ancestors: the
    leaves with the same l hang off the same ancestor. Each such group is split into sibling
    subtrees and rebuilt recursively. Every consistency check that fails raises `RevealError`.
    """
    import sys
    if n == 1:
        return SumTree(1, [])
    cache: Dict[tuple, int] = {}

    def sizes(p, js):
        need = [j for j in js if (min(p, j), max(p, j)) not in cache]
        if need:
            got = lca_sizes([(p, j) for j in need])
            for j, l in zip(need, got):
                if not 2 <= l <= n:
                    raise RevealError(f"LCA size {l} for the pair ({p}, {j})")
                cache[min(p, j), max(p, j)] = int(l)
        return [cache[min(p, j), max(p, j)] for j in js]

    children: List[tuple] = []

    def split(G, l):
        """Partition G (the leaves that join p's ancestor of size l) into sibling subtrees."""
        parts, G = [], list(G)
        while G:
            q, rest = G[0], G[1:]
            ls = sizes(q, rest)
            if any(x > l for x in ls):
                raise RevealError("inconsistent LCA sizes inside a subtree")
            parts.append([q] + [j for j, x in zip(rest, ls) if x < l])
            G = [j for j, x in zip(rest, ls) if x == l]
        return parts

    def build(S):
        if len(S) == 1:
            return S[0]
        p, rest = S[0], S[1:]
        by = collections.defaultdict(list)
        for j, l in zip(rest, sizes(p, rest)):
            by[l].append(j)
        cur, size = p, 1
        for l in sorted(by):
            if size + len(by[l]) != l:
                raise RevealError(f"inconsistent LCA sizes: {size} + {len(by[l])} leaves "
                                  f"under a node of {l}")
            ids = [cur] + [build(P) for P in split(by[l], l)]
            children.append(tuple(ids))
            cur, size = n + len(children) - 1, l
        if size != len(S):
            raise RevealError("revealed structure does not cover the subtree")
        return cur

    old = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old, 4 * n + 200))
    try:
        build(list(range(n)))
    finally:
        sys.setrecursionlimit(old)
    t = SumTree(n, children, name=f"fprev({n})")
    t.queries = len(cache)
    return t


def tree_depth(t: SumTree) -> int:
    total, d = t.n_total, {}
    for k, ch in enumerate(t.children):
        d[total + k] = 1 + max(d.get(c, 0) for c in ch)
    return max(d.values()) if d else 0


def emulate_dot(tree: SumTree, a: Sequence[float], b: Sequence[float], fused: Sequence[bool],
                bias: Optional[float] = None, dtype: str = "float32") -> float:
    """Value of a dot-product kernel with summation tree `tree`: leaf i is the product
    a_i * b_i, rounded on its own unless `fused[i]` (then it enters its parent node unrounded,
    as in fma(a_i, b_i, acc)). `bias` is added to the root with one more rounding."""
    fmt = FORMATS[dtype]
    rnd = lambda v: round_to_format(v, fmt)  # noqa: E731
    n = tree.n
    val: Dict[int, Fraction] = {}
    for i in range(n):
        p = Fraction(float(a[i])) * Fraction(float(b[i]))
        val[i] = p if fused[i] else rnd(p)
    for z in range(n, tree.n_total):
        val[z] = Fraction(0)
    for k, ch in enumerate(tree.children):
        if sum(1 for c in ch if c < n and fused[c]) > 1:
            raise RevealError("two fused products under one node: not an FMA")
        val[tree.n_total + k] = rnd(sum((val[c] for c in ch), Fraction(0)))
    out = val[tree.n_total + len(tree.children) - 1] if tree.children else rnd(val[0])
    if bias is not None:
        out = rnd(out + Fraction(float(bias)))
    return float(out)


# --------------------------------------------------------------------------------------------
# PyTorch eager linear layer (cuBLAS on CUDA): y[r, c] = sum_k x[r, k] W[c, k] (+ b[c])
# --------------------------------------------------------------------------------------------

def linear_key(backend: str, a_shape, b_shape, source_op: str = "aten.linear",
               bias: bool = True) -> tuple:
    """Key of `CertifierConfig.order_evidence` for one matmul kernel call: the backend, the
    operand shapes [..., K] and [K, N] of the canonical matmul, the source operator, and whether
    its bias add is part of the same call."""
    return (backend, "matmul", tuple(int(d) for d in a_shape), tuple(int(d) for d in b_shape),
            source_op, bool(bias))


def _pinned(device):
    import torch
    pin = {"torch": torch.__version__, "device": str(device)}
    if str(device).startswith("cuda"):
        pin["gpu"] = torch.cuda.get_device_name(torch.device(device))
        pin["cuda"] = str(torch.version.cuda)
        pin["tf32"] = str(torch.backends.cuda.matmul.allow_tf32)
    return pin


def probe_eager_linear(x_shape, out_features: int, bias: bool = True, device: str = "cuda",
                       trials: int = 24, runs: int = 3, max_elements: int = 64,
                       seed: int = 0) -> OrderEvidence:
    """Reveal and validate the accumulation order of `F.linear(x, W, b)` in PyTorch eager, at
    exactly these shapes on this device.

    1. Tree: FPRev masks are placed in the weight rows (x = 1, so every product is exact), one
       pair per output column and call.
    2. Bias: its LCA with every product leaf must be the root, i.e. it is added last.
    3. Products: one probe per leaf tells whether the product is rounded on its own or fused
       into the accumulation (FMA).
    4. Validation: an exact emulator of (tree, fused flags, bias last) must reproduce the kernel
       bit for bit, on random and adversarial inputs, at every output element (or
       `max_elements` evenly spaced ones), and the kernel must be deterministic across runs.

    Any failed step raises `RevealError`: the order then stays unknown. The result has level
    "validated": it rests on these dynamic checks only, so a certificate that uses it lists it
    as an assumption (docs/ABSTRACT_INTERPRETATION.md, section 4.2).
    """
    import torch
    import torch.nn.functional as F
    x_shape = tuple(int(d) for d in x_shape)
    K, N = x_shape[-1], int(out_features)
    rows = int(np.prod(x_shape[:-1])) if len(x_shape) > 1 else 1
    fmt = FORMATS["float32"]
    Mv = mask_value(fmt, K + 1)
    f32 = torch.float32

    def run(x, W, b=None):
        with torch.no_grad():
            y = F.linear(torch.as_tensor(x, dtype=f32, device=device).reshape(x_shape),
                         torch.as_tensor(W, dtype=f32, device=device),
                         None if b is None else torch.as_tensor(b, dtype=f32, device=device))
        return y.reshape(rows, N).double().cpu().numpy()

    zero_b = np.zeros(N, np.float32) if bias else None
    ones_x = np.ones((rows, K), np.float32)

    def lca_sizes(pairs):
        out = []
        for s in range(0, len(pairs), N):
            W = np.ones((N, K), np.float32)
            for c, (i, j) in enumerate(pairs[s:s + N]):
                W[c, i], W[c, j] = Mv, -Mv
            y = run(ones_x, W, zero_b)[0]
            for c in range(len(pairs[s:s + N])):
                l = K - y[c]
                if not float(l).is_integer():
                    raise RevealError(f"kernel returned {y[c]} for a masked pair: not a fixed "
                                      "summation tree of the products")
                out.append(int(l))
        return out

    tree = reveal_order_batched(lca_sizes, K)
    if tree.multiterm:
        raise RevealError("revealed multi-term fused nodes: no IEEE model")

    notes = [f"tree depth {tree_depth(tree)}, {tree.queries} masked pairs"]
    if bias:                                     # the bias must join at the root
        for s in range(0, K, N):
            W = np.ones((N, K), np.float32)
            b = np.zeros(N, np.float32)
            cols = list(range(len(range(s, min(K, s + N)))))
            for c in cols:
                W[c, s + c], b[c] = Mv, -Mv
            y = run(ones_x, W, b)[0]
            if any(y[c] != 0.0 for c in cols):   # K + 1 leaves, all absorbed: l = K + 1
                raise RevealError("the bias is not added last (it enters inside the tree)")
        notes.append("bias added to the root with one rounding")

    # Which products are fused: product a^2 = 1 + 2^-11 + 2^-24 is not representable; with the
    # exact partner -(1 - 2^-24) the node returns 2^-11 + 2^-23 if the product enters unrounded
    # and 2^-11 + 2^-24 if it was rounded first.
    a = np.float32(1 + 2.0 ** -12)
    partner_w = np.float32(-(1 - 2.0 ** -12))
    sibling = {}
    leaves_under = {}
    for k, ch in enumerate(tree.children):
        node = tree.n_total + k
        leaves_under[node] = [x for c in ch for x in (leaves_under[c] if c >= tree.n_total
                                                      else [c])]
        for c in ch:
            if c < K:
                other = [d for d in ch if d != c][0]
                sibling[c] = other if other < K else leaves_under[other][0]
    fused = [False] * K
    xa = np.full((rows, K), a, np.float32)
    for s in range(0, K, N):
        W = np.zeros((N, K), np.float32)
        idx = list(range(s, min(K, s + N)))
        for c, i in enumerate(idx):
            W[c, i], W[c, sibling[i]] = a, partner_w
        y = run(xa, W, zero_b)[0]
        for c, i in enumerate(idx):
            if y[c] == 2.0 ** -11 + 2.0 ** -23:
                fused[i] = True
            elif y[c] != 2.0 ** -11 + 2.0 ** -24:
                raise RevealError(f"product probe of leaf {i} returned {y[c]!r}: neither a "
                                  "rounded nor a fused product")
    notes.append(f"{sum(fused)} of {K} products fused (FMA), the others rounded first")

    rng = np.random.default_rng(seed)
    elems = [(r, c) for r in range(rows) for c in range(N)]
    if len(elems) > max_elements:
        step = len(elems) / max_elements
        elems = sorted({elems[int(t * step)] for t in range(max_elements)} | {elems[-1]})
        notes.append(f"validated at {len(elems)} of {rows * N} output elements")
    xs = adversarial_inputs(K, trials, rng)
    for t in range(trials):
        x = np.stack([xs[t]] + [rng.standard_normal(K).astype(np.float32)
                                for _ in range(rows - 1)])
        W = np.stack(adversarial_inputs(K, N, rng))[rng.permutation(N)]
        if t % 5 in (1, 3, 4):                   # keep products finite and normal-ish
            W = rng.standard_normal((N, K)).astype(np.float32)
        b = rng.standard_normal(N).astype(np.float32) if bias else None
        y = run(x, W, b)
        for _ in range(runs - 1):
            if run(x, W, b).tobytes() != y.tobytes():
                raise RevealError("kernel is not deterministic across runs")
        for r, c in elems:
            want = emulate_dot(tree, x[r], W[c], fused, None if b is None else b[c])
            got = y[r, c]
            if not (got == want or (np.isnan(got) and abs(want) > fmt.max)
                    or (np.isinf(got) and abs(want) > fmt.max)):
                raise RevealError(f"emulated tree gives {want!r}, kernel gives {got!r} at "
                                  f"output ({r}, {c}): the order is value dependent or the "
                                  "floating-point model differs")
    ev = OrderEvidence(tree, "either", "validated", K,
                       kernel=f"torch eager F.linear x{list(x_shape)} W[{N}, {K}] "
                              f"bias={bias} on {device}",
                       pinned=_pinned(device), deterministic_runs=runs, emulator_trials=trials,
                       notes=notes)
    ev.fused = fused
    ev.key = linear_key("eager", x_shape, (K, N), "aten.linear", bias)
    return ev


def eager_linear_evidence(graphs, device: str = "cuda", **probe_options):
    """Probe every distinct PyTorch-eager `aten.linear` call of the given source graphs.

    Returns (evidence, failures): `evidence` maps `linear_key` to `OrderEvidence`, ready for
    `CertifierConfig.order_evidence`; `failures` maps the keys whose order could not be
    established to the reason (those calls keep the any-order bound)."""
    from .frontends import aten_name
    evidence, failures = {}, {}
    pin = (tuple(sorted(_pinned(device).items())), tuple(sorted(probe_options.items())))
    for g in graphs:
        for n in g.nodes.values():
            if aten_name(n.op) != "aten.linear":
                continue
            if (n.meta.get("impl") or g.meta.get("impl") or {}).get("backend") != "eager":
                continue
            x, w = g.values[n.inputs[0]], g.values[n.inputs[1]]
            if x.dtype != "float32" or w.dtype != "float32" or len(w.shape) != 2:
                continue
            has_bias = len(n.inputs) > 2 and n.inputs[2] is not None
            key = linear_key("eager", x.shape, (w.shape[1], w.shape[0]), "aten.linear", has_bias)
            if key in evidence or key in failures:
                continue
            if (key, pin) not in _PROBED:         # one probe per kernel call and device
                try:
                    _PROBED[key, pin] = probe_eager_linear(x.shape, w.shape[0], has_bias,
                                                           device, **probe_options)
                except RevealError as e:
                    _PROBED[key, pin] = str(e)
            got = _PROBED[key, pin]
            if isinstance(got, str):
                failures[key] = got
            else:
                evidence[key] = got
    return evidence, failures


_PROBED: Dict[tuple, object] = {}
