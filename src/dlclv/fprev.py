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
