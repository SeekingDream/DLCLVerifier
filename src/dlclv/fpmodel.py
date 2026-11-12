"""Floating-point error model and bounds on summation error for both accumulation strategies.

This module is pure numpy. It knows nothing about graphs. See docs/ABSTRACT_INTERPRETATION.md,
sections 2, 4.3 and 5.

Summation identity. For a reduction tree whose internal node k computes
fl(left + right) = left + right + d_k, summing over the tree gives exactly

    computed sum - exact sum = (leaf product roundings) + sum_k d_k.

So a bound is a sum of local bounds rho(M_k), where M_k bounds the magnitude of node k's exact
result. Two order models are bounded:

* `Known(tree)` (Strategy 1): `known_tree_bound` sums rho over the actual nodes.
  `relational_diff` bounds the difference of two known trees: subtrees that both trees compute
  identically give bitwise-equal values, so their local errors cancel.
* `AnyTree`, `Depth`, `Chunks` (Strategy 2): `admissible_bound` bounds every tree in the set.
  For `AnyTree` it uses the lemma: the j-th largest node weight of any binary tree is at most
  W - (sum of the j-1 smallest leaves). Adding the leaves sequentially, largest first, attains
  every one of these bounds, so this candidate worst path has the largest bound in the set.

Leaf magnitudes are given as an array X of shape [n, C]: n leaves for each of C independent
reductions (for example the columns of a weight matrix). Every bound is returned per column.

All bound arithmetic is in float64. A relative margin `SAFETY` covers float64 rounding in the
bound computation itself; an exact rational re-check is the job of a certificate checker.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

SAFETY = 1.0 + 2.0 ** -40


class UnsupportedFormat(ValueError):
    pass


# --------------------------------------------------------------------------------------------
# Formats
# --------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Fmt:
    name: str
    p: int            # precision in bits, including the hidden bit
    emin: int         # smallest normal exponent
    emax: int

    @property
    def u(self) -> float:
        """Unit roundoff for round to nearest: half an ulp of 1."""
        return 2.0 ** -self.p

    @property
    def eta(self) -> float:
        """Largest absolute rounding error of a correctly rounded result in the subnormal
        range: half the smallest subnormal."""
        return 2.0 ** (self.emin - self.p)

    @property
    def tiny(self) -> float:
        return 2.0 ** self.emin

    @property
    def max(self) -> float:
        return (2.0 - 2.0 ** (1 - self.p)) * 2.0 ** self.emax

    def ulp_bound(self, y: float) -> float:
        """Upper bound on ulp(v) for every |v| <= y."""
        return 2 * self.u * y + 2 * self.eta

    def rho(self, z: float, exact_sub: bool = False) -> float:
        """Rounding error bound of a correctly rounded op whose exact result has |z| <= Z.
        Additions and subtractions are exact when the result is subnormal (exact_sub)."""
        return self.u * z + (0.0 if exact_sub else self.eta)

    def gamma(self, n: int) -> float:
        nu = n * self.u
        return math.inf if nu >= 1 else nu / (1 - nu)


FORMATS = {
    "float32": Fmt("float32", 24, -126, 127),
    "float64": Fmt("float64", 53, -1022, 1023),
}


def fmt_of(dtype: str) -> Fmt:
    if dtype not in FORMATS:
        raise UnsupportedFormat(f"no rounding model for {dtype} (only float32 and float64)")
    return FORMATS[dtype]


def round_to_format(x: Fraction, f: Fmt) -> Fraction:
    """Correct rounding (to nearest, ties to even) of an exact rational. Used by emulators."""
    if x == 0:
        return Fraction(0)
    sign = -1 if x < 0 else 1
    a = abs(x)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    e = max(e, f.emin)
    scale = Fraction(2) ** (f.p - 1 - e)
    m = a * scale
    q, r = divmod(m.numerator, m.denominator)
    rem = Fraction(r, m.denominator)
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and q % 2 == 1):
        q += 1
    y = Fraction(q) / scale
    if y > Fraction(f.max):
        return Fraction(sign) * Fraction(10) ** 400   # overflow marker: larger than any float
    return sign * y


# --------------------------------------------------------------------------------------------
# Elementary functions
# --------------------------------------------------------------------------------------------

# Maximum error in ulps of single-precision CUDA math functions without fast math (CUDA C++
# Programming Guide, "Mathematical Functions", single precision). The certificate records the
# table it used; `CertifierConfig.elementary_ulp` overrides entries.
CUDA_FP32_ULP: Dict[str, float] = {
    "exp": 2, "log": 1, "tanh": 2, "erf": 2, "sqrt": 0.5, "rsqrt": 2, "sin": 2, "cos": 2,
    "pow": 4, "div": 0.5, "reciprocal": 0.5,
}


# --------------------------------------------------------------------------------------------
# Summation trees
# --------------------------------------------------------------------------------------------

class SumTree:
    """A summation tree over leaves 0..n-1, plus optional zero leaves (an accumulator that
    starts at 0, as in `acc = 0; acc = fma(a, b, acc)`).

    `children[k]` lists the child ids of internal node `n_total + k`, where `n_total` counts
    real and zero leaves; a child id below `n` is a real leaf, an id in [n, n_total) is a zero
    leaf. Internal nodes are listed children first; the last one is the root. A node with
    more than two children is a fused multi-term node (Tensor Core style).
    """

    def __init__(self, n: int, children: Sequence[Tuple[int, ...]], n_zero: int = 0,
                 name: str = ""):
        self.n = int(n)
        self.n_zero = int(n_zero)
        self.children = [tuple(int(c) for c in ch) for ch in children]
        self.name = name
        self.seq_order: Optional[List[int]] = None     # set by `sequential`
        self._check()

    @property
    def n_total(self) -> int:
        return self.n + self.n_zero

    def is_leaf(self, c: int) -> bool:
        return c < self.n_total

    def is_zero(self, c: int) -> bool:
        return self.n <= c < self.n_total

    @property
    def multiterm(self) -> bool:
        return any(len(ch) > 2 for ch in self.children)

    def _check(self):
        seen = set()
        total = self.n_total
        for k, ch in enumerate(self.children):
            if len(ch) < 2:
                raise ValueError("internal node with fewer than two children")
            for c in ch:
                if c >= total + k or c in seen:
                    raise ValueError(f"bad child {c} of node {total + k}")
                seen.add(c)
        if self.n + self.n_zero > 1 and len(seen) != total + len(self.children) - 1:
            raise ValueError("tree does not cover every leaf exactly once")

    # ---- constructors ----

    @classmethod
    def sequential(cls, n: int, order: Optional[Sequence[int]] = None, zero_init: bool = True):
        """acc = 0; for i in order: acc = acc + x_i."""
        order = list(range(n)) if order is None else list(order)
        ch, total = [], n + (1 if zero_init else 0)
        if zero_init:
            prev = n
            rest = order
        else:
            prev, rest = order[0], order[1:]
        for i in rest:
            ch.append((prev, i))
            prev = total + len(ch) - 1
        t = cls(n, ch, 1 if zero_init else 0, name=f"sequential({n})")
        t.seq_order = order
        return t

    @classmethod
    def pairwise(cls, n: int):
        ch = []
        level = list(range(n))
        while len(level) > 1:
            nxt = []
            for i in range(0, len(level) - 1, 2):
                ch.append((level[i], level[i + 1]))
                nxt.append(n + len(ch) - 1)
            if len(level) % 2:
                nxt.append(level[-1])
            level = nxt
        return cls(n, ch, name=f"pairwise({n})")

    @classmethod
    def from_nested(cls, nested, n: Optional[int] = None):
        """From nested tuples of leaf indices, e.g. ((0, 1), (2, 3))."""
        leaves = []

        def collect(t):
            if isinstance(t, int):
                leaves.append(t)
            else:
                for c in t:
                    collect(c)
        collect(nested)
        n = n if n is not None else max(leaves) + 1
        ch = []

        def build(t):
            if isinstance(t, int):
                return t
            ids = tuple(build(c) for c in t)
            ch.append(ids)
            return n + len(ch) - 1
        build(nested)
        return cls(n, ch)

    def levels(self):
        """Internal nodes grouped by height, for vectorized bottom-up passes over a binary
        tree: a list of (node ids, left child ids, right child ids). Every child of a group is
        a leaf or belongs to an earlier group."""
        if getattr(self, "_levels", None) is None:
            if self.multiterm:
                raise ValueError("multi-term fused nodes have no IEEE model")
            total, height, groups = self.n_total, {}, {}
            for k, (a, b) in enumerate(self.children):
                h = 1 + max(height.get(a, 0), height.get(b, 0))
                height[total + k] = h
                groups.setdefault(h, []).append((total + k, a, b))
            self._levels = [tuple(list(col) for col in zip(*groups[h])) for h in sorted(groups)]
        return self._levels

    def to_nested(self):
        total = self.n_total

        def nest(c):
            if c < total:
                return c if c < self.n else "0"
            return tuple(nest(x) for x in self.children[c - total])
        return nest(total + len(self.children) - 1) if self.children else 0

    def __eq__(self, other):
        return isinstance(other, SumTree) and self.n == other.n and \
            self.n_zero == other.n_zero and self.children == other.children

    def __hash__(self):
        return hash((self.n, self.n_zero, tuple(self.children)))

    def __repr__(self):
        return f"SumTree({self.name or self.n}, nodes={len(self.children)})"


# --------------------------------------------------------------------------------------------
# Order models
# --------------------------------------------------------------------------------------------

FP_MODES = ("fma", "separate", "either")


@dataclass(frozen=True)
class Known:
    """Strategy 1: the tree is established (static evidence or accepted FPRev evidence)."""
    tree: SumTree
    fp: str = "fma"             # how products enter the tree: fma | separate | either
    evidence: str = ""


@dataclass(frozen=True)
class AnyTree:
    """Strategy 2: any binary tree over the leaves, any leaf order."""
    fp: str = "either"


@dataclass(frozen=True)
class Depth:
    """Strategy 2: any binary tree of depth <= depth."""
    depth: int
    fp: str = "either"


@dataclass(frozen=True)
class Chunks:
    """Disjoint leaf ranges, each reduced by its own order, then the chunk sums combined."""
    ranges: Tuple[Tuple[int, int], ...]
    inner: Tuple[object, ...]
    combine: object = AnyTree("separate")


Order = Union[Known, AnyTree, Depth, Chunks]


def describe(order) -> str:
    if isinstance(order, Known):
        return f"known {order.tree.name or 'tree'} ({order.fp})"
    if isinstance(order, AnyTree):
        return f"any tree ({order.fp})"
    if isinstance(order, Depth):
        return f"any tree of depth <= {order.depth} ({order.fp})"
    if isinstance(order, Chunks):
        inner = describe(order.inner[0]) if order.inner else "-"
        return f"{len(order.ranges)} chunks of [{inner}], combined by {describe(order.combine)}"
    return str(order)


# --------------------------------------------------------------------------------------------
# Bounds
# --------------------------------------------------------------------------------------------

def _as_cols(X) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    if np.any(X < 0) or not np.all(np.isfinite(X)):
        raise ValueError("leaf bounds must be finite and non-negative")
    return X


def _node_rho(f: Fmt, W, n: int, products: bool):
    """Local error bound of an internal node whose leaves have total magnitude bound W."""
    if products:
        M = (1 + f.gamma(n + 1)) * (W + n * f.eta)
        return f.u * M + f.eta            # the node may be an FMA: subnormal results round
    return f.u * (1 + f.gamma(n)) * W     # additions are exact when the result is subnormal


def _leaf_rho(f: Fmt, X):
    return f.u * X + f.eta


def known_tree_bound(tree: SumTree, X, f: Fmt, products: bool, fp: str = "fma",
                     skip: Optional[set] = None, skip_leaf_rounding: bool = False):
    """Rounding error bound of one known tree, per column. `skip` holds internal node ids whose
    local errors cancel (relational case); their subtrees are skipped entirely."""
    X = _as_cols(X)
    if X.shape[0] != tree.n:
        raise ValueError(f"tree has {tree.n} leaves, got {X.shape[0]} leaf bounds")
    if tree.multiterm:
        raise ValueError("multi-term fused nodes have no IEEE model")
    n, total = tree.n, tree.n_total
    C = X.shape[1]
    W = np.zeros((total + len(tree.children), C))
    W[:n] = X
    err = np.zeros(C)
    inside_skip = np.zeros(total + len(tree.children), dtype=bool)
    for s in skip or ():
        inside_skip[_subtree_nodes(tree, s)] = True
    for k, ch in enumerate(tree.children):
        node = total + k
        W[node] = W[list(ch)].sum(0)
        if inside_skip[node]:
            continue
        if products and fp == "fma" and not skip_leaf_rounding:
            # One product per FMA is exact. When both children are products the tree does not
            # say which one is rounded first, so the larger one is charged.
            real = [c for c in ch if c < n]
            if len(real) >= 2:
                err = err + _leaf_rho(f, np.max(X[real], axis=0))
        err = err + _node_rho(f, W[node], n, products)
    if products and fp != "fma" and not skip_leaf_rounding:
        counted = ~inside_skip[:n]
        err = err + (_leaf_rho(f, X) * counted[:, None]).sum(0)
    return err * SAFETY


def _subtree_nodes(tree: SumTree, node: int):
    total = tree.n_total
    stack, out = [node], []
    while stack:
        c = stack.pop()
        out.append(c)
        if c >= total:
            stack.extend(tree.children[c - total])
    return out


def _subtree_keys(tree: SumTree, intern: dict, products: bool, fp: str):
    """Exact structural key per node: equal keys mean the same IEEE computation from the same
    leaves. Sums are commutative in IEEE arithmetic, so children are sorted. With FMA, a node
    over two products is not shareable: which product is rounded is not revealed by the tree."""
    total = tree.n_total
    keys = [("L", i) if i < tree.n else ("Z",) for i in range(total)]
    out = []
    for k, ch in enumerate(tree.children):
        ck = [keys[c] if c < total else out[c - total] for c in ch]
        if any(x is None for x in ck) or \
                (products and fp == "fma" and sum(1 for c in ch if c < tree.n) > 1):
            out.append(None)
            continue
        key = tuple(sorted(ck, key=repr))
        out.append(intern.setdefault(key, ("N", len(intern))))
    return out


def relational_diff(t1: SumTree, t2: SumTree, X, f: Fmt, products: bool, fp1: str, fp2: str):
    """Bound on |s1 - s2| when both trees are known and both programs read bitwise-equal leaves.

    Maximal subtrees that both trees compute identically (same leaf sets, same shape, same FP
    model) produce bitwise-equal values in both programs, so they act as shared leaves and their
    local errors cancel. With the "separate" FP model, leaf products are bitwise equal in both
    programs too. Returns (bound per column, number of shared internal nodes)."""
    if fp1 != fp2 or fp1 == "either":
        b = known_tree_bound(t1, X, f, products, fp1) + known_tree_bound(t2, X, f, products, fp2)
        return b, 0
    intern = {}
    k1 = _subtree_keys(t1, intern, products, fp1)
    k2 = _subtree_keys(t2, intern, products, fp2)
    common = ({k for k in k1 if k is not None} & {k for k in k2 if k is not None})
    s1 = {t1.n_total + i for i, k in enumerate(k1) if k in common}
    s2 = {t2.n_total + i for i, k in enumerate(k2) if k in common}
    leaf_cancel = products and fp1 == "separate"
    b1 = known_tree_bound(t1, X, f, products, fp1, skip=s1, skip_leaf_rounding=leaf_cancel)
    b2 = known_tree_bound(t2, X, f, products, fp2, skip=s2, skip_leaf_rounding=leaf_cancel)
    return b1 + b2, len(s1)


def anytree_bound(X, f: Fmt, products: bool):
    """Bound over every binary tree (and every leaf order) on the leaves, per column."""
    X = _as_cols(X)
    n = X.shape[0]
    if n <= 1:
        return (_leaf_rho(f, X).sum(0) if products else np.zeros(X.shape[1])) * SAFETY
    s = np.sort(X, axis=0)
    W = s.sum(0)
    prefix = np.cumsum(s, axis=0)                    # prefix[j-1] = s_1 + ... + s_j
    # node weights W, W - s_1, ..., W - (s_1 + ... + s_{n-2}): n-1 nodes
    weights_sum = (n - 1) * W - prefix[: n - 2].sum(0)
    if products:
        M_sum = (1 + f.gamma(n + 1)) * (weights_sum + (n - 1) * n * f.eta)
        node = f.u * M_sum + (n - 1) * f.eta
        node = node + _leaf_rho(f, X).sum(0)         # products may be rounded on their own
    else:
        node = f.u * (1 + f.gamma(n)) * weights_sum
    return node * SAFETY


def depth_bound(X, depth: int, f: Fmt, products: bool):
    """Every leaf passes through at most `depth` nodes, so sum_k W_k <= depth * W."""
    X = _as_cols(X)
    n = X.shape[0]
    W = X.sum(0)
    nodes = max(n - 1, 0)
    if products:
        b = f.u * (1 + f.gamma(n + 1)) * (depth * W + nodes * n * f.eta) + nodes * f.eta
        b = b + _leaf_rho(f, X).sum(0)
    else:
        b = f.u * (1 + f.gamma(n)) * depth * W
    return np.minimum(b * SAFETY, anytree_bound(X, f, products))


def sequential_bound(X, f: Fmt, products: bool, fp: str = "fma", zero_init: bool = True):
    """known_tree_bound for SumTree.sequential, in O(n) per column without building the tree."""
    X = _as_cols(X)
    n = X.shape[0]
    P = np.cumsum(X, axis=0)
    weights = P if zero_init else P[1:]
    err = _node_rho(f, weights, n, products).sum(0)
    if products and fp != "fma":
        err = err + _leaf_rho(f, X).sum(0)
    elif products and not zero_init and n > 1:
        err = err + _leaf_rho(f, np.max(X[:2], axis=0))   # fma over two products
    return err * SAFETY


def bound(order, X, f: Fmt, products: bool):
    """Rounding error bound of `order` (Known or admissible set), per column."""
    X = _as_cols(X)
    if isinstance(order, Known):
        t = order.tree
        if t.seq_order is not None:
            return sequential_bound(X[t.seq_order], f, products, order.fp,
                                    zero_init=t.n_zero == 1)
        return known_tree_bound(t, X, f, products, order.fp)
    if isinstance(order, AnyTree):
        return anytree_bound(X, f, products)
    if isinstance(order, Depth):
        return depth_bound(X, order.depth, f, products)
    if isinstance(order, Chunks):
        inner, sums = [], []
        n = X.shape[0]
        for (a, b), o in zip(order.ranges, order.inner):
            if b <= a:
                sums.append(np.zeros(X.shape[1]))     # an empty chunk contributes an exact 0
                continue
            Xc = X[a:b]
            inner.append(bound(o, Xc, f, products))
            g = f.gamma(b - a + 1)
            sums.append((1 + g) * (Xc.sum(0) + (b - a) * f.eta))
        S = np.stack(sums)
        total = np.sum(inner, axis=0) if inner else np.zeros(X.shape[1])
        if n and S.shape[0] > 1:
            total = total + bound(order.combine, S, f, products=False)
        return total * SAFETY
    raise TypeError(f"unknown order model {order!r}")


def same_known(o1, o2) -> bool:
    return isinstance(o1, Known) and isinstance(o2, Known) and o1.tree.n == o2.tree.n


def relational(o1, o2, X, f: Fmt, products: bool):
    """Strategy 1 relational bound for two known orders on bitwise-equal leaves.
    Returns (bound per column, note)."""
    t1, t2 = o1.tree, o2.tree
    if t1 == t2 and o1.fp == o2.fp and o1.fp != "either":
        return np.zeros(_as_cols(X).shape[1]), "identical known trees: results are bitwise equal"
    b, shared = relational_diff(t1, t2, X, f, products, o1.fp, o2.fp)
    return b, f"known trees share {shared} subtrees"


# --------------------------------------------------------------------------------------------
# Small helpers for transfer functions
# --------------------------------------------------------------------------------------------

def sup_increasing_times_decreasing(inc, dec, hi: float, cells: int = 4096) -> float:
    """Upper bound of inc(t) * dec(t) on [0, hi] for nonnegative inc nondecreasing and dec
    nonincreasing: on each cell [a, b], inc(t) * dec(t) <= inc(b) * dec(a)."""
    if hi <= 0:
        return float(inc(0.0) * dec(0.0))
    t = np.linspace(0.0, hi, cells + 1)
    return float(np.max(inc(t[1:]) * dec(t[:-1])))
