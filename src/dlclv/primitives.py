"""Bit-exact arithmetic building blocks, written with explicit eager PyTorch operations.

Every function here is a straight-line (or counted-loop) composition of single ATen calls:
one IEEE addition, subtraction, multiplication, cast, comparison or data movement per call.
Nothing is delegated to `sum`, `matmul`, `addcmul` or any other library reduction, so the
dependency order, every rounding and every cast is visible in the code.

`PrimitiveGuard` enforces that mechanically: while it is active, an ATen operation outside the
allowlists below raises `NonPrimitiveOperation`. The reconstruction frontend runs every lowered
kernel under it and reports the operations that were executed.

What is relied on (docs/RECONSTRUCTION.md, "Execution assumptions"): each allowed rounding
operation is the correctly rounded IEEE 754 operation (round to nearest, ties to even) in the
tensor's dtype, subnormals are not flushed, and separate eager calls are never contracted.
The float32 FMA below needs these properties of *float64* addition, subtraction and
multiplication, of `nextafter`, and of the float64 -> float32 cast.

NaN: results that are NaN are NaN, but the sign and payload of a generated NaN differ between
hardware (x86 SSE produces a negative quiet NaN, GPUs a positive one). NaN bits are outside
every guarantee of this module.
"""
from __future__ import annotations

import collections
from typing import Optional, Sequence

import torch
from torch.utils._python_dispatch import TorchDispatchMode


class NonPrimitiveOperation(RuntimeError):
    """A lowered kernel executed an operation that is not an explicit primitive."""


# Operations that round (or may round). Exact overloads: `add(alpha != 1)` is checked below.
# `sqrt` is one of the five basic IEEE 754 operations and is correctly rounded on x86 and in
# CUDA; it is used only by the TIR interpreter, where the lowered program itself calls it.
ROUNDING_OPS = frozenset("""
aten.add.Tensor aten.add.Scalar aten.sub.Tensor aten.sub.Scalar
aten.mul.Tensor aten.mul.Scalar aten.div.Tensor aten.div.Scalar
aten._to_copy.default aten.nextafter.default aten.sqrt.default
""".split())

# Operations whose result bits do not depend on the implementation: data movement, selection,
# comparison, integer bit operations, creation of constants, gather/scatter by index, and the
# integer index arithmetic of the TIR interpreter (`floor_divide`, `remainder`: exact on
# integers, which is the only way they are used). Listed by family (all overloads).
EXACT_FAMILIES = frozenset("aten." + n for n in """
view _unsafe_view reshape permute transpose t select slice expand unsqueeze squeeze clone alias
detach stack cat constant_pad_nd zeros zeros_like ones ones_like full full_like new_zeros
eq ne lt le gt ge abs neg bitwise_and bitwise_or bitwise_not logical_and logical_or logical_not
where isfinite isnan lift_fresh lift_fresh_copy contiguous scalar_tensor _local_scalar_dense
index index_put arange floor_divide remainder
""".split())


def op_family(name: str) -> str:
    parts = name.split(".")
    return ".".join(parts[:2])


class PrimitiveGuard(TorchDispatchMode):
    """Fail closed on any ATen call outside `ROUNDING_OPS` and `EXACT_FAMILIES`.

    `counts` is the histogram of executed operations (overload names)."""

    def __init__(self, counts: Optional[collections.Counter] = None):
        super().__init__()
        self.counts = collections.Counter() if counts is None else counts

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = str(func)
        if name not in ROUNDING_OPS and op_family(name) not in EXACT_FAMILIES:
            raise NonPrimitiveOperation(f"{name} is not an explicit primitive operation")
        if op_family(name) in ("aten.add", "aten.sub") and kwargs.get("alpha", 1) != 1:
            raise NonPrimitiveOperation(f"{name} with alpha != 1 is a multiply-add")
        if name == "aten._to_copy.default" and kwargs.get("device") is not None and \
                args and torch.device(kwargs["device"]) != args[0].device:
            raise NonPrimitiveOperation("device transfer inside a lowered kernel")
        self.counts[name] += 1
        return func(*args, **kwargs)


# --------------------------------------------------------------------------------------------
# Error-free addition and the float32 fused multiply-add
# --------------------------------------------------------------------------------------------

def two_sum(a, b):
    """Knuth's TwoSum: s = fl(a + b) and e with a + b = s + e exactly (finite, no overflow)."""
    s = a + b
    bb = s - a
    aa = s - bb
    da = a - aa
    db = b - bb
    e = da + db
    return s, e


def fma32(a, b, c):
    """fl32(a * b + c) with one rounding: the IEEE 754 binary32 fusedMultiplyAdd.

    a, b, c are float32 tensors (broadcastable). Steps, each a separate operation:

    1. p = a * b in float64. Exact: two 24-bit significands give at most 48 bits, and the
       exponent range of float32 products lies inside float64's normal range.
    2. (s, e) = TwoSum(p, c) in float64: p + c = s + e exactly, s = fl64(p + c).
    3. Round to odd: if e != 0 and the last significand bit of s is 0, move s one float64 ulp
       towards s + e. The result r is the float64 neighbour of p + c with an odd significand,
       or p + c itself when that is representable.
    4. Cast r to float32. Because float64 carries at least two more bits than float32, rounding
       r to nearest equals rounding the exact p + c to nearest (Boldo and Melquiond, "Emulation
       of a FMA and correctly rounded sums: proved algorithms using rounding to odd", 2008).

    A plain `(a.double() * b.double() + c.double()).float()` rounds twice and is wrong when
    fl64(p + c) lands exactly on a float32 rounding boundary; see tests/test_primitives.py.

    Non-finite: if s is not finite it is cast unchanged (inf stays inf; NaN stays NaN with an
    unspecified payload). Signed zeros follow IEEE: the float64 sum has the sign rules of the
    fused operation.
    """
    f64 = torch.float64
    p = a.to(f64) * b.to(f64)
    s, e = two_sum(p, c.to(f64))
    inexact = e != 0                                   # False where e is 0; True for NaN
    even = (s.view(torch.int64) & 1) == 0
    inf = torch.full_like(s, float("inf"))
    finite = s.abs() != inf                            # NaN passes; it stays NaN below
    toward = torch.where(e > 0, inf, -inf)
    moved = torch.nextafter(s, toward)
    r = torch.where(inexact & even & finite, moved, s)
    return r.to(torch.float32)


# --------------------------------------------------------------------------------------------
# Dot products with an explicit accumulation order
# --------------------------------------------------------------------------------------------

def _zeros_mn(a, b, dtype=None):
    """+0 accumulator of the shape `a[..., :, :1] * b[..., :1, :]` broadcasts to."""
    lead = torch.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    return torch.zeros((*lead, a.shape[-2], b.shape[-1]), dtype=dtype or a.dtype, device=a.device)


def chain_dot(a, b, ks: Optional[Sequence[int]] = None, fp: str = "fma", acc=None):
    """acc = +0; for k in ks: acc = acc (+) a[..., :, k] * b[..., k, :].

    a is [..., M, K], b is [..., K, N]; the result is [..., M, N]. One accumulator per output
    element, updated in the order `ks` (default 0..K-1): a sequential chain, the order of
    `for k: acc += a[m, k] * b[k, n]`. `fp` says how each step rounds:

    * "fma":      acc = fl(a * b + acc), one rounding (`fma32`); float32 only.
    * "separate": acc = fl(acc + fl(a * b)), two roundings, in the dtype of the operands.
    """
    if fp not in ("fma", "separate"):
        raise ValueError(f"unknown product model {fp!r}")
    if fp == "fma" and (a.dtype != torch.float32 or b.dtype != torch.float32):
        raise TypeError("the fused multiply-add primitive is float32 only")
    if acc is None:
        acc = _zeros_mn(a, b)
    for k in (range(a.shape[-1]) if ks is None else ks):
        ak = a[..., :, k:k + 1]
        bk = b[..., k:k + 1, :]
        if fp == "fma":
            acc = fma32(ak, bk, acc)
        else:
            acc = acc + ak * bk
    return acc


def tree_dot(tree, a, b, fused: Optional[Sequence[bool]] = None,
             zero_init: Optional[Sequence[bool]] = None, bias=None, zero_root: bool = False):
    """Dot products a[..., M, K] @ b[..., K, N] accumulated along the binary tree `tree`
    (`fpmodel.SumTree` over K product leaves; the same tree for every output element).

    Leaf i is the product a[..., i] * b[i, ...]:

    * `fused[i]`: the product enters its parent unrounded, parent = fma(a_i, b_i, sibling);
    * otherwise it is rounded on its own; with `zero_init[i]` it is fl(a_i * b_i + (+0)), an
      accumulator that started at +0 (differs from the bare product only for a -0 product).

    Zero leaves of the tree are +0. Internal nodes without a fused child are one float
    addition. `zero_root` adds +0 to the root: some accumulator above the leaves started at +0,
    so a sum of -0 terms is +0 (where that +0 joins cannot be observed and does not matter).
    `bias` is then added with one more rounding. This is the tensor version
    of the exact rational emulator `fprev.emulate_dot`, with signed zeros.
    """
    n, total = tree.n, tree.n_total
    if a.shape[-1] != n or b.shape[-2] != n:
        raise ValueError(f"tree has {n} leaves, operands have K = {a.shape[-1]}, {b.shape[-2]}")
    if tree.multiterm:
        raise ValueError("multi-term fused nodes have no IEEE model")
    fused = [False] * n if fused is None else list(fused)
    zero_init = [False] * n if zero_init is None else list(zero_init)
    zeros = _zeros_mn(a, b)

    def operands(i):
        return a[..., :, i:i + 1], b[..., i:i + 1, :]

    def leaf(i):
        if i >= n:
            return zeros
        ai, bi = operands(i)
        p = ai * bi
        return p + zeros if zero_init[i] else p

    vals = {}

    def value(c):
        return vals.pop(c) if c >= total else leaf(c)

    for k, (lhs, rhs) in enumerate(tree.children):
        fl, fr = lhs < n and fused[lhs], rhs < n and fused[rhs]
        if fl and fr:
            raise ValueError("two fused products under one node: not an FMA")
        if fl or fr:
            f, other = (lhs, rhs) if fl else (rhs, lhs)
            if zero_init[f]:
                raise ValueError("a fused product has no accumulator of its own")
            af, bf = operands(f)
            vals[total + k] = fma32(af, bf, value(other))
        else:
            vals[total + k] = value(lhs) + value(rhs)
    out = vals[total + len(tree.children) - 1] if tree.children else leaf(0)
    if zero_root:
        out = out + zeros
    if bias is not None:
        out = out + bias
    return out


def tree_sum(tree, x, dim: int, zero_init: bool = False, keepdim: bool = False):
    """Sum `x` along `dim` in the order of the binary tree `tree` (one addition per node).
    `zero_init`: the kernel's accumulator starts at +0, so an all -0 input sums to +0."""
    n, total = tree.n, tree.n_total
    if x.shape[dim] != n:
        raise ValueError(f"tree has {n} leaves, dimension {dim} has {x.shape[dim]}")
    if tree.multiterm:
        raise ValueError("multi-term fused nodes have no IEEE model")
    zeros = torch.zeros_like(x.select(dim, 0))
    vals = {}

    def value(c):
        if c >= total:
            return vals.pop(c)
        return x.select(dim, c) if c < n else zeros

    for k, (lhs, rhs) in enumerate(tree.children):
        vals[total + k] = value(lhs) + value(rhs)
    out = vals[total + len(tree.children) - 1] if tree.children else x.select(dim, 0)
    if zero_init:
        out = out + zeros
    return out.unsqueeze(dim) if keepdim else out


# --------------------------------------------------------------------------------------------
# Data movement
# --------------------------------------------------------------------------------------------

def im2col(x, kernel, dilation=(1, 1), padding=(0, 0), stride=(1, 1)):
    """`F.unfold` as explicit zero padding and strided slices: x[N, C, H, W] ->
    [N, C*kh*kw, Ho*Wo], channel-major, kernel positions and output positions row-major."""
    (kh, kw), (dh, dw), (ph, pw), (sh, sw) = kernel, dilation, padding, stride
    N, C, H, W = x.shape
    Ho = (H + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    Wo = (W + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    xp = torch.constant_pad_nd(x, (pw, pw, ph, ph), 0.0) if (ph or pw) else x
    patches = []
    for i in range(kh):
        for j in range(kw):
            patches.append(xp[:, :, i * dh: i * dh + sh * (Ho - 1) + 1: sh,
                              j * dw: j * dw + sw * (Wo - 1) + 1: sw])
    cols = torch.stack(patches, dim=2)                 # [N, C, kh*kw, Ho, Wo]
    return cols.reshape(N, C * kh * kw, Ho * Wo)
