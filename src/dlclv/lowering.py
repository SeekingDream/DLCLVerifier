"""Kernel lowerings: replace one library or DSL kernel call by explicit arithmetic.

A lowering looks at one ATen call of the program (with its concrete operands) and either
returns a `LoweredOp`, a function built only from `primitives` that computes the same bits, or
declines. It must say where the arithmetic order comes from:

* level "known":     read from the kernel's source (static) and corroborated on the device
                     (FPRev masks, FMA probe, bitwise validation of the lowered function);
* level "source":    read from the kernel's source only (probing disabled);
* level "validated": inferred from the kernel's behaviour only (FPRev masks, product and
                     signed-zero probes), then validated bitwise at the exact shapes. Dynamic
                     evidence: an assumption, never a proof (docs/RECONSTRUCTION.md).

A lowering that cannot establish the order raises `LoweringFailed`; it never substitutes a
generic `sum` or `matmul`.

Implemented:

* `DslGemmLowering`: the Triton and TileLang GEMM kernels of kernels.py, as called by
  `compilers.GemmMode` (linear, matmul, mm, bmm, addmm, conv1d/conv2d via im2col, split-K).
* `ProbedLinearLowering`: a PyTorch eager `F.linear` (BLAS, cuBLAS), order rebuilt by
  `fprev.probe_eager_linear`.
* `ProbedConvLowering`: a PyTorch eager `conv1d` / `conv2d` (cuDNN or CPU), tree rebuilt by
  `probe_conv`.
* `ProbedSumLowering`: a PyTorch eager `sum` / `mean` over given dimensions and
  `adaptive_avg_pool2d` to 1x1, order rebuilt by FPRev.

Kernels without an accumulation order (clamps, max pooling, inference batch normalization)
are in `eager_lowering.py`.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from . import primitives as P
from .bitwise import bitwise_equal
from .fpmodel import FORMATS, SumTree
from .fprev import (RevealError, detect_fma, mask_value, probe_eager_linear,
                    reveal_order_batched, tree_depth)


class LoweringFailed(RuntimeError):
    """The arithmetic of a kernel call could not be established."""


@dataclass
class LoweredOp:
    fn: Callable                       # (*args, **kwargs) of the ATen call -> same result
    kernel: str                        # what was lowered
    description: str                   # the explicit arithmetic, in words
    level: str                         # known | source | validated
    evidence: Dict[str, object] = field(default_factory=dict)
    assumptions: List[str] = field(default_factory=list)
    # Optional (*args, **kwargs) check run before `fn`, outside the primitive guard: raise
    # `LoweringFailed` if constants the lowering depends on are not the ones it was built for.
    check: Optional[Callable] = None


class Lowering:
    """Interface. `lower` is called once per call_function node, in program order, with the
    concrete operands of the capture run. Return None to leave the node alone."""

    name = "lowering"

    def lower(self, node_name: str, target, args, kwargs) -> Optional[LoweredOp]:
        raise NotImplementedError

    def finish(self):
        """Called after the capture run; raise if the program was not fully accounted for."""

    failures: Dict[str, str]


def bind(target, args, kwargs) -> dict:
    """Arguments of an ATen overload by schema name, defaults filled in."""
    out = {}
    for i, a in enumerate(target._schema.arguments):
        if i < len(args):
            out[a.name] = args[i]
        elif a.name in kwargs:
            out[a.name] = kwargs[a.name]
        elif a.has_default_value():
            out[a.name] = a.default_value
        else:
            out[a.name] = None
    return out


def _name(target) -> str:
    s = str(target)
    return ".".join(s.split(".")[:2])


# --------------------------------------------------------------------------------------------
# Validation inputs: the cases where two accumulation orders or product models disagree
# --------------------------------------------------------------------------------------------

KINDS = ("normal", "spread", "cancel", "tiny", "zeros")


def adversarial_tensor(shape, kind: str, gen, dtype, device, dim: int = -1):
    """One validation tensor. `spread`: exponents over 2^+-20; `cancel`: the second half along
    `dim` nearly negates the first; `tiny`: subnormals and values whose products underflow;
    `zeros`: about half the entries are +0 or -0, and the first slice is entirely -0."""
    import torch
    x = torch.randn(shape, generator=gen, dtype=torch.float64)
    if kind == "spread":
        x = x * 2.0 ** torch.randint(-20, 21, shape, generator=gen).double()
    elif kind == "cancel":
        n = x.shape[dim]
        if n >= 2:
            h = n // 2
            head = x.narrow(dim, 0, h)
            x = torch.cat([head, -head * (1 + 2.0 ** -20), x.narrow(dim, 2 * h, n - 2 * h)], dim)
            perm = torch.randperm(n, generator=gen)
            x = x.index_select(dim, perm)
    elif kind == "tiny":
        tiny = float(np.finfo(np.dtype(str(dtype).replace("torch.", ""))).tiny)
        x = x * tiny ** 0.5 * 2.0 ** torch.randint(-12, 3, shape, generator=gen).double()
    elif kind == "zeros":
        mask = torch.rand(shape, generator=gen) < 0.5
        sign = torch.where(torch.rand(shape, generator=gen) < 0.5, -0.0, 0.0).double()
        x = torch.where(mask, sign, x)
        if x.dim() and x.shape[0] > 0:
            x[0] = -0.0
    elif kind != "normal":
        raise ValueError(kind)
    return x.to(dtype).to(device)


def _mismatch(got, want, what: str) -> str:
    import torch
    nan = torch.isnan(got) | torch.isnan(want)
    diff = (got != want) | (torch.signbit(got) != torch.signbit(want))
    bad = int((diff & ~nan).sum())
    d = float((got.double() - want.double())[~nan].abs().max()) if bool((~nan).any()) else 0.0
    return (f"{what}: the explicit arithmetic differs from the kernel in {bad} of "
            f"{got.numel()} elements (max |diff| {d:.3g}, NaN in {int(nan.sum())})")


def validate_bitwise(kernel: Callable, lowered: Callable, make_inputs: Callable, trials: int,
                     what: str) -> int:
    """Run both on `trials` adversarial inputs; raise `LoweringFailed` on the first bit that
    differs. An input whose kernel result has a NaN is rejected too: validation inputs are
    built to stay finite, so a NaN means the kernel does something unexpected."""
    import torch
    for t in range(trials):
        inputs = make_inputs(t)
        with torch.no_grad():
            want = kernel(*inputs)
            got = lowered(*inputs)
        if bool(torch.isnan(want).any()) or not bitwise_equal(got, want):
            raise LoweringFailed(_mismatch(got, want, f"{what}, validation input {t} "
                                                      f"({KINDS[t % len(KINDS)]})"))
    return trials


# --------------------------------------------------------------------------------------------
# A library sum over one dimension (also the split-K combine of the DSL paths)
# --------------------------------------------------------------------------------------------

_SUM_CACHE: Dict[tuple, dict] = {}


CLASS_LIMIT = 64          # largest n for the per-position reveal (n(n-1)/2 kernel calls)


def _sum_classes(kernel, shaped, n, cells, Mv, dtype, device, lca):
    """Position classes: vectorized reductions give the lanes and the scalar tail different
    orders. Every pair (i, j) is asked at every position at once (basic FPRev, n(n-1)/2 calls);
    positions with the same answers share a tree. Returns [(tree, bool mask over positions)]."""
    import torch
    from .fprev import reveal_order
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    sizes = torch.empty((len(pairs), cells), dtype=torch.float64)
    for q, (i, j) in enumerate(pairs):
        flat = torch.ones((cells, n), dtype=dtype)
        flat[:, i], flat[:, j] = Mv, -Mv
        sizes[q] = lca(kernel(shaped(flat).to(device)).reshape(cells).double().cpu())
    index = {p: q for q, p in enumerate(pairs)}
    groups = {}
    for c in range(cells):
        groups.setdefault(tuple(sizes[:, c].tolist()), []).append(c)
    classes = []
    for column, members in groups.items():

        def masked_sum(x, column=column):          # replay the measured sizes for reveal_order
            i, j = int(np.argmax(x)), int(np.argmin(x))
            return n - column[index[min(i, j), max(i, j)]]

        tree = reveal_order(masked_sum, n, str(dtype).replace("torch.", ""))
        mask = torch.zeros(cells, dtype=torch.bool)
        mask[members] = True
        classes.append((tree, mask))
    return classes


REDUCTIONS = ("sum", "mean", "adaptive_avg_pool2d")


def probe_sum(shape, dim, dtype, device, trials: int = 5, seed: int = 0,
              op: str = "sum") -> dict:
    """Order of a library reduction of a contiguous `x` of exactly this shape, dtype and device.

    `dim` is one dimension or a list of them; `op` is the call that is probed:

    * "sum":  `torch.sum(x, dim)`;
    * "mean": `torch.mean(x, dim)`, modelled as the sum followed by one division by the
      number of elements n;
    * "adaptive_avg_pool2d": `F.adaptive_avg_pool2d(x, 1)` (dim must be the last two), the
      same model.

    The reduced dimensions are moved last and flattened row-major to n leaves (data movement
    only); the tree is over those n leaves.

    1. FPRev: masked inputs (+M, -M, ones) give the size of the lowest common ancestor of two
       leaves; the tree is rebuilt from them. Every other position of the tensor carries its
       own masked pair, so one call asks many pairs. If the answers are inconsistent and there
       are at most `CLASS_LIMIT` leaves, the order is revealed per position (`_sum_classes`).
       For the two averages the kernel returns (n - size) / n; the size is recovered by
       multiplying by n and must be an integer to within 1e-3.
    2. Signed zero: an all -0 input tells whether an accumulator starts at +0.
    3. Validation: `primitives.tree_sum` on the tree(s), then `/ n` for an average, must
       reproduce the kernel bit for bit at every element, on adversarial inputs.

    Returns {"apply": explicit reduction, "tree": the tree if it is the same everywhere, ...};
    raises `LoweringFailed` otherwise."""
    import torch
    import torch.nn.functional as F
    shape = tuple(int(s) for s in shape)
    dims = sorted({int(d) % len(shape) for d in ([dim] if isinstance(dim, int) else dim)})
    name = str(dtype).replace("torch.", "")
    key = (shape, tuple(dims), name, str(device), torch.get_num_threads(), trials, seed, op)
    if key in _SUM_CACHE:
        return _SUM_CACHE[key]
    if name not in FORMATS:
        raise LoweringFailed(f"no FPRev mask for {name} (float32 and float64 only)")
    if op not in REDUCTIONS or not dims:
        raise LoweringFailed(f"unsupported reduction {op!r} over {dims}")
    if op == "adaptive_avg_pool2d" and (len(shape) != 4 or dims != [2, 3]):
        raise LoweringFailed("adaptive_avg_pool2d is lowered only for output size 1 on NCHW")
    kept = [d for d in range(len(shape)) if d not in dims]
    perm = kept + dims
    inverse = [perm.index(d) for d in range(len(shape))]
    other = tuple(shape[d] for d in kept)
    red = tuple(shape[d] for d in dims)
    n = int(np.prod(red))
    cells = int(np.prod(other)) if other else 1
    scale = 1 if op == "sum" else n
    what = (f"torch.{op if op != 'adaptive_avg_pool2d' else 'nn.functional.' + op} over "
            f"dim{'s' if len(dims) > 1 else ''} {dims[0] if len(dims) == 1 else dims} of "
            f"{name}{list(shape)} on {device}")
    if op != "sum":
        what += " as sum / n"

    def shaped(flat):                        # [cells, n] -> the kernel's contiguous layout
        return flat.reshape(*other, *red).permute(inverse).contiguous()

    def leaves(x):                           # the kernel's layout -> [*other, n]
        return x.permute(perm).reshape(*other, n)

    def kernel(x):
        with torch.no_grad():
            if op == "sum":
                return torch.sum(x, dims)
            if op == "mean":
                return torch.mean(x, dims)
            return F.adaptive_avg_pool2d(x, 1).reshape(other)

    def lca(y):                              # kernel results (float64) -> LCA sizes
        l = n - y * scale
        r = torch.round(l)
        if bool(((l - r).abs() > (0 if op == "sum" else 1e-3)).any()):
            raise RevealError(f"the kernel returned {float(y.flatten()[0])} for a masked pair")
        return r

    classes = None
    try:
        if n == 1:
            tree = SumTree(1, [])
        else:
            Mv = mask_value(FORMATS[name], n)

            def lca_sizes(pairs):
                out = []
                for s in range(0, len(pairs), cells):
                    chunk = pairs[s:s + cells]
                    flat = torch.ones((cells, n), dtype=dtype)
                    for c, (i, j) in enumerate(chunk):
                        flat[c, i], flat[c, j] = Mv, -Mv
                    y = lca(kernel(shaped(flat).to(device)).reshape(cells).double().cpu())
                    out += [int(v) for v in y[:len(chunk)]]
                return out

            try:
                tree = reveal_order_batched(lca_sizes, n)
            except RevealError:
                if cells == 1 or n > CLASS_LIMIT:
                    raise
                classes = _sum_classes(kernel, shaped, n, cells, Mv, dtype, device, lca)
                tree = classes[0][0] if len(classes) == 1 else None
        if any(t.multiterm for t in ([tree] if classes is None else [c[0] for c in classes])):
            raise RevealError("multi-term fused nodes: no IEEE model")
    except RevealError as e:
        raise LoweringFailed(f"{what}: {e}") from e
    neg = kernel(torch.full(shape, -0.0, dtype=dtype, device=device))
    signs = torch.signbit(neg)
    if bool(signs.any()) != bool(signs.all()) or bool((neg != 0).any()):
        raise LoweringFailed(f"{what}: an all -0 tensor does not reduce to a uniform zero")
    zero_init = not bool(signs.all())
    masks = [] if classes is None else [m.reshape(other).to(device) for _, m in classes]
    keep_shape = tuple(1 if d in dims else shape[d] for d in range(len(shape)))

    def apply(x, keepdim=False):
        flat = leaves(x)
        if classes is None or len(classes) == 1:
            out = P.tree_sum(tree, flat, -1, zero_init)
        else:
            out = P.tree_sum(classes[0][0], flat, -1, zero_init)
            for (t, _), m in zip(classes[1:], masks[1:]):
                out = torch.where(m, P.tree_sum(t, flat, -1, zero_init), out)
        if op != "sum":
            out = out / n
        return out.reshape(keep_shape) if keepdim else out

    gen = torch.Generator().manual_seed(seed)
    validate_bitwise(kernel, apply,
                     lambda t: (adversarial_tensor(shape, KINDS[t % len(KINDS)], gen, dtype,
                                                   device, dims[-1]),), trials, what)
    trees = [tree] if classes is None else [c[0] for c in classes]
    res = {"apply": apply, "tree": tree, "zero_init": zero_init, "kernel": what,
           "leaves": n, "divide_by": None if op == "sum" else n,
           "depth": max(tree_depth(t) for t in trees), "position_classes": len(trees),
           "masked_pairs": getattr(tree, "queries", n * (n - 1) // 2),
           "validation_inputs": trials, "threads": torch.get_num_threads()}
    _SUM_CACHE[key] = res
    return res


class ProbedSumLowering(Lowering):
    """PyTorch eager `sum` / `mean` over given dimensions and `adaptive_avg_pool2d` to 1x1, on
    a contiguous float32/float64 tensor."""

    name = "probed-sum"
    TARGETS = {"aten.sum.dim_IntList": "sum", "aten.mean.dim": "mean",
               "aten.adaptive_avg_pool2d.default": "adaptive_avg_pool2d"}

    def __init__(self, trials: int = 5, seed: int = 0):
        self.trials, self.seed, self.failures = trials, seed, {}

    def lower(self, node_name, target, args, kwargs):
        import torch
        op = self.TARGETS.get(str(target))
        if op is None:
            return None
        a = bind(target, args, kwargs)
        x = a["self"]
        try:
            if op == "adaptive_avg_pool2d":
                if x.dim() != 4 or [int(v) for v in a["output_size"]] != [1, 1]:
                    raise LoweringFailed("adaptive_avg_pool2d only to output size 1x1 on NCHW")
                dims, keep = [2, 3], True
            else:
                dims, keep = a["dim"], bool(a["keepdim"])
                if a["dtype"] is not None or not dims:
                    raise LoweringFailed("only a reduction over given dimensions without a "
                                         "dtype change")
            if not x.is_contiguous() or x.numel() == 0:
                raise LoweringFailed("only contiguous, non-empty operands were probed")
            ev = probe_sum(x.shape, [int(d) for d in dims], x.dtype, x.device, self.trials,
                           self.seed, op)
        except LoweringFailed as e:
            self.failures[node_name] = str(e)
            return None
        apply, zero_init = ev["apply"], ev["zero_init"]
        shape, dtype = tuple(x.shape), x.dtype

        def fn(*args, **kwargs):
            x = bind(target, args, kwargs)["self"]
            if tuple(x.shape) != shape or x.dtype != dtype or not x.is_contiguous():
                raise LoweringFailed("operand differs from the probed shape/dtype/layout")
            return apply(x, keep)

        classes = ev["position_classes"]
        return LoweredOp(
            fn, ev["kernel"],
            f"binary summation tree of depth {ev['depth']} over {ev['leaves']} leaves "
            "(reduced dimensions flattened row-major), one addition per node"
            + (f"; {classes} trees, selected by output position" if classes > 1 else "")
            + (", accumulator starts at +0" if zero_init else "")
            + (f"; then one division by {ev['divide_by']}" if ev["divide_by"] else ""),
            "validated",
            {k: v for k, v in ev.items() if k not in ("tree", "kernel", "apply")},
            [f"order[{ev['kernel']}] inferred by FPRev and validated on "
             f"{ev['validation_inputs']} inputs: fixed for this shape, layout, device, thread "
             "count and library build, and independent of the values (dynamic evidence only)."])


def check_fused(tree: SumTree, fused) -> None:
    """A node may take one product unrounded (an FMA). Two unrounded products under one node
    mean a two-term fused dot product or a wider intermediate: no float32 IEEE model."""
    for lhs, rhs in tree.children:
        if lhs < tree.n and rhs < tree.n and fused[lhs] and fused[rhs]:
            raise RevealError(f"products {lhs} and {rhs} both enter one node unrounded: not a "
                              "float32 FMA (a two-term fused dot or a wider intermediate)")


# --------------------------------------------------------------------------------------------
# PyTorch eager F.linear
# --------------------------------------------------------------------------------------------

_LINEAR_CACHE: Dict[tuple, object] = {}


def probe_linear(x_shape, out_features: int, bias: bool, device, trials: int = 10,
                 seed: int = 0, via: str = "linear") -> dict:
    """Arithmetic of `F.linear(x, W, b)` in PyTorch eager at exactly these shapes.

    `fprev.probe_eager_linear` gives the tree over the K products, which products are fused
    (FMA) and that the bias is added last. Added here:

    * signed zeros: with every other product underflowing to -0, a product that is exactly -0
      shows whether its accumulator started at +0;
    * validation of `primitives.tree_dot` against the kernel, bit for bit at every output
      element, on adversarial inputs including signed zeros and underflow.

    `via="addmm"` / `via="mm"`: the program calls `torch.addmm(b, x, W.t())` or
    `torch.mm(x, W.t())` (how older exports spell a 2-D linear). The tree is revealed through
    `F.linear`; the signed-zero probes and the validation run the call itself, so a different
    kernel behind it is rejected.
    """
    import torch
    import torch.nn.functional as F
    x_shape = tuple(int(d) for d in x_shape)
    K, N = x_shape[-1], int(out_features)
    key = (x_shape, N, bool(bias), str(device), torch.get_num_threads(), trials, seed, via)
    if key in _LINEAR_CACHE:
        got = _LINEAR_CACHE[key]
        if isinstance(got, str):
            raise LoweringFailed(got)
        return got
    what = (f"torch eager {'F.linear' if via == 'linear' else via} x{list(x_shape)} "
            f"W[{N}, {K}] bias={bias} on {device}")
    f32 = torch.float32
    if via not in ("linear", "addmm", "mm") or (via != "linear" and (
            len(x_shape) != 2 or bias != (via == "addmm"))):
        raise LoweringFailed(f"{what}: unsupported call form")

    def kernel(x, W, b=None):
        with torch.no_grad():
            if via == "linear":
                return F.linear(x, W, b)
            return torch.addmm(b, x, W.t()) if via == "addmm" else torch.mm(x, W.t())

    try:
        try:
            ev = probe_eager_linear(x_shape, N, bias, str(device), trials=max(trials, 5),
                                    seed=seed)
        except RevealError as e:
            raise LoweringFailed(f"{what}: {e}") from e
        tree, fused = ev.tree, list(ev.fused)
        try:
            check_fused(tree, fused)
        except RevealError as e:
            raise LoweringFailed(f"{what}: {e}") from e
        # Signed zeros. With x = 2^-80 and W = -2^-80 every product underflows to -0 in any
        # model (rounded, or fused onto a -0 or +0 partner sum of such products), so:
        # 1. if the kernel returns +0, a +0 joins somewhere above the leaves (`zero_root`);
        #    which accumulators started at +0 is then unobservable;
        # 2. otherwise a product that is exactly -0 shows whether its own accumulator started
        #    at +0 (the result turns +0).
        zero_init, zero_root = [False] * K, False
        x = torch.full(x_shape, 2.0 ** -80, dtype=f32, device=device)
        nb = torch.full((N,), -0.0, dtype=f32, device=device) if bias else None
        y = kernel(x, torch.full((N, K), -(2.0 ** -80), dtype=f32, device=device), nb)
        signs = torch.signbit(y)
        if bool((y != 0).any()) or bool(signs.any()) != bool(signs.all()):
            raise LoweringFailed(f"{what}: underflowing products do not give a uniform zero")
        zero_root = not bool(signs.all())
        for s in ([] if zero_root else range(0, K, N)):
            idx = list(range(s, min(K, s + N)))
            W = torch.full((N, K), -(2.0 ** -80), dtype=f32)
            for c, i in enumerate(idx):
                W[c, i] = -0.0
            y = kernel(x, W.to(device), nb).reshape(-1, N)[0].cpu()
            for c, i in enumerate(idx):
                if float(y[c]) != 0.0:
                    raise LoweringFailed(f"{what}: signed-zero probe of product {i} returned "
                                         f"{float(y[c])!r}")
                zero_init[i] = not bool(torch.signbit(y[c]))
                if zero_init[i] and fused[i]:
                    raise LoweringFailed(f"{what}: product {i} looks fused and zero-initialized")

        def lowered(x, W, b=None):
            out = P.tree_dot(tree, x.reshape(1, -1, K), W.t().unsqueeze(0), fused, zero_init,
                             zero_root=zero_root)
            out = out.view(*x.shape[:-1], N)
            return out if b is None else out + b

        gen = torch.Generator().manual_seed(seed)

        def make(t):
            kx = KINDS[t % len(KINDS)]
            kw = "normal" if kx in ("spread", "cancel") else kx
            xs = adversarial_tensor(x_shape, kx, gen, f32, device)
            W = adversarial_tensor((N, K), kw, gen, f32, device)
            if not bias:
                return xs, W
            b = adversarial_tensor((N,), "zeros" if kx == "zeros" else "normal", gen, f32, device)
            return xs, W, b

        validate_bitwise(kernel, lowered, make, trials, what)
    except LoweringFailed as e:
        _LINEAR_CACHE[key] = str(e)
        raise
    res = {"tree": tree, "fused": fused, "zero_init": zero_init, "zero_root": zero_root,
           "kernel": what,
           "depth": tree_depth(tree), "fused_products": sum(fused),
           "zero_init_products": sum(zero_init), "validation_inputs": trials,
           "notes": list(ev.notes), "pinned": dict(ev.pinned),
           "threads": torch.get_num_threads()}
    _LINEAR_CACHE[key] = res
    return res


class ProbedLinearLowering(Lowering):
    """PyTorch eager `aten.linear` on float32 operands (or `aten.addmm(b, x, W.t())` /
    `aten.mm(x, W.t())`, the forms older exports give a 2-D linear), as an explicit tree of the
    K products."""

    name = "probed-linear"

    def __init__(self, trials: int = 10, seed: int = 0):
        self.trials, self.seed, self.failures = trials, seed, {}

    @staticmethod
    def _operands(target, args, kwargs):
        """(x, b[K, N] operand of the dot products, bias, via) or None."""
        a = bind(target, args, kwargs)
        if str(target) == "aten.linear.default":
            W = a["weight"]
            return a["input"], (W.t() if W.dim() == 2 else None), a["bias"], "linear"
        if str(target) == "aten.addmm.default" and a["beta"] == 1 and a["alpha"] == 1:
            return a["mat1"], a["mat2"], a["self"], "addmm"
        if str(target) == "aten.mm.default":
            return a["self"], a["mat2"], None, "mm"
        return None

    def lower(self, node_name, target, args, kwargs):
        import torch
        ops = self._operands(target, args, kwargs)
        if ops is None:
            return None
        x, Wt, b, via = ops
        try:
            if Wt is None or Wt.dim() != 2 or x.dtype != torch.float32 or \
                    Wt.dtype != torch.float32 or x.dim() < 1 or x.numel() == 0:
                raise LoweringFailed("only float32 operands with a 2-D weight were probed")
            if via != "linear" and (x.dim() != 2 or not Wt.t().is_contiguous()
                                    or (b is not None and b.dim() != 1)):
                raise LoweringFailed(f"{via} is lowered only as [bias[N] +] x[M, K] @ W.t() "
                                     "with a contiguous W")
            ev = probe_linear(x.shape, Wt.shape[1], b is not None, x.device, self.trials,
                              self.seed, via)
        except LoweringFailed as e:
            self.failures[node_name] = str(e)
            return None
        tree, fused, zero_init, zero_root = (ev["tree"], ev["fused"], ev["zero_init"],
                                             ev["zero_root"])
        x_shape, (K, N) = tuple(x.shape), tuple(Wt.shape)
        operands = self._operands

        def fn(*args, **kwargs):
            x, Wt, b, _ = operands(target, args, kwargs)
            if tuple(x.shape) != x_shape or tuple(Wt.shape) != (K, N):
                raise LoweringFailed("operands differ from the probed shapes")
            out = P.tree_dot(tree, x.reshape(1, -1, K), Wt.unsqueeze(0), fused, zero_init,
                             zero_root=zero_root)
            out = out.view(*x_shape[:-1], N)
            return out if b is None else out + b

        return LoweredOp(
            fn, ev["kernel"],
            f"tree of depth {ev['depth']} over {K} products: {ev['fused_products']} enter "
            f"their node unrounded (FMA), {ev['zero_init_products']} of the rounded ones on a "
            "+0 accumulator" + ("; +0 added to the root" if zero_root else "")
            + "; bias added last with one rounding", "validated",
            {k: v for k, v in ev.items() if k not in ("tree", "fused", "zero_init", "kernel")},
            [f"order[{ev['kernel']}] inferred by FPRev, product and signed-zero probes and "
             f"validated on {ev['validation_inputs']} inputs: fixed for these shapes, device, "
             "thread count and BLAS build, the same for every output element, and independent "
             "of the values (dynamic evidence only)."])


# --------------------------------------------------------------------------------------------
# Triton and TileLang GEMM kernels (kernels.py), as called by compilers.GemmMode
# --------------------------------------------------------------------------------------------

def _cdiv(a, b):
    return -(-a // b)


def k_ranges(backend: str, K: int, split_k: int, block_k: int):
    """[(k_lo, k_hi, padded)] per partial accumulator: the k range it sums in increasing order
    and whether the kernel runs further steps on masked (zero) operands afterwards.

    triton (compilers.triton_bmm, kernels.triton_bmm_kernel):
        k_per = cdiv(cdiv(K, split_k), BLOCK_K) * BLOCK_K; split s loops
        `for k in range(s*k_per, min((s+1)*k_per, K), BLOCK_K)`, each step a BLOCK_K-wide
        tl.dot whose loads are masked to 0 at kk >= k_hi.
    tilelang (compilers.tilelang_bmm, kernels.tilelang_bmm_kernel):
        split_k > 1 zero-pads K to split_k * Kc with Kc = cdiv(K, split_k) and runs one kernel
        per chunk; the kernel loops `ko < ceildiv(Kc, block_k)`, `k < block_k`, over tiles
        whose out-of-range part is zero.
    """
    out = []
    if backend == "triton":
        k_per = _cdiv(_cdiv(K, split_k), block_k) * block_k
        for s in range(split_k):
            lo, hi = s * k_per, min((s + 1) * k_per, K)
            hi = max(hi, lo)
            out.append((lo, hi, (hi - lo) % block_k != 0))
    elif backend == "tilelang":
        Kc = _cdiv(K, split_k) if split_k > 1 else K
        steps = _cdiv(Kc, block_k) * block_k
        for s in range(split_k):
            lo, hi = min(s * Kc, K), min((s + 1) * Kc, K)
            out.append((lo, hi, steps > hi - lo))
    else:
        raise ValueError(backend)
    return out


def dsl_bmm(a, b, backend: str, split_k: int, block_k: int, combine: Optional[dict] = None):
    """Explicit arithmetic of `triton_bmm` / `tilelang_bmm`: a[Bt, M, K] @ b[Bt, K, N].

    Per output element and partial accumulator: acc = +0, then acc = fma(a[m,k], b[k,n], acc)
    for k increasing over the range (`primitives.chain_dot`). A step on masked operands is
    fma(+0, +0, acc): it leaves acc alone except that it turns -0 into +0, so one such step is
    applied when the kernel runs any. With split_k > 1 the partial sums are stacked on a new
    dimension 1 and added in the order `combine` (`probe_sum` of the `sum(1)` the path calls).
    """
    import torch
    Bt = a.shape[0]
    if b.shape[0] != Bt:
        b = b.expand(Bt, -1, -1)
    K = a.shape[2]
    partials = []
    for lo, hi, padded in k_ranges(backend, K, split_k, block_k):
        acc = P.chain_dot(a, b, range(lo, hi), "fma")
        if padded:
            zero = torch.zeros_like(acc)
            acc = P.fma32(zero, zero, acc)
        partials.append(acc)
    if split_k == 1:
        return partials[0]
    stacked = torch.stack(partials, dim=1)
    if "apply" in combine:
        return combine["apply"](stacked)
    return P.tree_sum(combine["tree"], stacked, 1, combine["zero_init"])


# The handlers of compilers.GemmMode with `bmm` as a parameter. Reshapes, the im2col and the
# bias addition (a separate eager add after the kernel) are part of the program's arithmetic.

def _h_matmul(bmm, a, b):
    import torch
    if b.dim() == 2:
        out = bmm(a.reshape(1, -1, a.shape[-1]), b.unsqueeze(0))
        return out.view(*a.shape[:-1], b.shape[-1])
    batch = torch.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    M, K, N = a.shape[-2], a.shape[-1], b.shape[-1]
    a3 = a.expand(*batch, M, K).reshape(-1, M, K)
    b3 = b.expand(*batch, K, N).reshape(-1, K, N)
    return bmm(a3, b3).view(*batch, M, N)


def _h_addmm(bmm, bias, a, b):
    return _h_matmul(bmm, a, b) + bias


def _h_linear(bmm, x, weight, bias=None):
    out = bmm(x.reshape(1, -1, x.shape[-1]), weight.t().unsqueeze(0))
    out = out.view(*x.shape[:-1], weight.shape[0])
    return out if bias is None else out + bias


def _h_conv2d(bmm, x, w, bias, stride, padding, dilation, groups):
    N, _, H, W = x.shape
    OC, Cg, kh, kw = w.shape
    G, K = groups, Cg * kh * kw
    Ho = (H + 2 * padding[0] - dilation[0] * (kh - 1) - 1) // stride[0] + 1
    Wo = (W + 2 * padding[1] - dilation[1] * (kw - 1) - 1) // stride[1] + 1
    cols = P.im2col(x, (kh, kw), dilation, padding, stride)
    L = cols.shape[-1]
    cols = cols.view(N, G, K, L).permute(1, 2, 0, 3).reshape(G, K, N * L)
    out = bmm(w.reshape(G, OC // G, K), cols)
    out = out.view(G, OC // G, N, L).permute(2, 0, 1, 3).reshape(N, OC, Ho, Wo)
    return out if bias is None else out + bias.view(1, -1, 1, 1)


def _h_conv1d(bmm, x, w, bias, stride, padding, dilation, groups):
    out = _h_conv2d(bmm, x.unsqueeze(2), w.unsqueeze(2), bias, (1, stride[0]), (0, padding[0]),
                    (1, dilation[0]), groups)
    return out.squeeze(2)


def _pair(v, n):
    v = list(v) if isinstance(v, (list, tuple)) else [v]
    return tuple(int(i) for i in (v * n if len(v) == 1 else v))


def _dsl_call(target, args, kwargs):
    """(kind, function(bmm), convolution signature or None) for a GEMM-shaped ATen call."""
    name = _name(target)
    a = bind(target, args, kwargs)
    if name == "aten.linear":
        return "linear", lambda bmm: _h_linear(bmm, a["input"], a["weight"], a["bias"]), None
    if name in ("aten.matmul", "aten.mm", "aten.bmm"):
        other = a.get("other") if name == "aten.matmul" else a.get("mat2")
        if a["self"].dim() < 2 or other.dim() < 2:
            return None
        return name[5:], lambda bmm: _h_matmul(bmm, a["self"], other), None
    if name == "aten.addmm":
        if a["beta"] != 1 or a["alpha"] != 1:
            return None
        return "addmm", lambda bmm: _h_addmm(bmm, a["self"], a["mat1"], a["mat2"]), None
    if name in ("aten.conv2d", "aten.conv1d", "aten.convolution"):
        x, w = a["input"], a["weight"]
        if isinstance(a["padding"], str) or a.get("transposed") or \
                any(a.get("output_padding") or ()):
            return None
        nd = w.dim() - 2
        if nd not in (1, 2) or x.dim() != nd + 2 or \
                (name != "aten.convolution" and name[-2] != str(nd)):
            return None
        conv = _h_conv2d if nd == 2 else _h_conv1d
        s, p, d = _pair(a["stride"], nd), _pair(a["padding"], nd), _pair(a["dilation"], nd)
        return f"conv{nd}d", lambda bmm: conv(bmm, x, w, a["bias"], s, p, d, int(a["groups"])), \
            ("conv", nd, tuple(x.shape), tuple(w.shape))
    return None


def _prod(shape):
    return int(np.prod(shape)) if len(shape) else 1


def _record_signature(c) -> tuple:
    """What one record of the kernel path's trace ran, in a form that does not depend on how
    the exported graph spells the call (torch.export versions decompose `linear` and `matmul`
    into `addmm`/`mm`/`bmm` on reshaped operands): the operand shapes of the kernel's bmm, as
    `compilers.GemmMode` builds them, or the input and weight shapes of a convolution."""
    op, sh = c["op"], [tuple(s) for s in c["shapes"]]
    if op in ("conv1d", "conv2d"):
        return ("conv", int(op[4]), sh[0], sh[1])
    if op == "linear":
        x, w = sh[0], sh[1]
        return ("gemm", (1, _prod(x[:-1]), x[-1]), (w[1], w[0]))
    a, b = (sh[1], sh[2]) if op == "addmm" else (sh[0], sh[1])
    if len(a) < 2 or len(b) < 2:
        return ("other", op, tuple(sh))
    if len(b) == 2:
        return ("gemm", (1, _prod(a[:-1]), a[-1]), b)
    batch = tuple(np.broadcast_shapes(a[:-2], b[:-2]))
    return ("gemm", (_prod(batch), a[-2], a[-1]), b[-2:])


def _canon(tree: SumTree, table: dict):
    """Key of the root that is equal for two trees exactly when they are the same tree up to
    the order of children (IEEE addition is commutative). `table` is shared by the two."""
    keys = list(range(tree.n)) + ["zero"] * tree.n_zero
    for ch in tree.children:
        keys.append(table.setdefault(frozenset(keys[c] for c in ch), ("node", len(table))))
    return keys[-1]

# --------------------------------------------------------------------------------------------
# PyTorch eager convolution (cuDNN, or the CPU kernels), from behaviour
# --------------------------------------------------------------------------------------------

_CONV_CACHE: Dict[tuple, object] = {}


def _conv_geometry(target, args, kwargs):
    """2-D view of a conv1d/conv2d call: (x4, w4, bias, stride, padding, dilation, groups, nd)."""
    a = bind(target, args, kwargs)
    x, w = a["input"], a["weight"]
    nd = w.dim() - 2
    s, p, d = _pair(a["stride"], nd), _pair(a["padding"], nd), _pair(a["dilation"], nd)
    if nd == 1:
        return (x.unsqueeze(2), w.unsqueeze(2), a["bias"], (1, s[0]), (0, p[0]), (1, d[0]),
                int(a["groups"]), nd)
    return x, w, a["bias"], s, p, d, int(a["groups"]), nd


def probe_conv(target, args, kwargs, trials: int = 5, seed: int = 0) -> dict:
    """Arithmetic of one PyTorch eager `conv1d` / `conv2d` call at exactly these shapes.

    Model: every output element is a dot product of the K = C/groups * kh * kw weights of its
    output channel with its im2col column (padding contributes products with +0), accumulated
    along one binary tree that is the same at every output position, then the bias is added
    with one more rounding. Established from the kernel only:

    1. FPRev masks in the weights (input = ones, so products are exact), one pair per output
       channel and call, read at the output positions whose window lies inside the input.
       All of them, in every batch element, must give the same answer.
    2. One probe per product tells whether it is rounded on its own or fused (FMA).
    3. Signed zeros: whether an accumulator starts at +0 (as in `probe_linear`).
    4. `primitives.tree_dot` over the explicit im2col must reproduce the kernel bit for bit at
       every output element (borders included), on adversarial inputs and weights.

    Algorithms that are not a fixed tree of the K products (Winograd, FFT), position-dependent
    orders and layers without an interior output position fail one of the steps and raise
    `LoweringFailed`. cuDNN autotuning (`cudnn.benchmark`) is refused: the algorithm would not
    be a function of the configuration."""
    import torch
    call = _dsl_call(target, args, kwargs)
    if call is None or call[2] is None:
        raise LoweringFailed("not a plain (non-transposed, numerically padded) convolution")
    x4, w4, bias, (sh, sw), (ph, pw), (dh, dw), G, nd = _conv_geometry(target, args, kwargs)
    a = bind(target, args, kwargs)
    x, w = a["input"], a["weight"]
    f32, device = torch.float32, x.device
    if x.dtype != f32 or w.dtype != f32 or x.numel() == 0 or not x.is_contiguous():
        raise LoweringFailed("only contiguous float32 operands were probed")
    if device.type == "cuda" and torch.backends.cudnn.enabled and torch.backends.cudnn.benchmark:
        raise LoweringFailed("cudnn.benchmark selects the algorithm by timing: no fixed kernel")
    OC, Cg, kh, kw = w4.shape
    Nb, _, H, W = x4.shape
    K = Cg * kh * kw
    Ho = (H + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    Wo = (W + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    names = [arg.name for arg in target._schema.arguments]
    rest = {n: a[n] for n in names if n not in ("input", "weight", "bias")}
    key = (str(target), tuple(x.shape), tuple(w.shape), bias is not None,
           tuple(sorted((k, str(v)) for k, v in rest.items())), str(device),
           torch.get_num_threads(), torch.backends.cudnn.enabled,
           torch.backends.cudnn.deterministic, trials, seed)
    what = (f"torch eager conv{nd}d x{list(x.shape)} w{list(w.shape)} stride {(sh, sw)[2 - nd:]} "
            f"padding {(ph, pw)[2 - nd:]} dilation {(dh, dw)[2 - nd:]} groups {G} "
            f"bias={bias is not None} on {device}"
            + (f" (cudnn.enabled={torch.backends.cudnn.enabled})" if device.type == "cuda"
               else ""))
    if key in _CONV_CACHE:
        got = _CONV_CACHE[key]
        if isinstance(got, str):
            raise LoweringFailed(got)
        return got
    # FPRev's rebuild and the tree walks recurse once per level; a k-sequential chain of K
    # products has K levels.
    limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(limit, 8 * K + 2000))
    try:
        return _probe_conv(target, args, kwargs, trials, seed, key, what)
    finally:
        sys.setrecursionlimit(limit)


def _probe_conv(target, args, kwargs, trials, seed, key, what):
    import torch
    x4, w4, bias, (sh, sw), (ph, pw), (dh, dw), G, nd = _conv_geometry(target, args, kwargs)
    a = bind(target, args, kwargs)
    x, w = a["input"], a["weight"]
    f32, device = torch.float32, x.device
    OC, Cg, kh, kw = w4.shape
    Nb, _, H, W = x4.shape
    K = Cg * kh * kw
    Ho = (H + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    Wo = (W + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    names = [arg.name for arg in target._schema.arguments]
    rest = {n: a[n] for n in names if n not in ("input", "weight", "bias")}

    def kernel(xs, ws, b=None):
        with torch.no_grad():
            return target(xs, ws, b, **rest)

    def out4(y):                                           # kernel result as [N, OC, Ho, Wo]
        return y.unsqueeze(2) if nd == 1 else y

    rows = [i for i in range(Ho) if i * sh - ph >= 0 and i * sh - ph + dh * (kh - 1) <= H - 1]
    cols = [j for j in range(Wo) if j * sw - pw >= 0 and j * sw - pw + dw * (kw - 1) <= W - 1]
    try:
        if not rows or not cols:
            raise LoweringFailed(f"{what}: no output position has its window inside the input; "
                                 "the order of the products cannot be probed")
        r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
        ones = torch.ones(x.shape, dtype=f32, device=device)
        zero_b = torch.zeros(OC, dtype=f32, device=device) if bias is not None else None

        def interior(xs, ws):
            """[OC] values at the interior positions; they must all agree."""
            y = out4(kernel(xs, ws.reshape(w.shape).to(device), zero_b))[:, :, r0:r1, c0:c1]
            y = y.transpose(0, 1).reshape(OC, -1).double().cpu()
            if bool((y != y[:, :1]).any()):
                raise RevealError("output positions disagree: the order depends on the position "
                                  "or the kernel is not a tree of the products")
            return y[:, 0]

        if K == 1:
            tree = SumTree(1, [])
        else:
            Mv = mask_value(FORMATS["float32"], K)

            def lca_sizes(pairs):
                out = []
                for s in range(0, len(pairs), OC):
                    chunk = pairs[s:s + OC]
                    ws = torch.ones((OC, K), dtype=f32)
                    for c, (i, j) in enumerate(chunk):
                        ws[c, i], ws[c, j] = Mv, -Mv
                    y = interior(ones, ws)
                    for c in range(len(chunk)):
                        l = K - float(y[c])
                        if not l.is_integer():
                            raise RevealError(f"kernel returned {float(y[c])} for a masked pair: "
                                              "not a fixed summation tree of the products")
                        out.append(int(l))
                return out

            try:
                tree = reveal_order_batched(lca_sizes, K)
            except RecursionError as e:
                raise RevealError("tree too deep to rebuild") from e
            if tree.multiterm:
                raise RevealError("multi-term fused nodes: no IEEE model")
        # Which products are fused (see fprev.probe_eager_linear for the values).
        fused = [False] * K
        if K > 1:
            av = float(np.float32(1 + 2.0 ** -12))
            partner = float(np.float32(-(1 - 2.0 ** -12)))
            under, sibling = {}, {}
            for k, ch in enumerate(tree.children):
                node = tree.n_total + k
                under[node] = [v for c in ch for v in (under[c] if c >= tree.n_total else [c])
                               if v < K]
                for c in ch:
                    if c < K:
                        o = [d for d in ch if d != c][0]
                        sibling[c] = o if o < K else (under[o][0] if o >= tree.n_total and
                                                      under[o] else None)
            xa = torch.full(x.shape, av, dtype=f32, device=device)
            for s in range(0, K, OC):
                idx = list(range(s, min(K, s + OC)))
                ws = torch.zeros((OC, K), dtype=f32)
                for c, i in enumerate(idx):
                    if sibling.get(i) is None:
                        raise RevealError(f"product {i} has no product sibling to probe against")
                    ws[c, i], ws[c, sibling[i]] = av, partner
                y = interior(xa, ws)
                for c, i in enumerate(idx):
                    if float(y[c]) == 2.0 ** -11 + 2.0 ** -23:
                        fused[i] = True
                    elif float(y[c]) != 2.0 ** -11 + 2.0 ** -24:
                        raise RevealError(f"product probe of leaf {i} returned {float(y[c])!r}: "
                                          "neither a rounded nor a fused product")
        check_fused(tree, fused)
        # Signed zeros, as in probe_linear: every product underflows to -0.
        zero_init, zero_root = [False] * K, False
        xt = torch.full(x.shape, 2.0 ** -80, dtype=f32, device=device)
        nb = torch.full((OC,), -0.0, dtype=f32, device=device) if bias is not None else None

        def zeros_at(ws):
            y = out4(kernel(xt, ws.reshape(w.shape).to(device), nb))[:, :, r0:r1, c0:c1]
            y = y.transpose(0, 1).reshape(OC, -1).cpu()
            sg = torch.signbit(y)
            if bool((y != 0).any()) or bool((sg != sg[:, :1]).any()):
                raise RevealError("underflowing products do not give a uniform zero")
            return sg[:, 0]

        sg = zeros_at(torch.full((OC, K), -(2.0 ** -80), dtype=f32))
        if bool(sg.any()) != bool(sg.all()):
            raise RevealError("output channels disagree on the sign of a zero sum")
        zero_root = not bool(sg.all())
        for s in ([] if zero_root else range(0, K, OC)):
            idx = list(range(s, min(K, s + OC)))
            ws = torch.full((OC, K), -(2.0 ** -80), dtype=f32)
            for c, i in enumerate(idx):
                ws[c, i] = -0.0
            sg = zeros_at(ws)
            for c, i in enumerate(idx):
                zero_init[i] = not bool(sg[c])
                if zero_init[i] and fused[i]:
                    raise RevealError(f"product {i} looks fused and zero-initialized")

        def bmm(aa, bb):
            return P.tree_dot(tree, aa, bb, fused, zero_init, zero_root=zero_root)

        def lowered(xs, ws, b=None):
            run = _dsl_call(target, (xs, ws, b), rest)[1]
            return run(bmm)

        gen = torch.Generator().manual_seed(seed)

        def make(t):
            kx = KINDS[t % len(KINDS)]
            kw_ = "normal" if kx in ("spread", "cancel") else kx
            xs = adversarial_tensor(tuple(x.shape), kx, gen, f32, device)
            ws = adversarial_tensor(tuple(w.shape), kw_, gen, f32, device)
            if bias is None:
                return xs, ws
            return xs, ws, adversarial_tensor((OC,), "zeros" if kx == "zeros" else "normal",
                                              gen, f32, device)

        validate_bitwise(kernel, lowered, make, trials, what)
    except (RevealError, LoweringFailed) as e:
        msg = str(e) if str(e).startswith(what) else f"{what}: {e}"
        _CONV_CACHE[key] = msg
        raise LoweringFailed(msg) from e
    res = {"bmm": bmm, "kernel": what, "K": K, "depth": tree_depth(tree),
           "tree": str(tree.to_nested())[:400], "fused_products": sum(fused),
           "zero_init_products": sum(zero_init), "zero_root": zero_root,
           "masked_pairs": getattr(tree, "queries", 0),
           "interior_positions": len(rows) * len(cols), "output_positions": Ho * Wo,
           "validation_inputs": trials, "threads": torch.get_num_threads()}
    _CONV_CACHE[key] = res
    return res


class ProbedConvLowering(Lowering):
    """PyTorch eager `conv1d` / `conv2d` (also spelled `aten.convolution`) on float32, as an
    explicit im2col followed by one tree of the K products per output element."""

    name = "probed-conv"

    def __init__(self, trials: int = 5, seed: int = 0):
        self.trials, self.seed, self.failures = trials, seed, {}

    def lower(self, node_name, target, args, kwargs):
        if _name(target) not in ("aten.conv2d", "aten.conv1d", "aten.convolution"):
            return None
        try:
            ev = probe_conv(target, args, kwargs, self.trials, self.seed)
        except LoweringFailed as e:
            self.failures[node_name] = str(e)
            return None
        bmm = ev["bmm"]
        a = bind(target, args, kwargs)
        shapes = (tuple(a["input"].shape), tuple(a["weight"].shape))

        def fn(*args, **kwargs):
            b = bind(target, args, kwargs)
            if (tuple(b["input"].shape), tuple(b["weight"].shape)) != shapes:
                raise LoweringFailed("operands differ from the probed shapes")
            return _dsl_call(target, args, kwargs)[1](bmm)

        return LoweredOp(
            fn, ev["kernel"],
            f"im2col by zero padding and strided slices; per output element a tree of depth "
            f"{ev['depth']} over {ev['K']} products: {ev['fused_products']} enter their node "
            f"unrounded (FMA), {ev['zero_init_products']} of the rounded ones on a +0 accumulator"
            + ("; +0 added to the root" if ev["zero_root"] else "")
            + "; bias (if any) added last with one rounding", "validated",
            {k: v for k, v in ev.items() if k not in ("bmm", "kernel")},
            [f"order[{ev['kernel']}] inferred by FPRev, product and signed-zero probes at the "
             f"{ev['interior_positions']} interior output positions and validated at all "
             f"{ev['output_positions']} on {ev['validation_inputs']} inputs: one tree of the "
             "products for every output element and channel, padding as products with +0, "
             "fixed for these shapes, device and library build and independent of the values "
             "and of the weights (dynamic evidence only)."])


STATIC_EVIDENCE = {
    "triton": "kernels.triton_bmm_kernel: `acc = tl.zeros(...)`, then `acc = tl.dot(a, b, acc, "
              "input_precision=\"ieee\")` for k blocks in increasing order with loads masked "
              "to 0.0 beyond k_hi; compilers.triton_bmm: split_k partial results, `c.sum(1)`.",
    "tilelang": "kernels.tilelang_bmm_kernel: `T.clear(Cl)`, then `Cl[i, j] += As[i, k] * "
                "Bs[k, j]` for `ko` (T.Pipelined) and `k` (T.serial) increasing; "
                "compilers.tilelang_bmm: zero padding to split_k chunks, `.sum(1)`.",
}


class DslGemmLowering(Lowering):
    """The GEMM-shaped operations a `triton` / `tilelang` compiled program ran on its kernel.

    Which calls ran on the kernel comes from the path's own call trace on the example inputs
    (`Compiled.meta["kernel_trace"]`), matched to the ATen nodes in program order; a call that
    fell back to PyTorch is left alone. For every kernel call the operands of the underlying
    `bmm` are rebuilt as the path builds them and the kernel is replaced by `dsl_bmm`.

    With `probe=True` (default) the kernel itself is consulted on the device
    (`Compiled.meta["bmm"]`): FPRev masks must reveal the tree `dsl_bmm` computes (the
    sequential chain over k), the product probe must report FMA, and `dsl_bmm` must reproduce
    the kernel bit for bit on `trials` adversarial inputs at every distinct bmm shape of the
    program. With `probe=False` the order rests on the kernel source
    alone (level "source"), except the split-K combine, which is a library `sum` and is always
    probed.
    """

    name = "dsl-gemm"

    def __init__(self, compiled, example_inputs, probe: bool = True, trials: int = 5,
                 seed: int = 0, kernel: Optional[Callable] = None):
        import torch
        self.backend = compiled.compiler
        if self.backend not in ("triton", "tilelang"):
            raise ValueError(f"not a DSL kernel path: {self.backend}")
        self.options = dict(compiled.options)
        self.split_k = int(self.options.get("split_k", 1) or 1)
        self.block_k = int(self.options["block_k"])
        self.probe, self.trials, self.seed = probe, trials, seed
        self.failures = {}
        trace = (compiled.meta or {}).get("kernel_trace")
        if trace is None:
            raise LoweringFailed(f"compiled {self.backend} program has no kernel_trace: cannot "
                                 "tell which operations ran on the kernel")
        with torch.no_grad():
            _, self.calls = trace(*example_inputs)
        self.pos = 0
        self.stats = {"calls": len(self.calls), "kernel": 0, "fallback": 0}
        if kernel is None:
            kernel = (compiled.meta or {}).get("bmm")
        if kernel is None:                                 # older Compiled objects
            mod = sys.modules[type(compiled).__module__]
            bmm = getattr(mod, f"{self.backend}_bmm")
            opts = self.options
            kernel = lambda a, b: bmm(a, b, **opts)        # noqa: E731
        self.kernel = kernel
        self._validated, self._chains = {}, {}

    # ---- evidence ----

    def _expected_tree(self, K: int, combine: Optional[dict]) -> SumTree:
        """The tree over the K products that `dsl_bmm` computes: one chain per k range, the
        chains joined as the split-K combine joins them (empty ranges contribute +0)."""
        children, roots = [], []
        for lo, hi, _ in k_ranges(self.backend, K, self.split_k, self.block_k):
            prev = lo if hi > lo else None
            for k in range(lo + 1, hi):
                children.append((prev, k))
                prev = K + len(children) - 1
            roots.append(prev)
        if self.split_k > 1:
            ct = combine["tree"]
            vals = list(roots)
            for lhs, rhs in ct.children:
                x, y = vals[lhs], vals[rhs]
                if x is None or y is None:
                    vals.append(y if x is None else x)
                else:
                    children.append((x, y))
                    vals.append(K + len(children) - 1)
        return SumTree(K, children)

    def _chain_evidence(self, K: int, device, combine: Optional[dict]) -> dict:
        """FPRev on the kernel as configured: the revealed tree over k must be the tree of
        `dsl_bmm` (the sequential chain; with split-K, chains joined by the combine). Without
        split-K a two-term probe must also show fused products."""
        import torch
        key = (K, str(device))                 # per lowering: one kernel, one configuration
        if key in self._chains:
            return self._chains[key]
        bmm = self.kernel
        f32, cols = torch.float32, 64
        ev = {"K": K}
        if K > 1:
            Mv = mask_value(FORMATS["float32"], K)

            def lca_sizes(pairs):
                out = []
                for s in range(0, len(pairs), cols):
                    chunk = pairs[s:s + cols]
                    b = torch.ones((1, K, len(chunk)), dtype=f32)
                    for c, (i, j) in enumerate(chunk):
                        b[0, i, c], b[0, j, c] = Mv, -Mv
                    a = torch.ones((1, 2, K), dtype=f32, device=device)
                    with torch.no_grad():
                        y = bmm(a, b.to(device))[0].double().cpu()
                    if not torch.equal(y[0], y[1]):
                        raise RevealError("rows of one kernel call disagree on a masked pair")
                    for c in range(len(chunk)):
                        l = K - float(y[0, c])
                        if not l.is_integer():
                            raise RevealError(f"kernel returned {float(y[0, c])} for a masked "
                                              "pair")
                        out.append(int(l))
                return out

            try:
                tree = reveal_order_batched(lca_sizes, K)
            except RevealError as e:
                raise LoweringFailed(f"{self.backend} kernel, K = {K}: {e}") from e
            table = {}
            if tree.multiterm or _canon(tree, table) != _canon(self._expected_tree(K, combine),
                                                               table):
                raise LoweringFailed(
                    f"{self.backend} kernel, K = {K}: FPRev reveals a tree of depth "
                    f"{tree_depth(tree)} that is not the k-sequential accumulation the kernel "
                    "source describes")
            ev["fprev"] = (f"k-sequential chain{'s per split' if self.split_k > 1 else ''} "
                           f"({tree.queries} masked pairs)")
            if self.split_k == 1:
                with torch.no_grad():
                    fma = detect_fma(lambda A, B: bmm(A[None], B[None])[0], device=str(device))
                if fma != "fma":
                    raise LoweringFailed(f"{self.backend} kernel: product probe reports "
                                         f"{fma!r}, not a fused multiply-add")
                ev["fma_probe"] = fma
        self._chains[key] = ev
        return ev

    def _bmm_for(self, shapes, device) -> Callable:
        """The explicit bmm for the operand shapes of one call, after establishing evidence."""
        import torch
        (Bt, M, K), (Bb, _, N) = shapes
        combine = None
        if self.split_k > 1:
            combine = probe_sum((Bt, self.split_k, M, N), 1, torch.float32, device,
                                self.trials, self.seed)
            if combine["tree"] is None:
                raise LoweringFailed(f"{combine['kernel']}: the split-K combine uses "
                                     f"{combine['position_classes']} position-dependent orders")
        backend, S, bk = self.backend, self.split_k, self.block_k
        lowered = lambda a, b: dsl_bmm(a, b, backend, S, bk, combine)      # noqa: E731
        key = (tuple(shapes[0]), tuple(shapes[1]), str(device))
        if key not in self._validated:
            ev = {"static": STATIC_EVIDENCE[backend]}
            if self.probe:
                ev.update(self._chain_evidence(K, device, combine))
                gen = torch.Generator().manual_seed(self.seed)

                def make(t):
                    ka = KINDS[t % len(KINDS)]
                    kb = "normal" if ka in ("spread", "cancel") else ka
                    return (adversarial_tensor((Bt, M, K), ka, gen, torch.float32, device),
                            adversarial_tensor((Bb, K, N), kb, gen, torch.float32, device, 1))

                ev["validation_inputs"] = validate_bitwise(
                    self.kernel, lowered, make, self.trials,
                    f"{backend} bmm [{Bt}, {M}, {K}] @ [{Bb}, {K}, {N}] {self.options}")
            if combine is not None:
                ev["split_k_combine"] = {k: v for k, v in combine.items()
                                         if k not in ("tree", "apply")}
                ev["split_k_combine"]["tree"] = str(combine["tree"].to_nested())
            self._validated[key] = ev
        return lowered, self._validated[key]

    # ---- Lowering interface ----

    def lower(self, node_name, target, args, kwargs):
        import torch
        call = _dsl_call(target, args, kwargs)
        if call is None:
            return None
        handler, run, sig = call
        seen = []

        def stub(a, b):                                    # operand shapes of the bmm
            seen.append((tuple(a.shape), tuple(b.shape)))
            Bt = max(a.shape[0], b.shape[0])
            return torch.zeros((Bt, a.shape[1], b.shape[2]), dtype=a.dtype, device=a.device)

        with torch.no_grad():
            run(stub)
        if sig is None:
            sig = ("gemm", seen[0][0], seen[0][1][1:])
        rec = None
        for j in range(self.pos, len(self.calls)):
            if _record_signature(self.calls[j]) == sig:
                rec, self.pos = self.calls[j], j + 1
                break
        if rec is None:
            raise LoweringFailed(
                f"{target} ({node_name}, {sig}) has no matching call in the "
                f"{self.backend} path's trace: the graph and the kernel path disagree on what ran")
        if not rec["handled"]:
            self.stats["fallback"] += 1
            return None
        self.stats["kernel"] += 1
        operands = [t for t in list(args) + list(kwargs.values()) if isinstance(t, torch.Tensor)]
        device = operands[0].device
        try:
            lowered, ev = self._bmm_for(seen[0], device)
        except LoweringFailed as e:
            raise LoweringFailed(f"{node_name}: {e}") from e

        def fn(*args, **kwargs):
            _, run, _ = _dsl_call(target, args, kwargs)
            return run(lowered)

        handler = rec["op"]
        (Bt, M, K), (_, _, N) = seen[0]
        ranges = k_ranges(self.backend, K, self.split_k, self.block_k)
        desc = (f"{handler} -> bmm [{Bt}, {M}, {K}] @ [.., {K}, {N}]: per output element "
                f"acc = +0; acc = fma(a[m,k], b[k,n], acc) for k increasing over "
                f"{[(lo, hi) for lo, hi, _ in ranges]}"
                + ("; one fma(+0, +0, acc) for the masked tail" if any(p for *_, p in ranges)
                   else "")
                + (f"; partial sums added as {ev['split_k_combine']['tree']}"
                   if self.split_k > 1 else "")
                + ("; im2col by zero padding and strided slices" if handler.startswith("conv")
                   else "")
                + "; bias (if any) added afterwards with one rounding")
        assumptions = [
            f"The {self.backend} compiler lowers the kernel in kernels.py to one float32 "
            "accumulator per output element updated by a fused multiply-add in increasing k, "
            "for every tile and every output position"
            + (" (kernel source; FPRev chain, FMA probe and bitwise validation on this device "
               "agree)." if self.probe else " (kernel source only; not probed)."),
            "The bias addition after the kernel is one correctly rounded float32 addition."]
        if self.split_k > 1:
            assumptions.append(
                f"order[{ev['split_k_combine']['kernel']}] of the split-K combine inferred by "
                "FPRev and validated: fixed for this shape and device, independent of the "
                "values (dynamic evidence only).")
        return LoweredOp(fn, f"{self.backend} GEMM kernel {self.options}", desc,
                         "known" if self.probe else "source", dict(ev), assumptions)

    def finish(self):
        rest = [c for c in self.calls[self.pos:] if c["handled"]]
        if rest:
            raise LoweringFailed(
                f"{len(rest)} kernel call(s) of the {self.backend} path have no node in the "
                f"exported graph (first: {rest[0]['op']} {rest[0]['shapes']})")
