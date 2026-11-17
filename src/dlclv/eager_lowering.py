"""Lowerings of PyTorch eager library kernels that are not dot products: clamps (ReLU,
hardtanh), max pooling and inference batch normalization.

None of these has an accumulation order. What is not fixed by the operator's definition, and
therefore has to come from the kernel, is:

* clamp / ReLU:  the sign of a zero result. `relu(-0.0)` is -0 on the CPU and +0 on CUDA
                 (PyTorch 2.6), because one kernel returns the operand on a tie and the other
                 the bound.
* max pooling:   which of two equal maxima is returned; it matters only for +0 against -0.
                 The kernels scan the window in row-major order from -inf and replace the
                 running maximum on a strict `>` (aten/src/ATen/native/cuda/DilatedMaxPool2d.cu).
* batch norm:    the association of `w * (x - mean) * invstd + b`, whether it is first folded to
                 `x * alpha + beta`, and which multiply-add is fused. PyTorch 2.6 computes
                 `fma(fl(fl(x - mean) * w), invstd, b)` on CUDA (native and cuDNN) and
                 `fma(x, alpha, fma(-mean, alpha, b))` with `alpha = fl(invstd * w)` on x86 CPUs.

Each lowering states a small set of explicit candidate programs (`dlclv.primitives` only), runs
the kernel on adversarial inputs, discards every candidate that differs in one bit, and accepts
the call only if a candidate survives. Candidates that survive together were bitwise identical
on every input tried; the first is used and the others are listed in the evidence. Evidence
level: "validated" (dynamic; an assumption for all inputs), except max pooling on CUDA, whose
scan order is also read from the kernel source ("known").

`invstd` of batch normalization is `rsqrt(var + eps)`, a library function that is not
correctly rounded on every device and has no explicit primitive. It depends on the frozen
running variance only, not on the program input, so it is read from the kernel once (the call
with mean 0, weight 1, bias 0 on an input of ones returns it exactly in every candidate form)
and kept as a constant of the reconstruction. The lowered call checks on every execution that
the statistics and affine parameters are the tensors it was built for.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

from . import primitives as P
from .bitwise import bitwise_equal
from .lowering import (KINDS, LoweredOp, Lowering, LoweringFailed, adversarial_tensor, bind,
                       _pair)

Candidate = Tuple[str, Callable]


def select_candidates(kernel: Callable, candidates: Sequence[Candidate], inputs: Sequence,
                      what: str) -> List[Candidate]:
    """Candidates that reproduce `kernel` bit for bit on every input, in the given order.
    Raises `LoweringFailed` if the kernel returns a NaN (the inputs are built to stay finite
    or infinite, never NaN) or if no candidate survives."""
    import torch
    alive = list(candidates)
    closest = ""
    for t, x in enumerate(inputs):
        with torch.no_grad():
            want = kernel(x)
            if bool(torch.isnan(want).any()):
                raise LoweringFailed(f"{what}: the kernel returned NaN on validation input {t}")
            keep = []
            for name, fn in alive:
                with P.PrimitiveGuard():
                    got = fn(x)
                if bitwise_equal(got, want):
                    keep.append((name, fn))
                elif len(alive) == 1:
                    bad = int(((got != want) | (torch.signbit(got) != torch.signbit(want))).sum())
                    closest = (f"; last candidate `{name}` differs in {bad} of {want.numel()} "
                               f"elements on input {t}")
        alive = keep
        if not alive:
            raise LoweringFailed(f"{what}: none of {len(candidates)} explicit candidates "
                                 f"reproduces the kernel{closest}")
    return alive


def _float_inputs(shape, dtype, device, trials: int, seed: int, extra=()):
    import torch
    gen = torch.Generator().manual_seed(seed)
    xs = [adversarial_tensor(shape, KINDS[t % len(KINDS)], gen, dtype, device)
          for t in range(trials)]
    xs += [torch.full(shape, v, dtype=dtype, device=device) for v in (-0.0, 0.0)]
    return xs + list(extra)


# --------------------------------------------------------------------------------------------
# clamp, ReLU, hardtanh
# --------------------------------------------------------------------------------------------

def clamp_candidates(lo: Optional[float], hi: Optional[float]) -> List[Candidate]:
    """All tie conventions of `min(max(x, lo), hi)` (lo <= hi). On a tie x == bound the kernel
    returns either its operand or the bound. The two are different programs only when the
    bound is a zero (a nonzero value has one bit pattern), so one convention is stated for a
    nonzero bound."""
    def lower(tie_bound):
        if lo is None:
            return None, lambda x: x
        if tie_bound:
            return f"where(x > {lo!r}, x, {lo!r})", \
                lambda x: _where(x > lo, x, P_full(x, lo))
        return f"where(x < {lo!r}, {lo!r}, x)", \
            lambda x: _where(x < lo, P_full(x, lo), x)

    def upper(tie_bound):
        if hi is None:
            return None, lambda y: y
        if tie_bound:
            return f"where(y < {hi!r}, y, {hi!r})", \
                lambda y: _where(y < hi, y, P_full(y, hi))
        return f"where(y > {hi!r}, {hi!r}, y)", \
            lambda y: _where(y > hi, P_full(y, hi), y)

    out = []
    for tl in ((True, False) if lo == 0.0 else (True,)):
        for th in ((True, False) if hi == 0.0 else (True,)):
            (dl, fl), (du, fu) = lower(tl), upper(th)
            desc = "; ".join(d for d in (dl and "y = " + dl, du and "out = " + du) if d)
            out.append((desc, lambda x, fl=fl, fu=fu: fu(fl(x))))
    return out


def _where(c, a, b):
    import torch
    return torch.where(c, a, b)


def P_full(x, value):
    import torch
    return torch.full_like(x, value)


class ClampLowering(Lowering):
    """`relu`, `hardtanh` (also `relu6`) and `clamp` with scalar bounds, on float32/float64."""

    name = "clamp"
    TARGETS = ("aten.relu.default", "aten.relu6.default", "aten.hardtanh.default",
               "aten.clamp.default")

    def __init__(self, trials: int = 5, seed: int = 0):
        self.trials, self.seed, self.failures = trials, seed, {}
        self._cache: Dict[tuple, tuple] = {}

    @staticmethod
    def _bounds(target, a):
        name = str(target)
        if name == "aten.relu.default":
            return 0.0, None
        if name == "aten.relu6.default":
            return 0.0, 6.0
        if name == "aten.hardtanh.default":
            return float(a["min_val"]), float(a["max_val"])
        lo, hi = a["min"], a["max"]
        return (None if lo is None else float(lo)), (None if hi is None else float(hi))

    def lower(self, node_name, target, args, kwargs):
        import torch
        if str(target) not in self.TARGETS:
            return None
        a = bind(target, args, kwargs)
        x = a["self"]
        try:
            lo, hi = self._bounds(target, a)
            if x.dtype not in (torch.float32, torch.float64) or x.numel() == 0:
                raise LoweringFailed("only non-empty float32/float64 operands")
            if (lo is None and hi is None) or (lo is not None and hi is not None and lo > hi) \
                    or any(v is not None and v != v for v in (lo, hi)):
                raise LoweringFailed(f"unsupported bounds [{lo}, {hi}]")
            shape, dtype, device = tuple(x.shape), x.dtype, x.device
            key = (str(target), lo, hi, shape, str(dtype), str(device),
                   torch.get_num_threads())
            if key not in self._cache:
                what = f"{target}[{lo}, {hi}] on {str(dtype)[6:]}{list(shape)} on {device}"

                def kernel(t):
                    return target(t, *args[1:], **kwargs)

                extra = [torch.full(shape, v, dtype=dtype, device=device)
                         for v in (lo, hi) if v is not None]
                # values on both sides of each bound, one ulp apart
                for v in (lo, hi):
                    if v is not None:
                        base = torch.full(shape, v, dtype=dtype, device=device)
                        extra.append(torch.nextafter(base, base + 1))
                        extra.append(torch.nextafter(base, base - 1))
                alive = select_candidates(kernel, clamp_candidates(lo, hi),
                                          _float_inputs(shape, dtype, device, self.trials,
                                                        self.seed, extra), what)
                self._cache[key] = (what, alive, self.trials + 2 + len(extra))
            what, alive, n_inputs = self._cache[key]
        except LoweringFailed as e:
            self.failures[node_name] = str(e)
            return None
        desc, cand = alive[0]

        def fn(*args, **kwargs):
            x = bind(target, args, kwargs)["self"]
            if tuple(x.shape) != shape or x.dtype != dtype:
                raise LoweringFailed("operand differs from the validated shape/dtype")
            return cand(x)

        zero_bound = any(v == 0.0 for v in (lo, hi) if v is not None)
        return LoweredOp(
            fn, what, desc + " (a selection: no rounding)", "validated",
            {"validation_inputs": n_inputs, "candidates": len(clamp_candidates(lo, hi)),
             "indistinguishable": [d for d, _ in alive[1:]],
             "zero_bound": zero_bound},
            [f"tie convention[{what}] (the sign of a zero result) selected among explicit "
             f"candidates by bitwise validation on {n_inputs} inputs, including all -0, all +0 "
             "and one ulp around each bound: the same for every element and independent of the "
             "other values (dynamic evidence only)."])


# --------------------------------------------------------------------------------------------
# max pooling
# --------------------------------------------------------------------------------------------

def max_pool2d_explicit(x, kernel, stride, padding, dilation):
    """Row-major scan of each window from -inf with a strict `>`: the first maximum wins.
    Padding is -inf, as in the kernel (a padded position never replaces a value)."""
    import torch
    (kh, kw), (sh, sw), (ph, pw), (dh, dw) = kernel, stride, padding, dilation
    H, W = x.shape[-2], x.shape[-1]
    Ho = (H + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    Wo = (W + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    xp = torch.constant_pad_nd(x, (pw, pw, ph, ph), float("-inf")) if (ph or pw) else x
    acc = None
    for i in range(kh):
        for j in range(kw):
            patch = xp[..., i * dh: i * dh + sh * (Ho - 1) + 1: sh,
                       j * dw: j * dw + sw * (Wo - 1) + 1: sw]
            acc = patch if acc is None else torch.where(patch > acc, patch, acc)
    return acc


class MaxPoolLowering(Lowering):
    """`max_pool2d` without `ceil_mode` on float32/float64 (3-D or 4-D input). Older exports
    spell it `max_pool2d_with_indices`; its first result is lowered and the indices are not
    produced (a program that uses them is rejected by the frontend)."""

    name = "max-pool"
    TARGETS = ("aten.max_pool2d.default", "aten.max_pool2d_with_indices.default")

    def __init__(self, trials: int = 5, seed: int = 0):
        self.trials, self.seed, self.failures = trials, seed, {}
        self._cache: Dict[tuple, tuple] = {}

    def lower(self, node_name, target, args, kwargs):
        import torch
        if str(target) not in self.TARGETS:
            return None
        a = bind(target, args, kwargs)
        x = a["self"]
        pair = "with_indices" in str(target)

        def kernel(t):
            out = target(t, *args[1:], **kwargs)
            return out[0] if pair else out

        try:
            if a["ceil_mode"] or x.dim() not in (3, 4) or x.numel() == 0 or \
                    x.dtype not in (torch.float32, torch.float64):
                raise LoweringFailed("only float32/float64 max_pool2d without ceil_mode")
            k = _pair(a["kernel_size"], 2)
            s = _pair(a["stride"], 2) if a["stride"] else k
            p, d = _pair(a["padding"], 2), _pair(a["dilation"], 2)
            shape, dtype, device = tuple(x.shape), x.dtype, x.device
            key = (shape, k, s, p, d, str(dtype), str(device), torch.get_num_threads())
            what = (f"max_pool2d kernel {k} stride {s} padding {p} dilation {d} on "
                    f"{str(dtype)[6:]}{list(shape)} on {device}")
            if key not in self._cache:
                # A reversed scan and a non-strict comparison are stated too, so that the
                # validation is known to be able to tell them apart (they must not survive
                # unless the window has one element).
                def last_wins(t):
                    xp = torch.constant_pad_nd(t, (p[1], p[1], p[0], p[0]), float("-inf"))
                    Ho = (t.shape[-2] + 2 * p[0] - d[0] * (k[0] - 1) - 1) // s[0] + 1
                    Wo = (t.shape[-1] + 2 * p[1] - d[1] * (k[1] - 1) - 1) // s[1] + 1
                    acc = None
                    for i in range(k[0]):
                        for j in range(k[1]):
                            patch = xp[..., i * d[0]: i * d[0] + s[0] * (Ho - 1) + 1: s[0],
                                       j * d[1]: j * d[1] + s[1] * (Wo - 1) + 1: s[1]]
                            acc = patch if acc is None else torch.where(patch < acc, acc, patch)
                    return acc

                cands = [("first maximum in row-major window order",
                          lambda t: max_pool2d_explicit(t, k, s, p, d)),
                         ("last maximum in row-major window order", last_wins)]
                alive = select_candidates(
                    kernel, cands,
                    _float_inputs(shape, dtype, device, self.trials, self.seed), what)
                if alive[0][0] != cands[0][0]:
                    raise LoweringFailed(f"{what}: the kernel does not return the first maximum")
                self._cache[key] = (len(alive) > 1, self.trials + 2)
            ambiguous, n_inputs = self._cache[key]
        except LoweringFailed as e:
            self.failures[node_name] = str(e)
            return None

        def fn(*args, **kwargs):
            x = bind(target, args, kwargs)["self"]
            if tuple(x.shape) != shape or x.dtype != dtype:
                raise LoweringFailed("operand differs from the validated shape/dtype")
            out = max_pool2d_explicit(x, k, s, p, d)
            return (out, None) if pair else out

        cuda = device.type == "cuda"
        ev = {"validation_inputs": n_inputs, "tie_order_observable": not ambiguous}
        if cuda:
            ev["static"] = ("aten/src/ATen/native/cuda/DilatedMaxPool2d.cu (v2.6.0): `maxval = "
                            "-inf; for h, w in the window: if (val > maxval || isnan(val)) "
                            "maxval = val`")
        return LoweredOp(
            fn, what,
            f"-inf padding; acc = first window element; acc = where(v > acc, v, acc) over the "
            f"{k[0] * k[1]} window positions in row-major order (a selection: no rounding)",
            "known" if cuda else "validated", ev,
            [f"scan order[{what}]: the first maximum in row-major window order is returned at "
             "every output position (it decides only +0 against -0)"
             + (", read from the kernel source and" if cuda else ",")
             + f" validated bitwise on {n_inputs} inputs."])


# --------------------------------------------------------------------------------------------
# inference batch normalization
# --------------------------------------------------------------------------------------------

def batch_norm_candidates() -> List[Candidate]:
    """Explicit float32 programs for w * (x - m) * s + b; s is the kernel's invstd constant.
    Each takes (x, m, w, b, s) with the per-channel tensors already shaped to broadcast."""
    f = P.fma32
    return [
        ("fma(fl(fl(x - mean) * w), invstd, b)", lambda x, m, w, b, s: f((x - m) * w, s, b)),
        ("fma(x, alpha, fma(-mean, alpha, b)), alpha = fl(invstd * w)",
         lambda x, m, w, b, s: f(x, s * w, f(-m, s * w, b))),
        ("fma(x, alpha, fl(b - fl(mean * alpha))), alpha = fl(invstd * w)",
         lambda x, m, w, b, s: f(x, s * w, b - m * (s * w))),
        ("fl(fl(x * alpha) + fl(b - fl(mean * alpha))), alpha = fl(invstd * w)",
         lambda x, m, w, b, s: x * (s * w) + (b - m * (s * w))),
        ("fl(fl(x * alpha) + fma(-mean, alpha, b)), alpha = fl(invstd * w)",
         lambda x, m, w, b, s: x * (s * w) + f(-m, s * w, b)),
        ("fl(fl(fl(fl(x - mean) * w) * invstd) + b)", lambda x, m, w, b, s: (x - m) * w * s + b),
        ("fma(fl(fl(x - mean) * invstd), w, b)", lambda x, m, w, b, s: f((x - m) * s, w, b)),
        ("fl(fl(fl(fl(x - mean) * invstd) * w) + b)", lambda x, m, w, b, s: (x - m) * s * w + b),
        ("fma(fl(x - mean), fl(invstd * w), b)", lambda x, m, w, b, s: f(x - m, s * w, b)),
        ("fl(fl(fl(x - mean) * fl(invstd * w)) + b)",
         lambda x, m, w, b, s: (x - m) * (s * w) + b),
    ]


class BatchNormLowering(Lowering):
    """`aten.batch_norm` in inference mode with affine parameters, float32. Older exports
    spell it `_native_batch_norm_legit_no_training`, a tuple whose first element is lowered."""

    name = "batch-norm"
    TARGETS = ("aten.batch_norm.default", "aten._native_batch_norm_legit_no_training.default")

    def __init__(self, trials: int = 5, seed: int = 0):
        self.trials, self.seed, self.failures = trials, seed, {}

    def lower(self, node_name, target, args, kwargs):
        import torch
        if str(target) not in self.TARGETS:
            return None
        a = bind(target, args, kwargs)
        a.setdefault("training", False)
        triple = "_native" in str(target)
        x = a["input"]
        params = [a["running_mean"], a["weight"], a["bias"], a["running_var"]]
        try:
            if a["training"] or any(p is None for p in params):
                raise LoweringFailed("only inference mode with running statistics and affine "
                                     "parameters")
            if x.dtype != torch.float32 or any(p.dtype != torch.float32 for p in params) \
                    or x.dim() < 2 or x.numel() == 0 or not x.is_contiguous():
                raise LoweringFailed("only contiguous float32 operands")
            mean, weight, bias, var = [p.detach().clone() for p in params]
            C, shape, device = x.shape[1], tuple(x.shape), x.device
            view = (1, C) + (1,) * (x.dim() - 2)
            what = (f"batch_norm (inference, eps={a['eps']}"
                    + (f", cudnn.enabled={torch.backends.cudnn.enabled}"
                       if device.type == "cuda" else "")
                    + f") on float32{list(shape)} on {device}")
            names = [arg.name for arg in target._schema.arguments]

            def kernel(t, m=mean, w=weight, b=bias):
                vals = dict(a, input=t, weight=w, bias=b, running_mean=m, running_var=var)
                out = target(*[vals[n] for n in names])
                return out[0] if triple else out

            # invstd: with mean 0, weight 1, bias +0 and x = 1 every candidate returns s.
            with torch.no_grad():
                s_full = kernel(torch.ones(shape, device=device), torch.zeros_like(mean),
                                torch.ones_like(weight), torch.zeros_like(bias))
            per_channel = s_full.transpose(0, 1).reshape(C, -1)
            s = per_channel[:, 0].clone()
            if bool((per_channel != s[:, None]).any()) or not bool(torch.isfinite(s).all()) \
                    or bool((s <= 0).any()):
                raise LoweringFailed(f"{what}: the invstd read from the kernel is not one "
                                     "positive constant per channel")
            consts = [t.view(view) for t in (mean, weight, bias, s)]
            # The two derivations of beta are per-channel constants. Where they give the same
            # bits for these parameters, the candidates that differ only in it are one program.
            with torch.no_grad():
                alpha = consts[3] * consts[1]
                same_beta = bitwise_equal(P.fma32(-consts[0], alpha, consts[2]),
                                          consts[2] - consts[0] * alpha)
            cands = [(name, lambda t, g=g: g(t, *consts)) for name, g in batch_norm_candidates()
                     if not (same_beta and "fl(b - fl(mean * alpha))" in name)]
            gen = torch.Generator().manual_seed(self.seed + 1)
            # inputs around each channel's mean (cancellation in x - mean) on top of the usual
            near = [mean.view(view) * (1 + 2.0 ** -e * torch.randn(shape, generator=gen)
                                       .to(device)) for e in (3, 12, 20)]
            alive = select_candidates(kernel, cands, _float_inputs(
                shape, torch.float32, device, max(self.trials, 5), self.seed, near), what)
        except LoweringFailed as e:
            self.failures[node_name] = str(e)
            return None
        desc, cand = alive[0]
        n_inputs = max(self.trials, 5) + 5

        def check(*args, **kwargs):
            b = bind(target, args, kwargs)
            now = [b["running_mean"], b["weight"], b["bias"], b["running_var"]]
            if b.get("training") or tuple(b["input"].shape) != shape or any(
                    q is None or q.shape != p.shape or not bool((q == p).all())
                    for q, p in zip(now, (mean, weight, bias, var))):
                raise LoweringFailed(f"{node_name}: batch-norm statistics or parameters differ "
                                     "from the ones the reconstruction was built for")

        def fn(*args, **kwargs):
            out = cand(bind(target, args, kwargs)["input"])
            return (out, None, None) if triple else out

        return LoweredOp(
            fn, what, f"per element: {desc}; invstd is a per-channel constant read from the "
            "kernel (rsqrt(var + eps))", "validated",
            {"validation_inputs": n_inputs, "candidates": len(cands),
             "indistinguishable": [d for d, _ in alive[1:]],
             "invstd": "kernel call with mean 0, weight 1, bias 0 on ones"},
            [f"formula[{what}] selected among {len(cands)} explicit candidates by bitwise "
             f"validation on {n_inputs} inputs: the same for every element and independent of "
             "the values (dynamic evidence only).",
             "invstd = rsqrt(running_var + eps) is taken from the kernel as a frozen "
             "per-channel constant; it is not recomputed from explicit primitives and is "
             "assumed to depend on the running variance and eps only."],
            check=check)


def eager_lowerings(trials: int = 5, seed: int = 0) -> List[Lowering]:
    return [ClampLowering(trials, seed), MaxPoolLowering(trials, seed),
            BatchNormLowering(trials, seed)]
