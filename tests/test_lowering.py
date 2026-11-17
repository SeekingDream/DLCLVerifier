"""Kernel lowerings on CPU.

The Triton/TileLang kernels need a GPU (tests/test_lowering_gpu.py). Here the kernel is a
stand-in: a literal transcription of the loops in kernels.py, evaluated with the exact rational
oracle (fp_oracle), behind the repository's own `compilers.GemmMode`. So the reconstruction is
compared with an independent implementation of the kernel source, and every part of the
frontend except the GPU compilers is exercised: trace matching, operand reshapes, im2col,
FPRev and FMA probes, validation, the report.

Library kernels (`torch.sum`, `F.linear` on this machine's BLAS) are probed for real; a test
skips when this machine's kernel has no order the probe can establish.
"""
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import compilers  # noqa: E402
import fp_oracle as fo  # noqa: E402
from dlclv import fpmodel as fpm  # noqa: E402
from dlclv import lowering as L  # noqa: E402
from dlclv import primitives as P  # noqa: E402
from dlclv.bitwise import bitwise_equal  # noqa: E402
from dlclv.reconstruction import UnsupportedReconstruction, reconstruct  # noqa: E402

f32 = torch.float32


# ---- the kernels of kernels.py, transcribed loop by loop -----------------------------------

def triton_kernel(a, b, block_k=4, split_k=1, step=fo.fma, reverse=False):
    """kernels.triton_bmm_kernel + compilers.triton_bmm on numpy float32 arrays."""
    Bt, M, K = a.shape
    N = b.shape[2]
    k_per = -(-(-(-K // split_k)) // block_k) * block_k
    c = np.zeros((Bt, split_k, M, N), np.float32)
    for bat in range(Bt):
        for ks in range(split_k):
            k_lo = ks * k_per
            k_hi = min(k_lo + k_per, K)
            for m in range(M):
                for n in range(N):
                    acc = np.float32(0.0)                      # tl.zeros
                    blocks = list(range(k_lo, k_hi, block_k))
                    for k in (blocks[::-1] if reverse else blocks):
                        for kk in range(k, k + block_k):       # tl.dot over one K block
                            live = kk < k_hi                   # mask, other=0.0
                            acc = step(a[bat, m, kk] if live else 0.0,
                                       b[bat, kk, n] if live else 0.0, acc)
                    c[bat, ks, m, n] = acc
    return c


def tilelang_kernel(a, b, block_k=4, split_k=1):
    """kernels.tilelang_bmm_kernel + compilers.tilelang_bmm (zero padding to split_k chunks)."""
    Bt, M, K = a.shape
    N = b.shape[2]
    Kc = -(-K // split_k) if split_k > 1 else K
    pad = Kc * split_k - K
    a = np.concatenate([a, np.zeros((Bt, M, pad), np.float32)], 2)
    b = np.concatenate([b, np.zeros((Bt, pad, N), np.float32)], 1)
    c = np.zeros((Bt, split_k, M, N), np.float32)
    for bat in range(Bt):
        for s in range(split_k):
            for m in range(M):
                for n in range(N):
                    acc = np.float32(0.0)                      # T.clear(Cl)
                    for ko in range(-(-Kc // block_k)):
                        for k in range(block_k):
                            kk = s * Kc + ko * block_k + k
                            live = ko * block_k + k < Kc       # tiles are zero out of range
                            acc = fo.fma(a[bat, m, kk] if live else 0.0,
                                         b[bat, kk, n] if live else 0.0, acc)
                    c[bat, s, m, n] = acc
    return c


def separate_step(x, y, acc):
    return fo.add(acc, fo.mul(x, y))


def as_bmm(kernel, **opts):
    """A torch `bmm(a, b)` around a transcribed kernel, combining split-K as the paths do."""
    def bmm(a, b):
        Bt = a.shape[0]
        b = b.expand(Bt, -1, -1) if b.shape[0] != Bt else b
        c = torch.from_numpy(kernel(a.contiguous().numpy(), b.contiguous().numpy(), **opts))
        return c[:, 0] if c.shape[1] == 1 else c.sum(1)
    return bmm


class CpuGemmMode(compilers.GemmMode):
    @staticmethod
    def _ok(*ts):
        return all(t.dtype == torch.float32 for t in ts)


def kernel_path(model, backend, bmm, **options):
    """compilers._compile_kernel_path without the CUDA requirement."""
    mode = CpuGemmMode(bmm)

    def run(*inputs):
        with mode:
            return model(*inputs)

    def kernel_trace(*inputs):
        mode.log = []
        try:
            out = run(*inputs)
        finally:
            calls, mode.log = mode.log, None
        return out, calls

    return compilers.Compiled(backend, dict(options), run, stats=mode.stats, source_model=model,
                              meta={"kernel_trace": kernel_trace, "bmm": bmm})


def cases(shape, n=5, seed=0, dtype=f32):
    gen = torch.Generator().manual_seed(seed)
    return [(L.adversarial_tensor(shape, L.KINDS[t % len(L.KINDS)], gen, dtype, "cpu"),)
            for t in range(n)]


# ---- k ranges and the explicit bmm ----------------------------------------------------------

def test_k_ranges_follow_the_kernel_sources():
    assert L.k_ranges("triton", 10, 1, 4) == [(0, 10, True)]
    assert L.k_ranges("triton", 8, 1, 4) == [(0, 8, False)]
    # k_per = cdiv(cdiv(10, 3), 4) * 4 = 4
    assert L.k_ranges("triton", 10, 3, 4) == [(0, 4, False), (4, 8, False), (8, 10, True)]
    # k_per = 8 covers K in the first two splits; the third is empty and has no steps
    assert L.k_ranges("triton", 10, 3, 8) == [(0, 8, False), (8, 10, True), (16, 16, False)]
    # Kc = 4: the chunks are [0,4) [4,8) [8,10)+2 zeros; block_k = 4 adds no further steps
    assert L.k_ranges("tilelang", 10, 3, 4) == [(0, 4, False), (4, 8, False), (8, 10, True)]
    assert L.k_ranges("tilelang", 10, 1, 4) == [(0, 10, True)]
    assert L.k_ranges("tilelang", 9, 3, 2) == [(0, 3, True), (3, 6, True), (6, 9, True)]


@pytest.mark.parametrize("backend,K,block_k,split_k", [
    ("triton", 7, 4, 1), ("triton", 8, 4, 1), ("triton", 10, 4, 3), ("triton", 10, 8, 3),
    ("tilelang", 7, 4, 1), ("tilelang", 10, 4, 3), ("tilelang", 9, 2, 3)])
def test_explicit_bmm_equals_the_transcribed_kernel(backend, K, block_k, split_k):
    """Partial sums are combined left to right here; the real combine is probed (below)."""
    kernel = triton_kernel if backend == "triton" else tilelang_kernel
    combine = {"tree": fpm.SumTree.sequential(split_k, zero_init=False), "zero_init": False}
    gen = torch.Generator().manual_seed(K * 100 + block_k * 10 + split_k)
    for t in range(5):
        kind = L.KINDS[t]
        a = L.adversarial_tensor((2, 3, K), kind, gen, f32, "cpu")
        b = L.adversarial_tensor((2, K, 2), "normal" if kind in ("spread", "cancel") else kind,
                                 gen, f32, "cpu", 1)
        c = kernel(a.numpy(), b.numpy(), block_k=block_k, split_k=split_k)
        want = c[:, 0]
        for s in range(1, split_k):
            want = np.vectorize(fo.add, otypes=[np.float32])(want, c[:, s])
        with P.PrimitiveGuard():
            got = L.dsl_bmm(a, b, backend, split_k, block_k, combine)
        assert got.numpy().tobytes() == want.tobytes(), kind


def test_masked_tail_changes_the_sign_of_zero():
    """One product that underflows to -0. With K a multiple of BLOCK_K the kernel returns -0;
    with a masked tail its fma(+0, +0, acc) returns +0. The reconstruction follows both."""
    for K, negative in ((4, True), (3, False)):
        a = torch.zeros(1, 1, K)
        b = torch.zeros(1, K, 1)
        a[0, 0, K - 1], b[0, K - 1, 0] = 2.0 ** -80, -(2.0 ** -80)
        want = triton_kernel(a.numpy(), b.numpy(), block_k=4)[:, 0]
        got = L.dsl_bmm(a, b, "triton", 1, 4)
        assert got.numpy().tobytes() == want.tobytes()
        assert bool(torch.signbit(got).item()) is negative and got.item() == 0.0


# ---- a whole program on the (stand-in) kernel path --------------------------------------------

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 4, 3, stride=2, padding=1, groups=2)
        self.fc = nn.Linear(36, 5)
        self.w = nn.Parameter(torch.randn(5, 3))

    def forward(self, x):
        y = self.conv(x).flatten(1)
        return torch.matmul(self.fc(y), self.w)


def net():
    torch.manual_seed(0)
    return Net().eval().requires_grad_(False)


@pytest.mark.parametrize("backend,kernel", [("triton", triton_kernel), ("tilelang", tilelang_kernel)])
def test_kernel_path_program_is_reconstructed_with_explicit_arithmetic(backend, kernel):
    model = net()
    compiled = kernel_path(model, backend, as_bmm(kernel, block_k=4), block_k=4, split_k=1)
    inputs = cases((2, 2, 6, 6), 6)
    rebuilt = reconstruct(compiled, inputs[0])
    r = rebuilt.report
    assert rebuilt.compare(compiled, inputs) == {"samples": 6, "bitwise_equal": True,
                                                 "is_proof": False}
    assert r["status"] == "conditional" and not r["universal_certificate"]
    assert r["lowering_levels"] == ["known"] and not r["preserved_kernels"]
    assert len(r["lowered_kernels"]) == 3
    for k in r["lowered_kernels"]:
        assert k["evidence"]["fma_probe"] == "fma" and "chain" in k["evidence"]["fprev"]
        assert k["evidence"]["validation_inputs"] == 5 and "kernels." in k["evidence"]["static"]
    ops = r["lowered_primitive_ops"]
    assert ops["aten.nextafter.default"] > 0
    assert not any(n.split(".")[1] in ("mm", "bmm", "matmul", "sum", "convolution", "addmm",
                                       "linear", "addcmul", "im2col") for n in ops)
    assert "fma(a[m,k], b[k,n], acc)" in rebuilt.code and "im2col" in rebuilt.code
    # The kernel path is not the eager model: the library kernels use another order.
    with torch.no_grad():
        assert not all(bitwise_equal(model(*x), rebuilt(*x)) for x in inputs)
    # frozen snapshot of the weights
    before = rebuilt(*inputs[0])
    model.fc.bias.add_(1.0)
    assert bitwise_equal(before, rebuilt(*inputs[0]))


class Mixed(nn.Module):
    """A kernel call, a call the path leaves to PyTorch (1-D matmul), and an activation."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(12, 6)
        self.v = nn.Parameter(torch.randn(6))

    def forward(self, x):
        return torch.matmul(torch.relu(self.fc(x)), self.v)


def test_fallback_calls_stay_opaque_and_are_reported():
    torch.manual_seed(1)
    model = Mixed().eval().requires_grad_(False)
    compiled = kernel_path(model, "triton", as_bmm(triton_kernel, block_k=4), block_k=4, split_k=1)
    inputs = cases((3, 12), 5, seed=2)
    assert compiled.stats["fallback_matmul"] == 0
    with pytest.raises(UnsupportedReconstruction, match="need lowered arithmetic"):
        reconstruct(compiled, inputs[0])                      # primitive mode: relu, 1-D matmul
    rebuilt = reconstruct(compiled, inputs[0], mode="eager-kernel")
    assert compiled.stats["fallback_matmul"] > 0
    assert rebuilt.compare(compiled, inputs)["bitwise_equal"]
    assert len(rebuilt.report["lowered_kernels"]) == 1
    assert any("relu" in k for k in rebuilt.report["preserved_kernels"])
    assert len(rebuilt.report["preserved_kernels"]) >= 2
    assert any("opaque eager calls" in a for a in rebuilt.report["assumptions"])


@pytest.mark.parametrize("opts,why", [
    (dict(step=separate_step), "fused multiply-add|differs from the kernel"),
    (dict(reverse=True), "not the k-sequential")])
def test_a_kernel_with_other_arithmetic_is_rejected(opts, why):
    """Same real-number result, different bits: products rounded separately, or the K blocks
    visited in reverse. Neither may be passed off as the kernel of kernels.py."""
    model = net()
    compiled = kernel_path(model, "triton", as_bmm(triton_kernel, block_k=4, **opts),
                           block_k=4, split_k=1)
    x = cases((2, 2, 6, 6), 1)[0]
    with pytest.raises(UnsupportedReconstruction, match=why):
        reconstruct(compiled, x)
    # Without probing, the order rests on the source alone; the example comparison still fails.
    with pytest.raises(UnsupportedReconstruction, match="differs in output bits"):
        reconstruct(compiled, x, kernel_probe=False)


def test_unprobed_lowering_is_reported_as_source_only():
    model = net()
    compiled = kernel_path(model, "triton", as_bmm(triton_kernel, block_k=4), block_k=4, split_k=1)
    x = cases((2, 2, 6, 6), 1)[0]
    r = reconstruct(compiled, x, kernel_probe=False).report
    assert r["lowering_levels"] == ["source"]
    assert all("validation_inputs" not in k["evidence"] for k in r["lowered_kernels"])
    assert any("not probed" in a for a in r["assumptions"])


def test_graph_and_trace_must_agree():
    model = net()
    compiled = kernel_path(model, "triton", as_bmm(triton_kernel, block_k=4), block_k=4, split_k=1)
    x = cases((2, 2, 6, 6), 1)[0]
    trace = compiled.meta["kernel_trace"]
    compiled.meta["kernel_trace"] = lambda *i: (trace(*i)[0], trace(*i)[1][:-1])   # one call lost
    with pytest.raises(UnsupportedReconstruction, match="no matching call"):
        reconstruct(compiled, x)
    compiled.meta["kernel_trace"] = lambda *i: (trace(*i)[0], trace(*i)[1] + trace(*i)[1][:1])
    with pytest.raises(UnsupportedReconstruction, match="have no node"):
        reconstruct(compiled, x)
    del compiled.meta["kernel_trace"]
    with pytest.raises(UnsupportedReconstruction, match="kernel_trace"):
        reconstruct(compiled, x)


@pytest.mark.parametrize("backend,kernel", [("triton", triton_kernel), ("tilelang", tilelang_kernel)])
def test_split_k_partial_sums_and_probed_combine(backend, kernel):
    torch.manual_seed(2)
    model = nn.Linear(10, 3).eval().requires_grad_(False)
    compiled = kernel_path(model, backend, as_bmm(kernel, block_k=4, split_k=3),
                           block_k=4, split_k=3)
    inputs = cases((2, 10), 6, seed=4)
    try:
        rebuilt = reconstruct(compiled, inputs[0])
    except UnsupportedReconstruction as e:
        if "torch.sum" in str(e):
            pytest.skip(f"order of this machine's sum kernel not established: {e}")
        raise
    assert rebuilt.compare(compiled, inputs)["bitwise_equal"]
    ev = rebuilt.report["lowered_kernels"][0]["evidence"]
    assert "split_k_combine" in ev and "per split" in ev["fprev"] and "fma_probe" not in ev
    assert any("split-K combine" in a for a in rebuilt.report["assumptions"])
    single = kernel_path(model, backend, as_bmm(kernel, block_k=4), block_k=4, split_k=1)
    assert not all(bitwise_equal(single(*x), compiled(*x)) for x in inputs)   # split-K matters


# ---- library kernels, probed -------------------------------------------------------------------

def test_validation_rejects_a_wrong_tree():
    seq, pair = fpm.SumTree.sequential(8, zero_init=False), fpm.SumTree.pairwise(8)
    gen = torch.Generator().manual_seed(0)
    make = lambda t: (L.adversarial_tensor((4, 8), L.KINDS[t % 5], gen, f32, "cpu"),)  # noqa: E731
    assert L.validate_bitwise(lambda x: P.tree_sum(seq, x, 1), lambda x: P.tree_sum(seq, x, 1),
                              make, 5, "same tree") == 5
    with pytest.raises(L.LoweringFailed, match="differs from the kernel"):
        L.validate_bitwise(lambda x: P.tree_sum(seq, x, 1), lambda x: P.tree_sum(pair, x, 1),
                           make, 5, "other tree")
    with pytest.raises(L.LoweringFailed):                      # signed zero alone is a difference
        L.validate_bitwise(lambda x: P.tree_sum(seq, x, 1),
                           lambda x: P.tree_sum(seq, x, 1, zero_init=True), make, 5, "zero init")


class SumRows(nn.Module):
    def forward(self, x):
        return x.sum(1)


@pytest.mark.parametrize("shape", [(3, 40), (2, 300), (4, 7, 5)])
def test_library_sum_is_rebuilt_from_its_behaviour(shape):
    model = SumRows()
    inputs = cases(shape, 10, seed=1)
    with pytest.raises(UnsupportedReconstruction):
        reconstruct(model, inputs[0])                          # no generic sum in primitive mode
    try:
        rebuilt = reconstruct(model, inputs[0], probe_kernels=True)
    except UnsupportedReconstruction as e:
        assert "lowering attempted" in str(e)
        pytest.skip(f"order of this machine's sum kernel not established: {e}")
    assert rebuilt.compare(model, inputs)["bitwise_equal"]
    k = rebuilt.report["lowered_kernels"][0]
    assert k["level"] == "validated" and rebuilt.report["lowering_levels"] == ["validated"]
    assert any("dynamic evidence only" in a for a in rebuilt.report["assumptions"])
    assert set(rebuilt.report["lowered_primitive_ops"]) <= {
        "aten.add.Tensor", "aten.select.int", "aten.zeros_like.default", "aten.permute.default",
        "aten.reshape.default", "aten.view.default", "aten._unsafe_view.default",
        "aten.clone.default", "aten.where.self"}                                     # where: per-position trees
    ev = L.probe_sum(shape, 1, f32, torch.device("cpu"))
    seq = fpm.SumTree.sequential(shape[1], zero_init=False)
    if ev["tree"] is None or ev["tree"].children != seq.children:   # the order is observable
        assert not all(bitwise_equal(P.tree_sum(seq, x, 1, ev["zero_init"]), model(x))
                       for (x,) in inputs)


@pytest.mark.parametrize("batch,K,N,bias", [(1, 96, 5, True), (4, 64, 8, True), (1, 33, 4, False)])
def test_library_linear_is_rebuilt_from_its_behaviour(batch, K, N, bias):
    torch.manual_seed(0)
    model = nn.Linear(K, N, bias=bias).eval().requires_grad_(False)
    inputs = cases((batch, K), 10, seed=3)
    try:
        rebuilt = reconstruct(model, inputs[0], probe_kernels=True)
    except UnsupportedReconstruction as e:
        assert "lowering attempted" in str(e)
        pytest.skip(f"order of this machine's BLAS kernel not established: {e}")
    assert rebuilt.compare(model, inputs)["bitwise_equal"]
    k = rebuilt.report["lowered_kernels"][0]
    assert k["level"] == "validated" and k["evidence"]["validation_inputs"] >= 5
    assert not rebuilt.report["preserved_kernels"]
    assert not any(n.split(".")[1] in ("mm", "addmm", "matmul", "linear", "sum")
                   for n in rebuilt.report["lowered_primitive_ops"])
    # the tree matters: a plain k-sequential chain of the same products gives other bits
    w, b = model.weight, model.bias
    with torch.no_grad():
        chain = [P.chain_dot(x.reshape(1, -1, K), w.t().unsqueeze(0), fp="separate")[0]
                 + (0 if b is None else b) for (x,) in inputs]
        assert not all(bitwise_equal(c, model(x)) for c, (x,) in zip(chain, inputs))


def test_probe_failure_is_reported_not_papered_over():
    class Odd(L.Lowering):
        failures = {}

        def lower(self, node_name, target, args, kwargs):
            if "sum" in str(target):
                self.failures[node_name] = "order not established on purpose"
            return None

    with pytest.raises(UnsupportedReconstruction, match="order not established on purpose"):
        reconstruct(SumRows(), (torch.ones(2, 3),), lowerings=[Odd()])
    # eager-kernel mode keeps the opaque call and says why it was not lowered
    r = reconstruct(SumRows(), (torch.ones(2, 3),), mode="eager-kernel", lowerings=[Odd()]).report
    assert r["preserved_kernels"] == ["aten.sum.dim_IntList"] and r["lowering_failures"]


def test_a_lowering_cannot_smuggle_in_a_library_call():
    class Cheat(L.Lowering):
        failures = {}

        def lower(self, node_name, target, args, kwargs):
            if "sum" not in str(target):
                return None
            return L.LoweredOp(lambda x, dim: torch.sum(x, dim), "sum", "claims to be explicit",
                               "known")

    with pytest.raises(UnsupportedReconstruction, match="not an explicit primitive"):
        reconstruct(SumRows(), (torch.ones(2, 3),), lowerings=[Cheat()])


def test_compare_separates_nan_bits_from_real_differences():
    from torch.fx.experimental.proxy_tensor import make_fx
    x = torch.tensor([float("nan"), 1.0])
    p = reconstruct(make_fx(lambda x: x * 1.0)(x), (x,), format="aten")

    def other_nan(x):
        y = x * 1.0
        return torch.where(torch.isnan(y), -y, y)              # same values, NaN sign flipped

    res = p.compare(other_nan, [(x,)])
    assert res["bitwise_equal"] is False and res["nan_bits_only"] is True
    res = p.compare(lambda x: x * 2.0, [(x,)])
    assert res["bitwise_equal"] is False and res["nan_bits_only"] is False
