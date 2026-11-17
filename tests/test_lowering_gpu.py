"""The real Triton and TileLang kernels, and cuBLAS, against their explicit reconstructions.

Needs CUDA (run on server3: CUDA_VISIBLE_DEVICES=7 pytest tests/test_lowering_gpu.py). Inputs
are the adversarial kinds of `lowering.adversarial_tensor`: wide exponents, cancellation,
subnormals and underflow, signed zeros. Agreement here is empirical evidence for the lowering's
assumptions on this GPU, not a proof for all inputs.
"""
import os
import sys

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)
nn = torch.nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import compilers  # noqa: E402
from dlclv import lowering as L  # noqa: E402
from dlclv import primitives as P  # noqa: E402
from dlclv.bitwise import bitwise_equal  # noqa: E402
from dlclv.reconstruction import UnsupportedReconstruction, reconstruct  # noqa: E402

compilers.strict_fp32()
f32 = torch.float32


def path(name):
    if name == "triton":
        pytest.importorskip("triton")
        return compilers.compile_triton
    pytest.importorskip("tilelang")
    return compilers.compile_tilelang


def cases(shape, n=10, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return [(L.adversarial_tensor(shape, L.KINDS[t % len(L.KINDS)], gen, f32, "cuda"),)
            for t in range(n)]


@pytest.mark.parametrize("name", ["triton", "tilelang"])
@pytest.mark.parametrize("K,split_k", [(70, 1), (64, 1), (300, 1), (70, 4), (300, 3), (5, 4)])
def test_linear_on_the_kernel_is_an_explicit_fma_chain(name, K, split_k):
    torch.manual_seed(0)
    model = nn.Linear(K, 10).cuda().eval().requires_grad_(False)
    inputs = cases((3, K), 10, seed=K)
    compiled = path(name)(model, inputs[0], split_k=split_k)
    rebuilt = reconstruct(compiled, inputs[0])                 # primitive mode: nothing opaque
    r = rebuilt.report
    assert rebuilt.compare(compiled, inputs) == {"samples": 10, "bitwise_equal": True,
                                                 "is_proof": False}
    assert r["lowering_levels"] == ["known"] and not r["preserved_kernels"]
    assert r["device"] == "cuda" and r["lowered_primitive_ops"]["aten.nextafter.default"] > 0
    ev = r["lowered_kernels"][0]["evidence"]
    assert ev["validation_inputs"] == 5 and "fprev" in ev
    if split_k == 1:
        assert ev["fma_probe"] == "fma"
    else:
        assert "split_k_combine" in ev
    # The explicit program gives the same bits on the CPU: nothing in it is GPU arithmetic.
    w, b = model.weight.cpu(), model.bias.cpu()
    combine = None
    if split_k > 1:
        combine = L.probe_sum((1, split_k, 3, 10), 1, f32, torch.device("cuda"))
        combine = {"tree": combine["tree"], "zero_init": combine["zero_init"]}
    for (x,) in inputs:
        cpu = L.dsl_bmm(x.cpu().reshape(1, 3, K), w.t().unsqueeze(0), name, split_k,
                        compiled.options["block_k"], combine)[0] + b
        assert bitwise_equal(cpu, compiled(x).cpu())


@pytest.mark.parametrize("name", ["triton", "tilelang"])
def test_kernel_order_is_observable(name):
    """The kernel path differs in bits from cuBLAS and from a separately rounded chain."""
    torch.manual_seed(0)
    K = 300
    model = nn.Linear(K, 10).cuda().eval().requires_grad_(False)
    inputs = cases((3, K), 10, seed=1)
    compiled = path(name)(model, inputs[0])
    with torch.no_grad():
        assert not all(bitwise_equal(model(x), compiled(x)) for (x,) in inputs)
        sep = [P.chain_dot(x.reshape(1, 3, K), model.weight.t().unsqueeze(0), fp="separate")[0]
               + model.bias for (x,) in inputs]
    assert not all(bitwise_equal(s, compiled(x)) for s, (x,) in zip(sep, inputs))


class Cnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 8, 3, stride=2, padding=1)
        self.c2 = nn.Conv2d(8, 8, 3, padding=1, groups=4, bias=False)
        self.c3 = nn.Conv1d(8, 4, 5, stride=2, dilation=2)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(4 * 4, 5)
        self.w = nn.Parameter(torch.randn(5, 7))

    def forward(self, x):
        y = self.pool(torch.relu(self.c2(torch.relu(self.c1(x)))))      # [N, 8, 4, 4]
        y = torch.relu(self.c3(y.flatten(2)))                           # [N, 4, 4]
        return torch.matmul(self.fc(y.flatten(1)), self.w)


@pytest.mark.parametrize("name,split_k", [("triton", 1), ("triton", 2), ("tilelang", 1)])
def test_cnn_on_the_kernel_path(name, split_k):
    torch.manual_seed(0)
    model = Cnn().cuda().eval().requires_grad_(False)
    inputs = cases((2, 3, 16, 16), 10, seed=5)
    compiled = path(name)(model, inputs[0], split_k=split_k)
    with pytest.raises(UnsupportedReconstruction, match="relu|max_pool"):
        reconstruct(compiled, inputs[0])
    rebuilt = reconstruct(compiled, inputs[0], mode="eager-kernel")
    r = rebuilt.report
    assert rebuilt.compare(compiled, inputs)["bitwise_equal"]
    assert len(r["lowered_kernels"]) == 5 and r["lowering_levels"] == ["known"]
    assert not any(k.split(".")[1] in ("convolution", "conv2d", "conv1d", "linear", "matmul",
                                       "mm", "bmm", "addmm") for k in r["preserved_kernels"])
    assert "im2col" in rebuilt.code


class Block(nn.Module):
    """A residual block as in ResNet: conv, batch norm, in-place ReLU, max pooling, skip
    connection, global average pooling, classifier."""

    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 8, 3, stride=2, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(8)
        self.c2 = nn.Conv2d(8, 8, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(3, 2, 1)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, 5)

    def forward(self, x):
        y = self.pool(self.relu(self.b1(self.c1(x))))
        out = self.b2(self.c2(y))
        out += y
        return self.fc(torch.flatten(self.avg(self.relu(out)), 1))


@pytest.mark.parametrize("name,split_k", [("triton", 1), ("triton", 2), ("tilelang", 1)])
def test_residual_block_has_no_opaque_kernel_left(name, split_k):
    """Kernel path GEMMs from the kernel source; batch norm, ReLU, pooling and the average
    from the behaviour of the library kernels: primitive mode, nothing preserved."""
    torch.manual_seed(0)
    model = Block()
    with torch.no_grad():
        for b in (model.b1, model.b2):
            b.weight.normal_(), b.bias.normal_(), b.running_mean.normal_()
            b.running_var.uniform_(0.1, 2.0)
    model = model.cuda().eval().requires_grad_(False)
    inputs = cases((2, 3, 20, 20), 10, seed=9)
    inputs += [(torch.full((2, 3, 20, 20), v, device="cuda"),) for v in (-0.0, 0.0, 1.0)]
    compiled = path(name)(model, inputs[0], split_k=split_k)
    rebuilt = reconstruct(compiled, inputs[0], probe_kernels=True)
    r = rebuilt.report
    assert rebuilt.mode == "primitive" and not r["preserved_kernels"]
    assert rebuilt.compare(compiled, inputs)["bitwise_equal"]
    assert r["lowering_levels"] == ["known", "validated"]
    kinds = [k["kernel"].split(" ")[0] for k in r["lowered_kernels"]]
    assert kinds.count("batch_norm") == 2 and kinds.count("max_pool2d") == 1
    assert sum("GEMM" in k["kernel"] for k in r["lowered_kernels"]) == 3
    assert not any(o.split(".")[1] in ("batch_norm", "relu", "max_pool2d", "mean", "sum",
                                       "adaptive_avg_pool2d", "convolution", "linear", "addmm")
                   for o in r["lowered_primitive_ops"])


@pytest.mark.parametrize("batch,K,N", [(1, 96, 5), (4, 64, 8), (1, 512, 10), (8, 768, 16)])
def test_cublas_linear_rebuilt_from_behaviour(batch, K, N):
    torch.manual_seed(0)
    model = nn.Linear(K, N).cuda().eval().requires_grad_(False)
    inputs = cases((batch, K), 10, seed=3)
    try:
        rebuilt = reconstruct(model, inputs[0], probe_kernels=True)
    except UnsupportedReconstruction as e:
        assert "lowering attempted" in str(e)
        pytest.skip(f"cuBLAS order not established: {str(e)[-300:]}")
    assert rebuilt.compare(model, inputs)["bitwise_equal"]
    assert rebuilt.report["lowering_levels"] == ["validated"]


@pytest.mark.parametrize("shape,dim", [((2, 4, 16, 16), 1), ((8, 300), 1), ((5, 7, 3), 0)])
def test_cuda_sum_rebuilt_from_behaviour(shape, dim):
    try:
        ev = L.probe_sum(shape, dim, f32, torch.device("cuda"), trials=10)
    except L.LoweringFailed as e:
        pytest.skip(f"CUDA sum order not established: {e}")
    for (x,) in cases(shape, 10, seed=7):
        assert bitwise_equal(ev["apply"](x), x.sum(dim))


def test_cudnn_convolution_that_is_not_a_product_tree_is_refused_or_exact():
    """A 3x3 stride-1 convolution with many channels runs a cuDNN algorithm that is not one
    float32 FMA tree of the products. Either the probe establishes a model that is then
    bitwise exact, or the call is refused with the reason; it is never approximated."""
    torch.manual_seed(0)
    model = nn.Conv2d(64, 64, 3, padding=1, bias=False).cuda().eval().requires_grad_(False)
    inputs = cases((1, 64, 16, 16), 6, seed=11)
    try:
        rebuilt = reconstruct(model, inputs[0], probe_kernels=True)
    except UnsupportedReconstruction as e:
        assert "lowering attempted" in str(e) and "conv2d" in str(e)
        kept = reconstruct(model, inputs[0], probe_kernels=True, mode="eager-kernel")
        assert kept.report["preserved_kernels"] == ["aten.conv2d.default"]
        assert kept.report["lowering_failures"] and kept.compare(model, inputs)["bitwise_equal"]
        return
    assert rebuilt.compare(model, inputs)["bitwise_equal"]
