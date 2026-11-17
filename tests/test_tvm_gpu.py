"""The TVM compiler path and its reconstruction from the lowered TIR.

Needs CUDA and a TVM with Relax (the build bundled with TileLang is enough):
CUDA_VISIBLE_DEVICES=7 pytest tests/test_tvm_gpu.py. Agreement is empirical evidence for the
two compiler conventions the TIR leaves open (FMA contraction, max/min ties); the order of
every reduction is read from the TIR.
"""
import os
import sys

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)
nn = torch.nn
F = torch.nn.functional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import compilers  # noqa: E402

try:
    compilers._import_tvm()
    from tvm import relax  # noqa: F401,E402
except Exception as exc:                                       # pragma: no cover
    pytest.skip(f"no TVM with Relax: {exc}", allow_module_level=True)

from dlclv import lowering as L  # noqa: E402
from dlclv import tir_lowering as TL  # noqa: E402
from dlclv.bitwise import bitwise_equal  # noqa: E402
from dlclv.reconstruction import UnsupportedReconstruction, reconstruct  # noqa: E402

compilers.strict_fp32()
f32 = torch.float32


def cases(shape, n=10, seed=0):
    gen = torch.Generator().manual_seed(seed)
    xs = [(L.adversarial_tensor(shape, L.KINDS[t % len(L.KINDS)], gen, f32, "cuda"),)
          for t in range(n)]
    return xs + [(torch.full(shape, v, device="cuda"),) for v in (-0.0, 0.0, 1.0)]


def prepared(model):
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.normal_(), m.bias.normal_(), m.running_mean.normal_()
                m.running_var.uniform_(0.1, 2.0)
    return model.cuda().eval().requires_grad_(False)


class Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.a, self.b = nn.Linear(70, 33), nn.Linear(33, 5, bias=False)

    def forward(self, x):
        return self.b(torch.relu(self.a(x)))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 8, 3, stride=2, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(8)
        self.c2 = nn.Conv2d(8, 8, 3, padding=1, groups=2)
        self.b2 = nn.BatchNorm2d(8)
        self.fc = nn.Linear(8, 5)

    def forward(self, x):
        y = F.max_pool2d(torch.relu(self.b1(self.c1(x))), 3, 2, 1)
        out = torch.relu(self.b2(self.c2(y)) + y)
        return self.fc(torch.flatten(F.adaptive_avg_pool2d(out, 1), 1))


@pytest.mark.parametrize("fuse", [False, True])
def test_mlp_is_rebuilt_from_its_tir(fuse):
    torch.manual_seed(0)
    model = prepared(Mlp())
    inputs = cases((3, 70), seed=1)
    compiled = compilers.compile_tvm(model, inputs[0], fuse=fuse)
    assert compiled.compiler == "tvm" and compiled.options == {"schedule": "fallback",
                                                                "fuse": fuse}
    rebuilt = reconstruct(compiled, inputs[0])
    r = rebuilt.report
    assert rebuilt.compare(compiled, inputs) == {"samples": len(inputs), "bitwise_equal": True,
                                                 "is_proof": False}
    assert rebuilt.mode == "primitive" and not r["preserved_kernels"]
    assert r["lowering_levels"] == ["known"]
    gemm = [k for k in r["lowered_kernels"] if "matmul" in k["kernel"]]
    assert len(gemm) == 2
    assert gemm[0]["evidence"]["serial_loops"] == [["ax2", 70]] or \
        gemm[0]["evidence"]["serial_loops"] == [("ax2", 70)]
    assert all(k["evidence"]["kernel_source_identical_to_program"] for k in r["lowered_kernels"])
    assert "contracted to one FMA" in gemm[0]["evidence"]["model"]
    # contraction was observable: no surviving candidate is the uncontracted reading (in a
    # fused matmul+relu kernel the max tie conventions can remain indistinguishable)
    assert not any("no FMA" in m for m in gemm[0]["evidence"]["indistinguishable_models"])
    # no library kernel ran inside the reconstruction
    assert not any(o.split(".")[1] in ("mm", "matmul", "linear", "addmm", "sum", "relu", "max")
                   for o in r["lowered_primitive_ops"])
    assert r["lowered_primitive_ops"]["aten.nextafter.default"] > 0   # the explicit FMA


@pytest.mark.parametrize("fuse", [False, True])
def test_cnn_block_is_rebuilt_from_its_tir(fuse):
    torch.manual_seed(0)
    model = prepared(Block())
    inputs = cases((2, 3, 12, 12), seed=2)
    compiled = compilers.compile_tvm(model, inputs[0], fuse=fuse)
    rebuilt = reconstruct(compiled, inputs[0])
    r = rebuilt.report
    assert rebuilt.compare(compiled, inputs)["bitwise_equal"]
    assert not r["preserved_kernels"] and r["lowering_levels"] == ["known"]
    names = " ".join(k["kernel"] for k in r["lowered_kernels"])
    assert "conv2d" in names and "max_pool2d" in names and "mean" in names
    conv = next(k for k in r["lowered_kernels"] if "conv2d" in k["kernel"])
    assert [n for _, n in conv["evidence"]["serial_loops"]] == [3, 3, 3]    # c, kh, kw order
    relu = next(k for k in r["lowered_kernels"] if "relu" in k["kernel"])
    assert "max/min" in relu["evidence"]["model"]
    assert "tir_conv2d" in rebuilt.code or "tir_fused" in rebuilt.code


def test_tvm_arithmetic_differs_from_cudnn_and_from_the_uncontracted_tir():
    """The path is worth reconstructing: it is not eager PyTorch, and reading the TIR without
    the FMA contraction gives other bits too."""
    torch.manual_seed(0)
    model = prepared(Mlp())
    inputs = cases((3, 70), seed=3)
    compiled = compilers.compile_tvm(model, inputs[0])
    with torch.no_grad():
        assert not all(bitwise_equal(model(x), compiled(x)) for (x,) in inputs)
    func = compiled.meta["tvm_lowered"]["matmul"]
    w = model.a.weight.t().contiguous()
    fused, plain = [], []
    for (x,) in inputs:
        want = torch.full((3, 33), float("nan"), device="cuda")
        compiled.meta["tir_call"]("matmul", [x, w, want])
        with torch.no_grad():
            fused.append(TL.same_bits(TL.run_primfunc(func, [x, w], 1, TL.Model(True)), want))
            plain.append(TL.same_bits(TL.run_primfunc(func, [x, w], 1, TL.Model(False)), want))
    assert all(fused) and not all(plain)


def test_max_convention_is_read_from_the_kernel_not_assumed():
    class Relu(nn.Module):
        def forward(self, x):
            return torch.relu(x * 1.0)

    x = torch.tensor([[-0.0, 0.0, -1.0, 2.0]], device="cuda")
    compiled = compilers.compile_tvm(Relu().cuda(), (x,))
    rebuilt = reconstruct(compiled, (x,))
    assert rebuilt.compare(compiled, [(x,), (-x,)])["bitwise_equal"]
    ev = next(k for k in rebuilt.report["lowered_kernels"] if "relu" in k["kernel"])["evidence"]
    assert "max/min" in ev["model"]
    # the conventions are different programs on signed zeros
    func = compiled.meta["tvm_lowered"]["relu"]
    y = x * 1.0
    outs = {t: TL.run_primfunc(func, [y], 1, TL.Model(True, t)) for t in TL.TIES}
    assert not bitwise_equal(outs["first"], outs["second"])


def test_unsupported_tir_is_refused_with_the_reason():
    torch.manual_seed(0)
    model = prepared(Block())
    inputs = cases((2, 3, 12, 12), 2, seed=4)
    compiled = compilers.compile_tvm(model, inputs[0], schedule="reduction")
    with torch.no_grad():
        assert compiled(*inputs[0]).shape == (2, 5)
    with pytest.raises(UnsupportedReconstruction, match="init / match_buffer|not modelled"):
        reconstruct(compiled, inputs[0])                       # cross-thread reduction blocks
    with pytest.raises(UnsupportedReconstruction, match="no eager kernels"):
        reconstruct(compilers.compile_tvm(model, inputs[0]), inputs[0], mode="eager-kernel")

    class Tanh(nn.Module):
        def forward(self, x):
            return torch.tanh(x)

    x = (torch.randn(2, 4, device="cuda"),)
    with pytest.raises(UnsupportedReconstruction, match="not an explicit primitive"):
        reconstruct(compilers.compile_tvm(Tanh().cuda(), x), x)


def test_load_compiler_knows_tvm():
    import utils
    assert utils.list_compilers()[6] == "tvm"
    torch.manual_seed(0)
    model = prepared(Mlp())
    x = (torch.randn(3, 70, device="cuda"),)
    compiled = utils.load_compiler("tvm")(model, x)
    with torch.no_grad():
        assert torch.allclose(compiled(*x), model(*x), atol=1e-4)
