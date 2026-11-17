"""PyTorch eager kernels without an accumulation order (ReLU and clamps, max pooling,
inference batch norm) and averages, rebuilt as explicit primitives.

Runs on the CPU and, when there is one, on the GPU: the two devices use different conventions
(`relu(-0.0)`, the batch-norm formula, the summation tree), which is the point of selecting
the explicit program from the kernel's behaviour instead of from the operator's definition.
Agreement is finite-sample evidence at level "validated".
"""
import os
import sys

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn
F = torch.nn.functional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dlclv import eager_lowering as E  # noqa: E402
from dlclv import lowering as L  # noqa: E402
from dlclv import primitives as P  # noqa: E402
from dlclv.bitwise import bitwise_equal  # noqa: E402
from dlclv.reconstruction import (ExecutionMismatch, UnsupportedReconstruction,  # noqa: E402
                                  reconstruct)

f32 = torch.float32
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
aten = torch.ops.aten


def cases(shape, device, n=10, seed=0):
    gen = torch.Generator().manual_seed(seed)
    xs = [(L.adversarial_tensor(shape, L.KINDS[t % len(L.KINDS)], gen, f32, device),)
          for t in range(n)]
    return xs + [(torch.full(shape, v, device=device),) for v in (-0.0, 0.0)]


def rebuild(model, inputs, **kw):
    """Primitive mode with behavioural lowerings; skip if this machine's kernel has no model."""
    try:
        return reconstruct(model, inputs, probe_kernels=True, **kw)
    except UnsupportedReconstruction as e:
        if "lowering attempted" not in str(e):
            raise
        pytest.skip(f"kernel not modelled on this machine: {str(e)[-300:]}")


class Act(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind

    def forward(self, x):
        y = x * 1.0                                 # a fresh intermediate
        if self.kind == "relu":
            return torch.relu(y)
        if self.kind == "relu_":
            return F.relu(y, inplace=True)
        if self.kind == "relu6":
            return F.relu6(y)
        if self.kind == "hardtanh":
            return F.hardtanh(y, -1.0, 1.0)
        return torch.clamp(y, min=0.0, max=0.5)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", ["relu", "relu_", "relu6", "hardtanh", "clamp"])
def test_clamps_are_selections_with_the_kernels_zero_sign(device, kind):
    model = Act(kind)
    inputs = cases((3, 4, 9), device, seed=1)
    with pytest.raises(UnsupportedReconstruction):
        reconstruct(model, inputs[0])                          # opaque without a lowering
    rebuilt = rebuild(model, inputs[0])
    r = rebuilt.report
    assert rebuilt.compare(model, inputs)["bitwise_equal"]
    assert not r["preserved_kernels"] and r["lowering_levels"] == ["validated"]
    k = r["lowered_kernels"][0]
    assert "where(" in k["arithmetic"] and k["evidence"]["validation_inputs"] >= 7
    # only selections ran: no arithmetic, no library clamp
    assert set(r["lowered_primitive_ops"]) <= {
        "aten.where.self", "aten.gt.Scalar", "aten.lt.Scalar", "aten.full_like.default"}
    if kind != "hardtanh":                                     # a zero bound: the tie matters
        assert k["evidence"]["zero_bound"] and not k["evidence"]["indistinguishable"]


def test_the_two_tie_conventions_differ_exactly_on_signed_zero():
    cands = E.clamp_candidates(0.0, None)
    assert len(cands) == 2
    x = torch.tensor([-0.0, 0.0, -1.0, 2.0, float("-inf")])
    a, b = (fn(x) for _, fn in cands)
    assert torch.equal(a, b) and not bitwise_equal(a, b)
    assert torch.signbit(a).tolist() != torch.signbit(b).tolist()
    # a kernel that follows neither convention is refused, not approximated
    with pytest.raises(L.LoweringFailed, match="none of 2 explicit candidates"):
        E.select_candidates(lambda t: torch.relu(t) + 0.25, cands, [x], "shifted relu")
    with pytest.raises(L.LoweringFailed, match="NaN"):
        E.select_candidates(lambda t: t * float("nan"), cands, [x], "nan kernel")


class Pool(nn.Module):
    def __init__(self, *a, **k):
        super().__init__()
        self.a, self.k = a, k

    def forward(self, x):
        return F.max_pool2d(x * 1.0, *self.a, **self.k)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("args,kwargs", [((3, 2, 1), {}), ((2,), {}), ((3, 1, 1), {"dilation": 2}),
                                         (((2, 3), (1, 2), (1, 0)), {})])
def test_max_pool_is_a_first_maximum_scan(device, args, kwargs):
    model = Pool(*args, **kwargs)
    inputs = cases((2, 3, 11, 12), device, seed=2)
    rebuilt = rebuild(model, inputs[0])
    r = rebuilt.report
    assert rebuilt.compare(model, inputs)["bitwise_equal"]
    assert not r["preserved_kernels"]
    assert r["lowering_levels"] == ["known" if device == "cuda" else "validated"]
    assert r["lowered_kernels"][0]["evidence"]["tie_order_observable"]
    assert set(r["lowered_primitive_ops"]) <= {
        "aten.where.self", "aten.gt.Tensor", "aten.slice.Tensor", "aten.constant_pad_nd.default"}


def test_max_pool_scan_order_decides_the_sign_of_zero():
    x = torch.tensor([[[[0.0, -0.0], [-0.0, -0.0]]]])
    y = torch.tensor([[[[-0.0, 0.0], [0.0, 0.0]]]])
    first = [E.max_pool2d_explicit(t, (2, 2), (2, 2), (0, 0), (1, 1)) for t in (x, y)]
    assert [bool(torch.signbit(t)) for t in first] == [False, True]
    for t, e in zip((x, y), first):
        assert bitwise_equal(F.max_pool2d(t, 2), e)
    # ceil_mode and index outputs are not modelled: reported, not approximated
    with pytest.raises(UnsupportedReconstruction, match="ceil_mode"):
        reconstruct(Pool(2, ceil_mode=True), (torch.ones(1, 1, 5, 5),), probe_kernels=True)

    class Indices(nn.Module):
        def forward(self, x):
            return F.max_pool2d(x * 1.0, 2, return_indices=True)[1]

    with pytest.raises(UnsupportedReconstruction):
        reconstruct(Indices(), (torch.ones(1, 1, 4, 4),), probe_kernels=True)


def bn(C, dims, device, seed=0):
    torch.manual_seed(seed)
    m = (nn.BatchNorm2d if dims == 4 else nn.BatchNorm1d)(C)
    with torch.no_grad():
        m.weight.normal_(), m.bias.normal_(), m.running_mean.normal_()
        m.running_var.uniform_(0.01, 3.0)
    return m.to(device).eval().requires_grad_(False)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", [(2, 16, 7, 5), (1, 8, 16, 16), (3, 16), (4, 6, 9)])
def test_batch_norm_formula_is_selected_from_the_kernel(device, shape):
    model = bn(shape[1], 4 if len(shape) == 4 else 3, device)
    inputs = cases(shape, device, seed=3)
    with pytest.raises(UnsupportedReconstruction):
        reconstruct(model, inputs[0])
    rebuilt = rebuild(model, inputs[0])
    r = rebuilt.report
    assert rebuilt.compare(model, inputs)["bitwise_equal"]
    assert not r["preserved_kernels"] and r["lowering_levels"] == ["validated"]
    k = r["lowered_kernels"][0]
    assert k["arithmetic"].split("per element: ")[1].split(";")[0] in [
        d for d, _ in E.batch_norm_candidates()]
    assert not k["evidence"]["indistinguishable"]              # the formula was identified
    assert any("invstd" in a and "not recomputed" in a for a in r["assumptions"])
    assert not any(n.split(".")[1] in ("batch_norm", "rsqrt", "sqrt", "addcmul")
                   for n in r["lowered_primitive_ops"])
    # the textbook formula is not what the kernel computes
    w, b, m, v = (t.view((1, -1) + (1,) * (len(shape) - 2)) for t in (
        model.weight, model.bias, model.running_mean, model.running_var))
    assert not all(bitwise_equal((x - m) / torch.sqrt(v + model.eps) * w + b, model(x))
                   for (x,) in inputs)


@pytest.mark.parametrize("device", DEVICES)
def test_batch_norm_constants_are_pinned(device):
    model = bn(4, 4, device)
    x = torch.randn(2, 4, 3, 3, device=device)
    args = (x, model.weight, model.bias, model.running_mean, model.running_var, False, 0.1,
            1e-5, True)
    low = E.BatchNormLowering()
    op = low.lower("bn", aten.batch_norm.default, args, {})
    assert op is not None, low.failures
    op.check(*args)
    with pytest.raises(L.LoweringFailed, match="statistics or parameters differ"):
        op.check(x, model.weight, model.bias, model.running_mean, model.running_var * 2, False,
                 0.1, 1e-5, True)
    # training mode and missing statistics are declined with a reason
    assert low.lower("t", aten.batch_norm.default, args[:5] + (True,) + args[6:], {}) is None
    assert "inference" in low.failures["t"]


class Reduce(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind

    def forward(self, x):
        y = x * 1.0
        if self.kind == "aap":
            return F.adaptive_avg_pool2d(y, 1)
        if self.kind == "mean":
            return y.mean((2, 3))
        if self.kind == "mean1":
            return y.mean(1, keepdim=True)
        return y.sum((1, 3))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", ["aap", "mean", "mean1", "sum2"])
@pytest.mark.parametrize("shape", [(2, 5, 7, 7), (1, 8, 4, 4), (3, 4, 2, 9)])
def test_averages_and_multi_dimension_sums(device, kind, shape):
    model = Reduce(kind)
    inputs = cases(shape, device, seed=4)
    with pytest.raises(UnsupportedReconstruction):
        reconstruct(model, inputs[0])
    rebuilt = rebuild(model, inputs[0])
    r = rebuilt.report
    assert rebuilt.compare(model, inputs)["bitwise_equal"]
    assert not r["preserved_kernels"] and r["lowering_levels"] == ["validated"]
    ev = r["lowered_kernels"][0]["evidence"]
    n = {"aap": shape[2] * shape[3], "mean": shape[2] * shape[3], "mean1": shape[1],
         "sum2": shape[1] * shape[3]}[kind]
    assert ev["leaves"] == n and ev["divide_by"] == (None if kind == "sum2" else n)
    assert not any(o.split(".")[1] in ("sum", "mean", "adaptive_avg_pool2d")
                   for o in r["lowered_primitive_ops"])


def test_adaptive_pool_to_other_sizes_is_not_lowered():
    class M(nn.Module):
        def forward(self, x):
            return F.adaptive_avg_pool2d(x, 2)

    with pytest.raises(UnsupportedReconstruction, match="output size 1x1|_adaptive_avg_pool2d"):
        reconstruct(M(), (torch.ones(1, 2, 4, 4),), probe_kernels=True)


class Head(nn.Module):
    """Everything after the convolutions of a small CNN block."""

    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(6)
        self.fc = nn.Linear(12, 4)

    def forward(self, x):
        y = F.max_pool2d(F.relu(self.bn(x)), 3, 2, 1)
        y = torch.cat([y, y * 0.5], 1)
        return self.fc(F.adaptive_avg_pool2d(y, 1).flatten(1))


@pytest.mark.parametrize("device", DEVICES)
def test_block_without_opaque_kernels(device):
    torch.manual_seed(0)
    model = Head()
    with torch.no_grad():
        model.bn.running_mean.normal_(), model.bn.running_var.uniform_(0.1, 2.0)
    model = model.to(device).eval().requires_grad_(False)
    inputs = cases((2, 6, 9, 9), device, seed=6)
    rebuilt = rebuild(model, inputs[0])
    r = rebuilt.report
    assert rebuilt.compare(model, inputs)["bitwise_equal"]
    assert not r["preserved_kernels"] and len(r["lowered_kernels"]) == 5
    assert rebuilt.mode == "primitive"
    with pytest.raises(ExecutionMismatch):
        rebuilt(inputs[0][0][:1])                              # another shape: not this program


class Conv(nn.Module):
    def __init__(self, *a, **k):
        super().__init__()
        self.conv = nn.Conv2d(*a, **k)

    def forward(self, x):
        return self.conv(x)


CONVS = [((3, 8, 3), dict(padding=1), (2, 3, 8, 8)),
         ((4, 8, 1), dict(bias=False), (1, 4, 8, 8)),
         ((3, 16, 7), dict(stride=2, padding=3, bias=False), (1, 3, 32, 32)),
         ((8, 8, 3), dict(stride=2, padding=1, groups=4), (2, 8, 16, 16)),
         ((8, 16, 3), dict(padding=2, dilation=2), (1, 8, 12, 12))]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("args,kwargs,shape", CONVS)
def test_eager_convolution_is_a_tree_of_products(device, args, kwargs, shape):
    """cuDNN's implicit-GEMM convolutions are one FMA chain per output element; the lowering
    finds that from the kernel (or reports why it cannot, and the test skips)."""
    torch.manual_seed(0)
    model = Conv(*args, **kwargs).to(device).eval().requires_grad_(False)
    inputs = cases(shape, device, seed=8)
    with pytest.raises(UnsupportedReconstruction):
        reconstruct(model, inputs[0])                          # no generic convolution
    rebuilt = rebuild(model, inputs[0])
    r = rebuilt.report
    assert rebuilt.compare(model, inputs)["bitwise_equal"]
    assert not r["preserved_kernels"] and r["lowering_levels"] == ["validated"]
    k = r["lowered_kernels"][0]
    K = model.conv.weight[0].numel()
    assert k["evidence"]["K"] == K and "im2col" in k["arithmetic"]
    assert k["evidence"]["interior_positions"] >= 1
    assert not any(o.split(".")[1] in ("convolution", "conv2d", "mm", "matmul", "sum", "unfold")
                   for o in r["lowered_primitive_ops"])


def test_convolution_without_interior_position_is_refused():
    model = Conv(4, 4, 3, padding=1).eval().requires_grad_(False)
    x = (torch.randn(1, 4, 2, 2),)
    with pytest.raises(UnsupportedReconstruction, match="no output position has its window"):
        reconstruct(model, x, probe_kernels=True)
    r = reconstruct(model, x, probe_kernels=True, mode="eager-kernel").report
    assert r["preserved_kernels"] in (["aten.conv2d.default"], ["aten.convolution.default"])
    assert "cannot be probed" in list(r["lowering_failures"].values())[0]


def test_two_unrounded_products_under_one_node_have_no_model():
    from dlclv.fpmodel import SumTree
    from dlclv.fprev import RevealError
    tree = SumTree(3, [(0, 1), (3, 2)])
    L.check_fused(tree, [False, True, True])                  # one FMA per node
    with pytest.raises(RevealError, match="both enter one node unrounded"):
        L.check_fused(tree, [True, True, False])
