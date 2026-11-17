"""Adversarial bits and fail-closed scope; passing tests are not a proof."""
import sys
import pytest
import numpy as np

torch = pytest.importorskip('torch')
from dlclv.bitwise import bitwise_equal
from dlclv.reconstruction import (reconstruct, ReconstructionFrontend,
                                  UnsupportedReconstruction, ExecutionMismatch)
from dlclv.frontends import check_fidelity, _same
from dlclv.graph import TensorGraph



@pytest.fixture(params=['aten', 'pytorch'])
def build(request):
    if request.param == 'pytorch':
        version = tuple(int(v) for v in torch.__version__.split('+')[0].split('.')[:2])
        if sys.version_info >= (3, 12) and version < (2, 4):
            pytest.skip('installed torch export does not support Python 3.12')
        return reconstruct
    def from_aten(model, inputs):
        # Fixture only: these straight-line functions have no branches/state.
        from torch.fx.experimental.proxy_tensor import make_fx
        gm = make_fx(model)(*inputs)
        return reconstruct(gm, inputs, format='aten')
    return from_aten

class Ordered(torch.nn.Module):
    def forward(self, x):
        return (x[0] + x[1]) + x[2]


class Pairwise(torch.nn.Module):
    def forward(self, x):
        return x[0] + (x[1] + x[2])


def test_order_and_partial_accumulators(build):
    x = torch.tensor([2**25, -2**25, 1.])
    a, b = build(Ordered(), (x,)), build(Pairwise(), (x,))
    assert bitwise_equal(a(x), Ordered()(x))
    assert bitwise_equal(b(x), Pairwise()(x))
    assert not bitwise_equal(a(x), b(x))
    assert a.report['status'] == 'conditional'
    assert not a.report['universal_certificate']
    assert a.compare(Ordered(), [(x,), (x.flip(0),)])['bitwise_equal']
    assert not a.report['empirical']['is_proof']
    assert 'aten.add' in a.code and 'sum' not in a.code


class Precision(torch.nn.Module):
    def forward(self, x):
        a = x.to(torch.float16)
        b = a + a
        return b.to(torch.float32)


def test_cast_rounding_and_overflow(build):
    x = torch.tensor([1 + 2**-11, 1 + 3 * 2**-11, 40000., 2**-25])
    p = build(Precision(), (x,))
    assert bitwise_equal(p(x), Precision()(x))
    assert not bitwise_equal(p(x), x + x)
    assert torch.isinf(p(x)[2])
    assert p(x)[0] == 2 and p(x)[1] == 2 + 2**-8


class Separate(torch.nn.Module):
    def forward(self, a, b, c):
        return a * b + c


def test_separate_product_is_not_fma(build):
    a, b, c = (torch.tensor(x) for x in (1 + 2**-23, 1 - 2**-23, -1.))
    p = build(Separate(), (a, b, c))
    assert bitwise_equal(p(a, b, c), torch.tensor(0.))
    # This particular product and sum are exact in binary64; a true binary32
    # FMA returns -2**-46, unlike the separately rounded product.
    fused_oracle = (a.double() * b.double() + c.double()).float()
    assert fused_oracle == -(2**-46)
    assert not bitwise_equal(p(a, b, c), fused_oracle)


def test_bits_signed_zero_nan_payload_bfloat16_and_noncontiguous():
    a = torch.tensor([0, -2147483648, 2143289345], dtype=torch.int32).view(torch.float32)
    b = torch.tensor([-2147483648, 0, 2143289346], dtype=torch.int32).view(torch.float32)
    assert not bitwise_equal(a, b) and not _same(a, b)
    assert bitwise_equal(a, a.clone())
    assert bitwise_equal(a, a.numpy())
    assert not bitwise_equal(a, b.numpy())
    assert not bitwise_equal(a[0], b[0])
    bf = torch.tensor([1., -0.], dtype=torch.bfloat16)
    assert bitwise_equal(bf, bf.clone())
    m = torch.randn(3, 4).t()
    assert bitwise_equal(m, m.contiguous())
    f = check_fidelity(TensorGraph('bits'), [a.numpy()], [b.numpy()], 'bits')
    assert not f['bitwise_equal']
    assert not bitwise_equal(np.array([0.]), np.array([-0.]))


@pytest.mark.parametrize('op', [lambda x: x.sum(), lambda x: x @ x,
                                lambda x: torch.exp(x),
                                lambda x: torch.add(x, x, alpha=2)])
def test_opaque_and_compound_ops_rejected(op, build):
    class M(torch.nn.Module):
        def forward(self, x):
            return op(x)
    with pytest.raises(UnsupportedReconstruction):
        build(M(), (torch.ones(2, 2),))


@pytest.mark.parametrize('fmt', ['onnx', 'tvm', 'tilelang', 'triton', 'tensorrt'])
def test_formats_require_lowered_adapter(fmt):
    with pytest.raises(UnsupportedReconstruction, match='lowered-kernel'):
        reconstruct(object(), (), format=fmt)


def test_compiled_source_graph_is_not_kernel_semantics():
    class Compiled:
        compiler = 'torch_compile'
        source_model = Ordered()
    with pytest.raises(UnsupportedReconstruction, match='generated kernel'):
        reconstruct(Compiled(), (torch.ones(3),))


def test_domain_and_environment_guards(build):
    x = torch.ones(3)
    p = build(Ordered(), (x,))
    with pytest.raises(ExecutionMismatch, match='shape/dtype'):
        p(x.double())
    with pytest.raises(ExecutionMismatch, match='shape/dtype'):
        p(torch.ones(6)[::2])
    with torch.autocast('cpu'):
        with pytest.raises(ExecutionMismatch, match='autocast'):
            p(x)
    old = torch.get_num_threads()
    try:
        torch.set_num_threads(old + 1)
        with pytest.raises(ExecutionMismatch, match='configuration'):
            p(x)
    finally:
        torch.set_num_threads(old)


def test_explicit_aten_adapter_and_rejection():
    g = torch.fx.Graph()
    x = g.placeholder('x')
    y = g.call_function(torch.ops.aten.mul.Tensor, (x, x))
    g.output(y)
    gm = torch.fx.GraphModule({}, g)
    x = torch.tensor([2., -0.])
    p = reconstruct(gm, (x,), format='aten')
    assert bitwise_equal(p(x), gm(x))
    frontend = ReconstructionFrontend()
    frontend.register('lowered_example', lambda a, inputs: reconstruct(a, inputs, format='aten'))
    assert bitwise_equal(frontend.reconstruct(gm, (x,), format='lowered_example')(x), gm(x))


def test_mutation_rejected():
    class M(torch.nn.Module):
        def forward(self, x):
            return x.add_(1)
    with pytest.raises(UnsupportedReconstruction):
        reconstruct(M(), (torch.ones(3),))


class Partials(torch.nn.Module):
    def forward(self, x):
        p0 = x[0] + x[2]
        p1 = x[1] + x[3]
        return p0 + p1


def test_partial_accumulator_tree(build):
    x = torch.tensor([2**25, -2**25, 1., 1.])
    p = build(Partials(), (x,))
    assert bitwise_equal(p(x), Partials()(x))
    sequential = ((x[0] + x[1]) + x[2]) + x[3]
    assert not bitwise_equal(p(x), sequential)


def test_fused_primitive_is_not_expanded():
    g = torch.fx.Graph()
    a, b, c = [g.placeholder(n) for n in ('a', 'b', 'c')]
    y = g.call_function(torch.ops.aten.addcmul.default, (c, a, b))
    g.output(y)
    with pytest.raises(UnsupportedReconstruction, match='FMA'):
        reconstruct(torch.fx.GraphModule({}, g), (torch.ones(1),) * 3, format='aten')


def test_constants_snapshot_and_nested_outputs():
    g = torch.fx.Graph()
    x = g.placeholder('x')
    w = g.get_attr('weight')
    y = g.call_function(torch.ops.aten.mul.Tensor, (x, w))
    g.output({'result': y, 'input': x})
    gm = torch.fx.GraphModule({'weight': torch.tensor([2.])}, g)
    x = torch.tensor([3.])
    p = reconstruct(gm, (x,), format='aten')
    gm.weight.fill_(4.)
    assert bitwise_equal(p(x)['result'], torch.tensor([6.]))
    assert not p.compare(gm, [(x,)])['bitwise_equal']
    assert p.compare(gm, [])['bitwise_equal'] is None


def _inplace_graph(alias):
    g = torch.fx.Graph()
    x, y = g.placeholder('x'), g.placeholder('y')
    a = g.call_function(torch.ops.aten.mul.Tensor, (x, y))
    src = g.call_function(torch.ops.aten.view.default, (a, [-1])) if alias else a
    out = g.call_function(torch.ops.aten.add_.Tensor, (src, y))
    g.output((out, a) if alias else out)
    return torch.fx.GraphModule({}, g)


def test_inplace_on_a_fresh_intermediate_runs_out_of_place():
    x, y = torch.tensor([2**25, 3.]), torch.tensor([1., -0.])
    gm = _inplace_graph(False)
    p = reconstruct(gm, (x, y), format='aten')
    assert bitwise_equal(p(x, y), x * y + y) and bitwise_equal(p(x, y), gm(x, y))
    # A mutation that an alias could observe is not rewritten, it is rejected.
    with pytest.raises(UnsupportedReconstruction, match='mutates a value'):
        reconstruct(_inplace_graph(True), (x, y), format='aten')
    g = torch.fx.Graph()
    x_ = g.placeholder('x')
    g.output(g.call_function(torch.ops.aten.add_.Tensor, (x_, x_)))
    with pytest.raises(UnsupportedReconstruction, match='mutates a value'):
        reconstruct(torch.fx.GraphModule({}, g), (x,), format='aten')
