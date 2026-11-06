"""Graph extraction, matching, and the certifier driver on small CPU programs."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from dlclv.canonical import register_lowering  # noqa: E402
from dlclv.certifier import TensorProgramCertifier  # noqa: E402
from dlclv.frontends import extract_graph  # noqa: E402
from dlclv.interpretation import DeferredInterpreter, ErrorTriple, UnitInterpreter  # noqa: E402
from dlclv.matching import match_graphs  # noqa: E402
from dlclv.outcomes import Inconclusive  # noqa: E402
from dlclv.stubs import Unimplemented  # noqa: E402


class Box:
    """Minimal input domain: a box per input."""

    def __init__(self, *shapes, lo=-1.0, hi=1.0):
        self.shapes, self.lower, self.upper = shapes, [lo] * len(shapes), [hi] * len(shapes)

    def sample(self, n, generator=None):
        return tuple(torch.rand((n, *s), generator=generator) * 2 - 1 for s in self.shapes)


def ex(*shapes, n=2):
    g = torch.Generator().manual_seed(0)
    return tuple(torch.randn((n, *s), generator=g) for s in shapes)


def match(p1, p2, x):
    return match_graphs(extract_graph(p1, x), extract_graph(p2, x))


# ---- fused kernel declared by the user ----

@torch.library.custom_op("dlclv_test::fused_add_relu", mutates_args=())
def fused_add_relu(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.relu(x + y)


@fused_add_relu.register_fake
def _(x, y):
    return torch.empty_like(x)


def _lower_fused_add_relu(L, args):
    return {0: L.unary("relu", L.binary("add", args("x"), args("y")))}


register_lowering("dlclv_test.fused_add_relu", _lower_fused_add_relu)


class AddRelu(nn.Module):
    def forward(self, x, y):
        return torch.relu(x + y) * 2


class FusedAddRelu(nn.Module):
    def forward(self, x, y):
        return fused_add_relu(x, y) * 2


def test_add_relu_matches_fused_kernel():
    m = match(AddRelu(), FusedAddRelu(), ex((4, 8), (4, 8)))
    assert m.complete, m.summary()
    fusion = [u for u in m.units if u.kind == "fusion"]
    assert len(fusion) == 1
    u = fusion[0]
    assert [o.split(".")[1] for o in u.ops_1] == ["add", "relu"]
    assert u.ops_2 == ["dlclv_test.fused_add_relu.default"]
    assert sorted(b.kind for b in u.inputs) == ["input", "input"]
    # x <-> x and y <-> y, not crossed
    assert all(b.source_1.replace("v:", "") == b.source_2.replace("v:", "") for b in u.inputs)
    assert len(u.outputs) == 1 and u.outputs[0].shape == (2, 4, 8)


class SumLeft(nn.Module):
    def forward(self, a, b, c):
        return (a + b) + c


class SumRight(nn.Module):
    def forward(self, a, b, c):
        return a + (b + c)


def test_reassociated_sum_is_one_unit():
    m = match(SumLeft(), SumRight(), ex((5,), (5,), (5,)))
    assert m.complete, m.summary()
    assert len(m.units) == 1 and m.units[0].kind == "many_to_many"
    assert len(m.units[0].source_nodes_1) == 2 and len(m.units[0].source_nodes_2) == 2


class Softmax(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.softmax(x, self.dim)


def test_attribute_difference_is_reported():
    m = match(Softmax(-1), Softmax(1), ex((6, 6)))
    assert not m.complete
    assert m.unmatched and not m.output_pairs[0][1]


def test_constant_difference_is_reported():
    torch.manual_seed(0)
    a, b = nn.Linear(4, 3), nn.Linear(4, 3)
    m = match(a, b, ex((4,)))
    assert not m.complete
    assert any("constants" in r.reason for r in m.unmatched)


def mlp():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4), nn.Softmax(-1)).eval()


def onnx_of(model, x, tmp_path, name="m.onnx"):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    p = str(tmp_path / name)
    torch.onnx.export(model, x, p, input_names=[f"input_{i}" for i in range(len(x))],
                      opset_version=17, dynamo=False)
    return p


def test_torch_vs_onnx_mlp(tmp_path):
    model, x = mlp(), ex((8,))
    m = match(model, onnx_of(model, x, tmp_path), x)
    assert m.complete, m.summary()
    assert m.graph_2.frontend == "onnx"
    assert any("Gemm" in "".join(u.ops_2) for u in m.units)


def test_gelu_expression_matches_gelu(tmp_path):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 8), nn.GELU()).eval()
    x = ex((8,))
    m = match(model, onnx_of(model, x, tmp_path), x)
    assert m.complete, m.summary()
    gelu = [u for u in m.units if "gelu" in u.logical_ops]
    assert gelu and gelu[0].kind == "fission" and gelu[0].notes


class ScaleThenMatmul(nn.Module):
    def forward(self, q, k):
        return torch.matmul(q / 8.0, k.transpose(-1, -2))


class MatmulThenScale(nn.Module):
    def forward(self, q, k):
        return torch.matmul(q, k.transpose(-1, -2)) * 0.125


def test_scalar_moves_out_of_matmul():
    m = match(ScaleThenMatmul(), MatmulThenScale(), ex((3, 4), (3, 4)))
    assert m.complete, m.summary()


def test_inexact_reciprocal_does_not_match():
    class A(nn.Module):
        def forward(self, x):
            return x / 3.0

    class B(nn.Module):
        def forward(self, x):
            return x * (1 / 3.0)

    assert not match(A(), B(), ex((4,))).complete


class ConvBN(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.conv, self.bn = nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4)
        self.bn.running_mean.uniform_(-1, 1)
        self.bn.running_var.uniform_(0.5, 2)
        self.head = nn.Linear(4, 2)

    def forward(self, x):
        y = torch.relu(self.bn(self.conv(x)))
        return self.head(y.mean((2, 3)))


def test_folded_batchnorm_is_an_unmatched_region_and_downstream_still_matches(tmp_path):
    model, x = ConvBN().eval(), ex((3, 8, 8))
    cert = TensorProgramCertifier(model, onnx_of(model, x, tmp_path), Box((3, 8, 8)),
                                  {"example_inputs": x})
    m = cert.match()
    assert not m.complete
    assert len(m.unmatched) == 1
    r = m.unmatched[0]
    assert any("batch_norm" in o for o in r.ops_1) and r.output_cuts
    assert any("relu" in u.logical_ops for u in m.units)        # matching resumed after the cut
    res = cert.certify()
    assert res.status == "inconclusive" and res.output_triples is None
    assert res.inconclusive_reason is Inconclusive.UNSUPPORTED_OPERATOR


class TwoRelu(nn.Module):
    def forward(self, x):
        return torch.relu(x) + torch.relu(x)


class OneRelu(nn.Module):
    def forward(self, x):
        r = torch.relu(x)
        return r + r


def test_common_subexpression_elimination_matches():
    m = match(TwoRelu(), OneRelu(), ex((4,)))
    assert m.complete, m.summary()
    u = next(u for u in m.units if "relu" in u.logical_ops)
    assert len(u.source_nodes_1) == 2 and len(u.source_nodes_2) == 1


class ThreeRelu(nn.Module):
    def forward(self, x):
        return torch.relu(x) + torch.relu(x) + torch.relu(x)


class TwoReluThreeTerms(nn.Module):
    def forward(self, x):
        a, b = torch.relu(x), torch.relu(x)
        return a + b + b


def test_unequal_duplicates_are_ambiguous():
    m = match(ThreeRelu(), TwoReluThreeTerms(), ex((4,)))
    assert any(a.kind == "duplicate_computation" for a in m.ambiguities)
    assert not m.complete
    cert = TensorProgramCertifier(ThreeRelu(), TwoReluThreeTerms(), Box((4,)))
    res = cert.certify()
    assert res.status == "inconclusive" and res.output_triples is None
    assert res.inconclusive_reason is Inconclusive.UNSUPPORTED_OPERATOR


def test_deferred_interpreter_stops_at_abstract_interpretation():
    model = mlp()
    cert = TensorProgramCertifier(model, model, Box((8,)),
                                  {"batch_size": 2, "interpreter": DeferredInterpreter()})
    g1, g2 = cert.extract_graphs()
    assert g1.inputs and g2.inputs
    assert cert.match().complete
    with pytest.raises(NotImplementedError):
        cert.certify()
    with pytest.raises(Unimplemented):
        cert.certify()


def test_default_interpreter_bounds_a_complete_match():
    model = mlp()
    res = TensorProgramCertifier(model, model, Box((8,)), {"batch_size": 2}).certify()
    assert res.status == "bounded" and res.interpreter == "BacksubAnalyzer"
    assert 0 < res.certified_bound < 1e-3


def test_driver_runs_units_in_dependency_order():
    class Recorder(UnitInterpreter):
        def __init__(self):
            self.seen = []

        def transfer(self, ctx, inputs):
            self.seen.append(ctx.unit.id)
            err = max(t.max_error for t in inputs) + 1e-6
            return [ErrorTriple(-10.0, 10.0, err) for _ in ctx.unit.outputs]

    rec = Recorder()
    model = mlp()
    cert = TensorProgramCertifier(model, model, Box((8,)), {"interpreter": rec})
    res = cert.certify()
    assert rec.seen == [u.id for u in cert.match().units]
    assert len(res.output_error_bounds) == 1 and res.output_error_bounds[0] > 0


def test_error_triple_validation():
    with pytest.raises(ValueError):
        ErrorTriple(1.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        ErrorTriple(0.0, 1.0, -1.0)
    with pytest.raises(ValueError):
        ErrorTriple(0.0, float("nan"), 0.0)


def test_graph_records_inputs_outputs_and_dependencies():
    g = extract_graph(mlp(), ex((8,)))
    g.validate()
    assert len(g.inputs) == 1 and len(g.outputs) == 1
    assert g.values[g.inputs[0]].shape == (2, 8)
    for n in g.nodes.values():
        assert all(i in g.values for i in n.inputs if i is not None)
    consts = [v for v in g.values.values() if v.kind == "const"]
    assert any(v.shape == (16, 8) for v in consts) and all(v.data is not None for v in consts)
