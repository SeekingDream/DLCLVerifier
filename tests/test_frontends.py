"""Graph extraction: what a graph may claim about its program (frontends.py, items 1 to 6)."""
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn
F = torch.nn.functional

from dlclv.certifier import TensorProgramCertifier  # noqa: E402
from dlclv.frontends import (ExtractionError, check_fidelity, extract_graph,  # noqa: E402
                             TorchExportFrontend)
from dlclv.graph import TensorGraph  # noqa: E402
from dlclv.outcomes import Inconclusive  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ex(*shapes, n=2):
    g = torch.Generator().manual_seed(0)
    return tuple(torch.randn((n, *s), generator=g) for s in shapes)


def ops(g):
    return [n.op.split(".")[1] for n in g.nodes.values()]


class Box:
    def __init__(self, *shapes):
        self.shapes, self.lower, self.upper = shapes, [-1.0] * len(shapes), [1.0] * len(shapes)

    def sample(self, n, generator=None):
        return tuple(torch.rand((n, *s), generator=generator) * 2 - 1 for s in self.shapes)


# ---- state, randomness, constants ----

def test_training_mode_state_is_rejected():
    model = nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4)).train()
    with pytest.raises(ExtractionError, match="mutates state"):
        extract_graph(model, ex((3, 8, 8)))
    g = extract_graph(model.eval(), ex((3, 8, 8)))
    assert g.meta["fidelity"]["bitwise_equal"] and g.meta["replay"]["deterministic"]


class Noise(nn.Module):
    def forward(self, x):
        return x + torch.rand(4)


def test_random_values_are_never_constants():
    g = extract_graph(Noise(), ex((4,)), check=False)
    assert "rand" in ops(g) and not g.meta["replay"]["deterministic"]
    assert not any(v.kind == "const" for v in g.values.values())
    with pytest.raises(ExtractionError, match="differs from the program"):
        extract_graph(Noise(), ex((4,)))          # the graph cannot reproduce the program


class Folds(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(4))
        self.W = nn.Parameter(torch.randn(4, 4))
        self.A = nn.Parameter(torch.randn(2, 4))
        self.B = nn.Parameter(torch.randn(4, 2))

    def forward(self, x):
        y = x + self.w.reshape(1, 4)              # data movement on a constant: folded
        y = y * (self.w * 2.0)                    # one IEEE multiplication: folded
        y = y * torch.exp(self.w)                 # exp is not correctly rounded: kept
        return F.linear(y, self.W + self.B @ self.A)   # matmul runs on the program's kernel


def test_only_implementation_independent_values_are_folded():
    g = extract_graph(Folds(), ex((4,)))
    kept = ops(g)
    assert "reshape" not in kept and "view" not in kept
    assert kept.count("mul") == 2                 # w * 2.0 is a constant, the two y * ... stay
    assert "exp" in kept and "matmul" in kept
    assert sorted(g.meta["constants"]["kept_on_constants"]) == ["aten.exp", "aten.matmul"]
    exp = next(n for n in g.nodes.values() if "exp" in n.op)
    assert all(g.values[i].kind == "const" for i in exp.inputs)


def test_fidelity_rejects_a_graph_of_another_program():
    model = nn.Linear(4, 3)
    x = ex((4,))
    with pytest.raises(ExtractionError, match="differs from the program"):
        TorchExportFrontend().extract(model, example_inputs=x, program=lambda t: model(t) + 1.0)
    with pytest.raises(ExtractionError, match="tensor outputs"):
        TorchExportFrontend().extract(model, example_inputs=x,
                                      program=lambda t: (model(t), model(t)))
    g = TensorGraph("g")
    with pytest.raises(ExtractionError, match="non-finite"):
        check_fidelity(g, [np.array([1.0, np.inf], np.float32)],
                       [np.array([1.0, 2.0], np.float32)], "t")
    fid = check_fidelity(g, [np.array([1.0, np.nan], np.float32)],
                         [np.array([1.0 + 1e-7, np.nan], np.float32)], "t")
    assert not fid["bitwise_equal"] and 0 < fid["max_abs_diff"] < 1e-6


def test_data_dependent_scalar_argument_is_rejected():
    class M(nn.Module):
        def forward(self, x):
            return x * x.shape[1] + torch.full((x.shape[0], 4), 0.5)   # shapes: fine

    g = extract_graph(M(), ex((4,)))
    assert g.meta["fidelity"]["bitwise_equal"]


# ---- Triton / TileLang paths: labels come from the path's own trace ----

class TwoLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.a, self.b = nn.Linear(8, 16), nn.Linear(16, 4)

    def forward(self, x):
        return self.b(torch.relu(self.a(x)))


class FakeKernelPath:
    compiler, options = "triton", {"split_k": 1}

    def __init__(self, model, calls):
        self.source_model, self.run = model, model
        self.meta = {"kernel_trace": lambda *x: (model(*x), calls)} if calls is not None else {}

    def __call__(self, *x):
        return self.run(*x)


def _call(handled, shapes):
    return {"op": "linear", "handled": handled, "shapes": shapes,
            "dtypes": ["float32"] * len(shapes)}


def test_kernel_label_needs_a_handled_call_in_the_trace():
    model, x = TwoLinear(), ex((8,))
    calls = [_call(True, [(2, 8), (16, 8), (16,)]), _call(False, [(2, 16), (4, 16), (4,)])]
    g = extract_graph(FakeKernelPath(model, calls), x)
    backends = [n.meta["impl"]["backend"] for n in g.nodes.values() if "linear" in n.op]
    assert backends == ["triton", "eager"]        # the fallback ran on PyTorch's kernel
    assert g.meta["kernel_trace"] == {"calls": 2, "kernel": 1, "fallback": 1}
    with pytest.raises(ExtractionError, match="no matching call"):
        extract_graph(FakeKernelPath(model, calls[:1]), x)
    with pytest.raises(ExtractionError, match="kernel_trace"):
        extract_graph(FakeKernelPath(model, None), x)


# ---- ONNX ----

def _onnx_model(nodes, inits=()):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    h, T = onnx.helper, onnx.TensorProto
    graph = h.make_graph(nodes, "g", [h.make_tensor_value_info("x", T.FLOAT, [2, 4])],
                         [h.make_tensor_value_info("y", T.FLOAT, [2, 4])], list(inits))
    m = h.make_model(graph, opset_imports=[h.make_opsetid("", 17)])
    m.ir_version = 8
    return m


def test_onnx_constants_and_random_nodes():
    onnx = pytest.importorskip("onnx")
    h = onnx.helper
    w = onnx.numpy_helper.from_array(np.arange(4, dtype=np.float32), "w")
    m = _onnx_model([
        h.make_node("Constant", [], ["c"], value_float=1.5),
        h.make_node("Exp", ["w"], ["e"]),                     # not exact: stays a node
        h.make_node("Neg", ["w"], ["nw"]),                    # exact: folded
        h.make_node("RandomNormal", [], ["r"], shape=[2, 4], seed=1.0),
        h.make_node("Add", ["x", "c"], ["t0"]),
        h.make_node("Mul", ["t0", "e"], ["t1"]),
        h.make_node("Add", ["t1", "nw"], ["t2"]),
        h.make_node("Add", ["t2", "r"], ["y"]),
    ], [w])
    g = extract_graph(m, ex((4,)))
    assert g.values["c"].kind == "const" and g.values["c"].dtype == "float32"
    assert g.values["nw"].kind == "const"
    assert g.values["e"].kind != "const" and g.values["r"].kind != "const"
    kinds = sorted(n.op.split("::")[1] for n in g.nodes.values())
    assert kinds == ["Add", "Add", "Add", "Exp", "Mul", "RandomNormal"]


# ---- torch.compile ----

def _inductor(model, x):
    sys.path.insert(0, ROOT)
    import compilers
    try:
        run, cap = compilers.compile_inductor(model)
        with torch.no_grad():
            run(*x)
    except Exception as e:                       # no C++ toolchain for Inductor on CPU
        pytest.skip(f"torch.compile is not usable here: {type(e).__name__}: {str(e)[:120]}")
    return compilers.Compiled("torch_compile", {"mode": "default"}, run, source_model=model,
                              meta={"aot_graphs": cap.records})


def test_torch_compile_graph_is_extracted_but_not_bounded():
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4)).eval()
    x = ex((8,))
    c = _inductor(model, x)
    torch._dynamo.reset()                        # forces a recompilation: the last one counts
    g = extract_graph(c, x)
    assert g.frontend == "torch.compile" and len(g.inputs) == 1 and len(g.outputs) == 1
    assert g.values[g.inputs[0]].shape == (2, 8)
    assert all(n.meta["impl"]["backend"] == "inductor" for n in g.nodes.values())
    assert sum(v.kind == "const" for v in g.values.values()) >= 4      # weights and biases
    assert g.meta["fidelity"]["max_abs_diff"] < 1e-5
    res = TensorProgramCertifier(model, c, Box((8,)), {"example_inputs": x}).certify()
    assert res.status == "inconclusive" and res.certified_bound is None
    assert res.inconclusive_reason is Inconclusive.UNSUPPORTED_PATH


def test_torch_compile_graph_break_is_rejected():
    class Branch(nn.Module):
        def forward(self, x):
            y = torch.relu(x)
            if y.sum() > 0:                        # data-dependent control flow: graph break
                return y * 2
            return y - 1

    x = ex((8,))
    c = _inductor(Branch(), x)
    with pytest.raises(ExtractionError, match="graph"):
        extract_graph(c, x)
