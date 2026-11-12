"""Frontends: turn a tensor program into a source-level `TensorGraph`.

Supported programs:

* `torch.nn.Module` (PyTorch eager, the reference): `torch.export`, functionalized
  (`run_decompositions({})`), ATen ops kept.
* ONNX `ModelProto` or a path to an `.onnx` file, including ONNX Runtime's optimized graphs with
  fused `com.microsoft` kernels.
* `Compiled` objects from the repository's `compilers.py` (duck-typed by `.compiler`):
  - `eager`: the source model;
  - `triton` / `tilelang`: the source model's graph, where an operation is annotated with the
    kernel only if the kernel path reports that it ran it (`meta["kernel_trace"]`);
  - `onnxruntime_gpu`: the optimized graph ONNX Runtime wrote for that session;
  - `torch_compile`: the ATen graph that AOTAutograd handed to TorchInductor for this model
    (`meta["aot_graphs"]`), after Inductor's decompositions and before its own passes, fusion
    and code generation. Every node is annotated `backend: inductor`.
* A `TensorGraph`, returned unchanged.

`tensorrt` artifacts are not supported yet and raise `Unimplemented`.

What a graph claims, and how the frontend keeps the claim honest:

1. **Shapes are fixed.** Shapes and dtypes come from running the graph once on the example
   inputs, so every certificate is for the example's shapes.
2. **Constants.** Weights are constants. A value computed only from constants and shapes is
   folded into a constant only when its bits do not depend on the implementation: integer and
   boolean results, data movement and selection, casts, and a single correctly rounded IEEE
   operation (`+ - * / sqrt`) that a kernel runs on its own. Anything else (`exp`, `matmul`, ...) stays a node on constant
   inputs, because the program's own kernel computes it and may round differently. Random
   operations and uninitialized memory are never constants.
3. **Determinism.** The graph is executed twice. A folded constant that differs between the two
   runs is an error; other differences are recorded in `meta["replay"]`.
4. **No hidden state.** A program that mutates its inputs or buffers (for example batch norm in
   training mode) is rejected.
5. **Fidelity.** The outputs of the extracted graph are compared with the outputs of the real
   program on the example inputs (`meta["fidelity"]`). A different number of outputs, different
   shapes or dtypes, or a deviation far above rounding level is an error. This is a sanity check
   of the extraction, not part of any bound.
6. **Implementation labels.** `node.meta["impl"]["backend"]` names what runs the node. Later
   stages must have a floating-point model for a backend before they bound it
   (`absint.MODELED_BACKENDS`).
"""
from __future__ import annotations

import operator
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence

import numpy as np

from .graph import GraphError, Node, TensorGraph, Value
from .stubs import Frontend, Unimplemented

# ATen op -> the handler name of compilers.GemmMode that can run it on a Triton/TileLang kernel.
KERNEL_HANDLERS = {"aten.linear": "linear", "aten.matmul": "matmul", "aten.mm": "mm",
                   "aten.bmm": "bmm", "aten.addmm": "addmm", "aten.conv1d": "conv1d",
                   "aten.conv2d": "conv2d"}
KERNEL_OPERANDS = {"addmm": 3}          # tensor operands compared with the trace (default 2)

# Fallback list; the authoritative test is the `nondeterministic_seeded` tag of the operator.
RANDOM_ATEN = {"aten.rand", "aten.randn", "aten.randint", "aten.randperm", "aten.bernoulli",
               "aten.normal", "aten.multinomial", "aten.poisson", "aten.native_dropout",
               "aten.uniform", "aten.rand_like", "aten.randn_like", "aten.randint_like",
               "aten.rrelu", "aten.exponential", "aten.geometric", "aten.cauchy",
               "aten.log_normal"}
DROPOUT_ATEN = {"aten.dropout", "aten.feature_dropout", "aten.alpha_dropout",
                "aten.feature_alpha_dropout"}
UNINITIALIZED_ATEN = {"aten.empty", "aten.empty_like", "aten.new_empty", "aten.empty_permuted",
                      "aten.empty_strided", "aten.new_empty_strided"}
# Results that depend on shapes only.
SHAPE_QUERY_ATEN = {"aten.sym_size", "aten.size", "aten.sym_numel", "aten.numel", "aten.dim",
                    "aten.sym_stride", "aten.sym_storage_offset"}
# Data movement, selection, comparison and creation: the result has the same bits in every
# implementation.
EXACT_ATEN = {"aten." + n for n in """
    view _unsafe_view reshape flatten unflatten squeeze unsqueeze view_as expand expand_as
    broadcast_to permute transpose t slice select narrow split split_with_sizes chunk unbind cat
    concat stack clone contiguous alias detach lift_fresh_copy lift_fresh index_select embedding
    gather index where masked_fill tril triu zeros ones full zeros_like ones_like full_like
    new_zeros new_ones new_full scalar_tensor neg abs relu maximum minimum amax amin clamp clip
    sign floor ceil round trunc constant_pad_nd flip roll repeat tile eye _to_copy to type_as
    eq ne lt le gt ge logical_and logical_or logical_not logical_xor bitwise_and bitwise_or
    bitwise_not bitwise_xor isnan isinf isfinite argmax argmin any all
    """.split()}
# One correctly rounded IEEE operation, run on its own by an eager kernel.
IEEE_ATEN = {"aten.add", "aten.sub", "aten.rsub", "aten.mul", "aten.div", "aten.sqrt"}

RANDOM_ONNX = {"RandomNormal", "RandomNormalLike", "RandomUniform", "RandomUniformLike",
               "Multinomial", "Bernoulli", "Dropout"}
SHAPE_ONLY_ONNX = {"Shape", "Size"}
ALIAS_ONNX = {"MemcpyFromHost", "MemcpyToHost", "Identity"}
EXACT_ONNX = {"Shape", "Size", "Gather", "GatherElements", "GatherND", "Concat", "Reshape",
              "Flatten", "Squeeze", "Unsqueeze", "Transpose", "Expand", "Slice", "Split", "Tile",
              "Pad", "Cast", "Where", "Equal", "Less", "LessOrEqual", "Greater",
              "GreaterOrEqual", "Not", "And", "Or", "Xor", "Neg", "Abs", "Relu", "Max", "Min",
              "Clip", "ConstantOfShape", "Trilu", "OneHot", "NonZero", "ArgMax", "ArgMin",
              "IsNaN", "IsInf", "EyeLike", "Sign", "Floor", "Ceil", "Round"}
IEEE_ONNX = {"Add", "Sub", "Mul", "Div", "Sqrt"}

FIDELITY_RTOL = 1e-3


class ExtractionError(GraphError):
    pass


def aten_name(target) -> str:
    """'aten.relu.default' -> 'aten.relu'."""
    s = str(target)
    parts = s.split(".")
    return ".".join(parts[:2]) if s.startswith("aten.") else s


def _is_float(dtype: str) -> bool:
    return np.dtype(dtype).kind in "fc"


# --------------------------------------------------------------------------------------------
# Fidelity: the extracted graph against the real program
# --------------------------------------------------------------------------------------------

def _to_np(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu()
        import torch
        if x.dtype == torch.bfloat16:
            x = x.float()
        return x.numpy()
    return np.asarray(x)


def tensor_leaves(out) -> list:
    """Tensor leaves of a program output (tensor, tuple, dict, HF ModelOutput), in order."""
    import torch
    from torch.utils import _pytree as pytree
    return [x for x in pytree.tree_leaves(out) if isinstance(x, torch.Tensor)]


def check_fidelity(g: TensorGraph, reference: Sequence, actual: Sequence, what: str,
                   rtol: float = FIDELITY_RTOL) -> dict:
    """Compare the outputs of the extracted graph (`reference`) with the program's (`actual`).

    Raises `ExtractionError` when the graph cannot be the program's graph: another number of
    outputs, other shapes or dtypes, non-finite values in other places, or a deviation above
    `rtol * (1 + max |reference|)`. Rounding-level differences are expected and recorded.
    """
    if len(reference) != len(actual):
        raise ExtractionError(f"{what}: the graph has {len(reference)} tensor outputs, the "
                              f"program returned {len(actual)}")
    worst, bitwise = 0.0, True
    for k, (r, a) in enumerate(zip(reference, actual)):
        r, a = _to_np(r), _to_np(a)
        if r.shape != a.shape or r.dtype != a.dtype:
            raise ExtractionError(f"{what}: output {k} is {r.dtype}{list(r.shape)} in the graph "
                                  f"and {a.dtype}{list(a.shape)} in the program")
        if r.dtype.kind not in "fc":
            if not np.array_equal(r, a):
                raise ExtractionError(f"{what}: integer or boolean output {k} differs")
            continue
        fin = np.isfinite(r) & np.isfinite(a)
        if not np.array_equal(r[~fin], a[~fin], equal_nan=True):
            raise ExtractionError(f"{what}: output {k} has different non-finite values")
        if fin.any():
            d = float(np.abs(r[fin].astype(np.float64) - a[fin].astype(np.float64)).max())
            scale = float(np.abs(r[fin]).max())
            if d > rtol * (1.0 + scale):
                raise ExtractionError(
                    f"{what}: output {k} of the extracted graph differs from the program by "
                    f"{d:.3g} (values up to {scale:.3g}); the graph is not this program's graph")
            worst = max(worst, d)
        bitwise &= bool(np.array_equal(r, a, equal_nan=True))
    fid = {"against": what, "max_abs_diff": worst, "bitwise_equal": bitwise,
           "outputs": len(reference)}
    g.meta["fidelity"] = fid
    return fid


# --------------------------------------------------------------------------------------------
# FX graphs of ATen operations (torch.export and AOTAutograd)
# --------------------------------------------------------------------------------------------

class _TraceableHFMasks:
    """transformers builds attention masks with vmap on torch >= 2.6, which export cannot trace.
    Its pre-2.6 builder makes the same mask with plain tensor ops. The fidelity check then
    compares the exported graph with the unpatched program. Causal masks (GPT-2) use vmap in
    both builders and still fail to export."""

    def __enter__(self):
        try:
            import transformers.masking_utils as mu
            self.mu, self.saved = mu, mu.sdpa_mask
            mu.sdpa_mask = mu.sdpa_mask_older_torch
        except (ImportError, AttributeError):
            self.mu = None
        return self

    def __exit__(self, *exc):
        if self.mu is not None:
            self.mu.sdpa_mask = self.saved


def _np(t):
    import torch
    t = t.detach().cpu()
    if t.dtype == torch.bfloat16:
        raise ExtractionError("bfloat16 constants are not supported")
    return t.numpy().copy()


def _np_dtype(torch_dtype) -> str:
    import torch
    return str(torch.empty((), dtype=torch_dtype).numpy().dtype)


def _literal(x):
    """Non-tensor op argument -> plain Python value."""
    import torch
    if isinstance(x, torch.dtype):
        return _np_dtype(x)
    if isinstance(x, (torch.device, torch.layout, torch.memory_format)):
        return str(x)
    if isinstance(x, (torch.SymInt, torch.SymFloat, torch.SymBool)):
        return x.node.hint if hasattr(x, "node") else x
    if isinstance(x, (list, tuple)):
        return [_literal(y) for y in x]
    return x


def _same(a, b) -> bool:
    """Bitwise-equal results of two runs (NaN equals NaN)."""
    import torch
    if isinstance(a, torch.Tensor):
        if not isinstance(b, torch.Tensor) or a.shape != b.shape or a.dtype != b.dtype:
            return False
        if a.is_floating_point() or a.is_complex():
            return bool(((a == b) | ((a != a) & (b != b))).all())
        return bool(torch.equal(a, b))
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and \
            all(_same(x, y) for x, y in zip(a, b))
    if isinstance(b, torch.Tensor):
        return False
    try:
        return a is b or bool(a == b)
    except Exception:
        return False


def _is_random(target, name: str, attrs: dict) -> bool:
    import torch
    if name in DROPOUT_ATEN:
        return attrs.get("train") is not False
    tag = getattr(torch.Tag, "nondeterministic_seeded", None)
    try:
        if tag is not None and tag in getattr(target, "tags", ()):
            return True
    except Exception:
        pass
    return name in RANDOM_ATEN


def _fold_is_exact(name: str, attrs: dict, in_dtypes, out_dtypes, separate: bool) -> bool:
    """A value computed from constants has implementation-independent bits (module docstring,
    item 2). `separate`: the operation is run on its own by an eager kernel, so it cannot be
    contracted with a neighbour (generated kernels may turn a*b+c into an FMA)."""
    if name in UNINITIALIZED_ATEN:
        return False
    if not any(_is_float(d) for d in list(in_dtypes) + list(out_dtypes)):
        return True                     # integer and boolean arithmetic
    if name in EXACT_ATEN:
        return True
    if name in IEEE_ATEN:               # x + alpha*y is two operations and may be contracted
        return separate and attrs.get("alpha", 1) in (None, 1)
    if name == "aten.arange":           # integer-valued start/end/step: exact in any float type
        return all(isinstance(attrs.get(k), int) or attrs.get(k) is None
                   for k in ("start", "end", "step"))
    return False


@dataclass
class Binding:
    """What a placeholder of an FX graph stands for."""
    kind: str                    # input | const | literal
    value: Any                   # the tensor (or Python value) fed to the graph
    index: Optional[int] = None  # input: position among the program's inputs
    name: str = ""


def _eager_impl(fx_node, name, operands):
    return {"backend": "eager"}


def fx_to_graph(gm, bindings: Sequence[Binding], g: TensorGraph,
                node_impl: Callable = _eager_impl, keep_outputs=None) -> list:
    """Fill `g` from an FX graph of ATen operations. Returns the graph's output tensors as
    computed by PyTorch's own kernels on the bound inputs.

    `node_impl(fx_node, aten_name, operand_values)` returns `meta["impl"]` of a node. It is
    called once per tensor-producing operation, in graph order. `keep_outputs` selects the
    program's outputs among the leaves of the output node (default: all).
    """
    import torch
    import torch.fx
    from torch.utils import _pytree as pytree
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    if len(placeholders) != len(bindings):
        raise ExtractionError(f"graph has {len(placeholders)} placeholders, "
                              f"{len(bindings)} are bound")
    env, lits, static, dynamic = {}, {}, set(), set()
    for ph, b in zip(placeholders, bindings):
        vid = f"v:{ph.name}"
        if b.kind == "input":
            if any(g.values[i].input_index == b.index for i in g.inputs):
                raise ExtractionError(f"program input {b.index} is bound to two placeholders")
            g.add_value(Value(vid, tuple(b.value.shape), _np_dtype(b.value.dtype), "input",
                              input_index=b.index, name=ph.name))
            env[ph.name] = vid
        elif b.kind == "const":
            g.add_const(_np(b.value), name=b.name or ph.name, vid=vid)
            static.add(vid)
            env[ph.name] = vid
        elif b.kind == "literal":
            lits[ph.name] = b.value
        else:
            raise ExtractionError(f"placeholder {ph.name}: {b.kind} {b.name}")
    feeds = [b.value for b in bindings]

    observed, unstable = {}, []

    class Recorder(torch.fx.Interpreter):
        def run_node(self, n):
            r = super().run_node(n)
            observed[n.name] = r
            return r

    class Replayer(torch.fx.Interpreter):
        def run_node(self, n):
            r = super().run_node(n)
            if n.op == "call_function" and not _same(observed[n.name], r):
                unstable.append(n.name)
            return r

    with torch.no_grad():
        Recorder(gm).run(*feeds)
        Replayer(gm).run(*feeds)
    unstable_set = set(unstable)
    g.meta["replay"] = {"runs": 2, "deterministic": not unstable,
                        "first_unstable": unstable[0] if unstable else None}

    def as_input(a, ins):
        if isinstance(a, torch.fx.Node):
            if a.name in lits:
                return _literal(lits[a.name])
            if a.name in dynamic:
                raise ExtractionError(f"{a.name}: a Python scalar that depends on the input "
                                      "data is used as an operator argument")
            v = env.get(a.name)
            if not isinstance(v, str):
                raise ExtractionError(f"argument {a.name} is not a single tensor")
            ins.append(v)
            return {"$in": len(ins) - 1}
        if isinstance(a, (list, tuple)) and any(isinstance(y, torch.fx.Node) for y in a):
            return [as_input(y, ins) for y in a]
        return _literal(a)

    stats = {"folded": 0, "kept_on_constants": []}
    reference = []
    for n in gm.graph.nodes:
        if n.op == "placeholder":
            continue
        if n.op == "get_attr":
            t = gm
            for part in str(n.target).split("."):
                t = getattr(t, part)
            if not isinstance(t, torch.Tensor):
                raise ExtractionError(f"attribute {n.target} is not a tensor")
            vid = f"v:{n.name}"
            g.add_const(_np(t), name=str(n.target), vid=vid)
            static.add(vid)
            env[n.name] = vid
            continue
        if n.op == "output":
            leaves = pytree.tree_leaves(n.args)
            for i in (range(len(leaves)) if keep_outputs is None else keep_outputs):
                o = leaves[i]
                if not isinstance(o, torch.fx.Node) or not isinstance(env.get(o.name), str):
                    raise ExtractionError("only tensor outputs are supported")
                g.outputs.append(env[o.name])
                reference.append(observed[o.name])
            continue
        if n.op != "call_function":
            raise ExtractionError(f"unsupported FX node {n.op} {n.target}")
        r = observed[n.name]
        if n.target is operator.getitem:
            src, idx = n.args[0], n.args[1]
            base = env.get(src.name)
            if isinstance(base, list) and base[idx] is not None:
                env[n.name] = base[idx]
            elif src.name in lits:
                lits[n.name] = lits[src.name][idx]
            else:
                dynamic.add(n.name)
            continue
        results = list(r) if isinstance(r, (list, tuple)) else [r]
        name_ = aten_name(n.target)
        if not any(isinstance(x, torch.Tensor) for x in results):
            if r is None:
                continue              # assertions
            deps = n.all_input_nodes
            if name_ in SHAPE_QUERY_ATEN or all(
                    d.name in lits or (isinstance(env.get(d.name), str)
                                       and env[d.name] in static) for d in deps):
                lits[n.name] = r      # depends on shapes and constants only
            else:
                dynamic.add(n.name)
            continue
        op = str(n.target)
        schema = getattr(n.target, "_schema", None)
        ins, attrs = [], {}
        if schema is not None:
            for i, a in enumerate(schema.arguments):
                if i < len(n.args):
                    val = n.args[i]
                elif a.name in n.kwargs:
                    val = n.kwargs[a.name]
                elif a.has_default_value():
                    val = a.default_value
                else:
                    val = None
                attrs[a.name] = as_input(val, ins)
        else:
            attrs["args"] = [as_input(a, ins) for a in n.args]
            attrs.update({k: as_input(v, ins) for k, v in n.kwargs.items()})
        impl = node_impl(n, name_, [g.values[i] for i in ins])
        out_dtypes = [_np_dtype(x.dtype) for x in results if isinstance(x, torch.Tensor)]
        all_static = all(i in static for i in ins)
        fold = (all_static and not _is_random(n.target, name_, attrs)
                and _fold_is_exact(name_, attrs, [g.values[i].dtype for i in ins], out_dtypes,
                                   impl.get("backend") == "eager"))
        if fold and n.name in unstable_set:
            raise ExtractionError(f"{op} ({n.name}) depends on constants only but returned "
                                  "different values in two runs")
        if all_static and not fold:
            stats["kept_on_constants"].append(name_)
        out_ids = []
        for k, x in enumerate(results):
            if not isinstance(x, torch.Tensor):
                out_ids.append(None)
                continue
            vid = f"v:{n.name}" if len(results) == 1 else f"v:{n.name}.{k}"
            if fold:
                g.add_const(_np(x), name=n.name, vid=vid)
                static.add(vid)
            else:
                g.add_value(Value(vid, tuple(x.shape), _np_dtype(x.dtype), name=n.name))
            out_ids.append(vid)
        env[n.name] = out_ids[0] if len(results) == 1 and not isinstance(r, (list, tuple)) \
            else out_ids
        if fold:
            stats["folded"] += 1
            continue
        node = Node(f"n:{n.name}", op, ins, [o for o in out_ids if o is not None], attrs,
                    source=op)
        node.meta["output_slots"] = [k for k, o in enumerate(out_ids) if o is not None]
        node.meta["impl"] = impl
        g.add_node(node)
    g.meta["constants"] = stats
    return reference


def _complete_inputs(g: TensorGraph, inputs: Sequence):
    """Every program input is a graph input, in the program's order, even when unused."""
    have = {g.values[i].input_index: i for i in g.inputs}
    for k, x in enumerate(inputs):
        if k not in have:
            vid = g.fresh("v:unused_input")
            g.values[vid] = Value(vid, tuple(x.shape), _np_dtype(x.dtype), "input",
                                  input_index=k, name=f"unused_input_{k}")
            have[k] = vid
    extra = sorted(set(have) - set(range(len(inputs))))
    if extra:
        raise ExtractionError(f"graph inputs {extra} are not program inputs")
    g.inputs = [have[k] for k in range(len(inputs))]


def _model_device(model):
    import torch
    for t in list(model.parameters()) + list(model.buffers()):
        return t.device
    return torch.device("cpu")


class TorchExportFrontend(Frontend):
    """A `torch.nn.Module` run by PyTorch eager, or by a path that swaps some of its operations
    for other kernels (`node_impl` says which)."""

    def extract(self, model, path_config=None, example_inputs: Sequence = (), impl=None,
                name: str = "", node_impl: Optional[Callable] = None, program=None,
                check: bool = True):
        import torch
        dev = _model_device(model)
        inputs = tuple(x.to(dev) for x in example_inputs)
        try:
            with torch.no_grad(), _TraceableHFMasks():
                ep = torch.export.export(model, inputs, strict=False)
                ep = ep.run_decompositions({})
        except Exception as e:
            raise ExtractionError(f"torch.export failed: {type(e).__name__}: {e}") from e

        gm = ep.graph_module
        g = TensorGraph(name or type(model).__name__, "torch.export")
        g.meta["impl"] = impl or {"backend": "eager"}
        if node_impl is None:
            if impl and impl.get("backend") != "eager":
                # Label by operator name only: for callers that build graphs without a real
                # kernel path. `_extract_compiled` passes the path's own trace instead.
                def node_impl(n, name_, operands):
                    return impl if name_ in KERNEL_HANDLERS else {"backend": "eager"}
            else:
                node_impl = _eager_impl

        sig = ep.graph_signature
        keep = []
        for i, spec in enumerate(sig.output_specs):
            if spec.kind.name != "USER_OUTPUT":
                raise ExtractionError(
                    f"the program mutates state ({spec.kind.name} of {spec.target}); a model in "
                    "training mode (batch norm statistics) is not a function of its inputs")
            keep.append(i)
        state, consts = ep.state_dict, ep.constants
        bindings, user_i = [], 0
        for spec in sig.input_specs:
            kind = spec.kind.name
            if kind == "USER_INPUT":
                if type(spec.arg).__name__ == "ConstantArgument":
                    bindings.append(Binding("literal", spec.arg.value))
                    continue
                if user_i >= len(inputs):
                    raise ExtractionError("more tensor inputs in the graph than example inputs")
                bindings.append(Binding("input", inputs[user_i], user_i))
                user_i += 1
            elif kind in ("PARAMETER", "BUFFER", "CONSTANT_TENSOR"):
                t = state[spec.target] if spec.target in state else consts[spec.target]
                bindings.append(Binding("const", t, name=spec.target))
            else:
                raise ExtractionError(f"unsupported export input kind {kind}")
        if user_i != len(inputs):
            raise ExtractionError(f"graph has {user_i} tensor inputs, got {len(inputs)} "
                                  "example inputs")
        reference = fx_to_graph(gm, bindings, g, node_impl, keep)
        _complete_inputs(g, inputs)
        g.validate()
        if check:
            with torch.no_grad():
                actual = (program or model)(*inputs)
            check_fidelity(g, reference, tensor_leaves(actual),
                           "exported graph vs the program")
        return g


class KernelTraceLabels:
    """Labels the operations of a source graph that a Triton/TileLang path really ran on its
    kernel, from the path's own call trace on the example inputs.

    `calls` is the list `compilers.GemmMode` logs during one run: one record per intercepted
    Python call, in program order, with `op` (handler name), `handled` (False: the call fell
    back to PyTorch) and the operand `shapes` and `dtypes`. `torch.export` traces the same
    Python execution, so the graph's GEMM-shaped nodes appear in the same order. A node is
    labelled with the kernel only when its record says `handled`; a node without a record makes
    the extraction fail.
    """

    def __init__(self, calls, impl):
        self.calls, self.impl, self.pos = list(calls), impl, 0
        self.stats = {"calls": len(self.calls), "kernel": 0, "fallback": 0}

    def __call__(self, fx_node, name, operands):
        handler = KERNEL_HANDLERS.get(name)
        if handler is None:
            return {"backend": "eager"}
        k = KERNEL_OPERANDS.get(handler, 2)
        shapes = [tuple(v.shape) for v in operands[:k]]
        dtypes = [v.dtype for v in operands[:k]]
        for j in range(self.pos, len(self.calls)):
            c = self.calls[j]
            if c["op"] == handler and [tuple(s) for s in c["shapes"][:k]] == shapes and \
                    list(c["dtypes"][:k]) == dtypes:
                self.pos = j + 1
                self.stats["kernel" if c["handled"] else "fallback"] += 1
                return self.impl if c["handled"] else {"backend": "eager"}
        raise ExtractionError(
            f"{name} ({fx_node.name}, operands {shapes}) has no matching call in the kernel "
            "path's trace: the graph and the kernel path disagree on what ran")


# --------------------------------------------------------------------------------------------
# ONNX
# --------------------------------------------------------------------------------------------

def _onnx_attr(a):
    import onnx
    from onnx import numpy_helper
    v = onnx.helper.get_attribute_value(a)
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, onnx.TensorProto):
        return numpy_helper.to_array(v)
    if isinstance(v, (list, tuple)):
        return [x.decode() if isinstance(x, bytes) else x for x in v]
    return v


def _onnx_constant(node):
    """Value of an ONNX `Constant` node with the dtype the specification gives it."""
    import onnx
    from onnx import numpy_helper
    for a in node.attribute:
        v = onnx.helper.get_attribute_value(a)
        if a.name == "value":
            return numpy_helper.to_array(v)
        if a.name == "value_float":
            return np.asarray(v, dtype=np.float32)
        if a.name == "value_floats":
            return np.asarray(list(v), dtype=np.float32)
        if a.name == "value_int":
            return np.asarray(v, dtype=np.int64)
        if a.name == "value_ints":
            return np.asarray(list(v), dtype=np.int64)
    raise ExtractionError(f"Constant node {node.name}: unsupported value attribute")


def _onnx_probe(model, feeds, providers):
    """Run the model once with every node output exposed; return name -> array."""
    import onnx
    import onnxruntime as ort
    m = onnx.ModelProto()
    m.CopyFrom(model)
    have = {o.name for o in m.graph.output}
    for n in m.graph.node:
        for o in n.output:
            if o and o not in have:
                m.graph.output.append(onnx.ValueInfoProto(name=o))
                have.add(o)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "probe.onnx")
        onnx.save_model(m, path, save_as_external_data=True, all_tensors_to_one_file=True,
                        location="weights.bin", size_threshold=1024)
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        so.log_severity_level = 3
        sess = ort.InferenceSession(path, so, providers=providers)
        names = [o.name for o in sess.get_outputs()]
        return dict(zip(names, sess.run(names, feeds)))


class OnnxFrontend(Frontend):
    def extract(self, model, path_config=None, example_inputs: Sequence = (), name: str = "",
                program=None, check: bool = True):
        import onnx
        from onnx import numpy_helper
        if isinstance(model, (str, os.PathLike)):
            name = name or os.path.basename(str(model))
            model = onnx.load(str(model), load_external_data=True)
        g = TensorGraph(name or model.graph.name, "onnx")
        g.meta["opset"] = {o.domain or "ai.onnx": o.version for o in model.opset_import}
        inits = {t.name: t for t in model.graph.initializer}
        real_inputs = [i for i in model.graph.input if i.name not in inits]
        if len(real_inputs) != len(example_inputs):
            raise ExtractionError(f"ONNX graph has {len(real_inputs)} inputs, "
                                  f"got {len(example_inputs)} example inputs")
        feeds = {}
        for k, (vi, x) in enumerate(zip(real_inputs, example_inputs)):
            arr = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
            dt = onnx.helper.tensor_dtype_to_np_dtype(vi.type.tensor_type.elem_type)
            arr = arr.astype(dt)
            feeds[vi.name] = arr
            g.add_value(Value(vi.name, arr.shape, arr.dtype.name, "input", input_index=k,
                              name=vi.name))
        import onnxruntime as ort
        avail = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail]
        observed = _onnx_probe(model, feeds, providers)
        g.meta["probe_providers"] = providers
        static = set()
        for t in inits.values():
            g.add_const(numpy_helper.to_array(t), name=t.name, vid=t.name)
            static.add(t.name)
        alias = {}
        used_ids = set()
        stats = {"folded": 0, "kept_on_constants": []}

        def dtype_of(vid):
            return g.values[vid].dtype

        for idx, n in enumerate(model.graph.node):
            op = f"{n.domain or 'ai.onnx'}::{n.op_type}"
            std = n.domain in ("", "ai.onnx")
            if any(a.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
                   for a in n.attribute):
                raise ExtractionError(f"{op} ({n.name}): control-flow subgraphs are not "
                                      "supported")
            ins = [alias.get(i, i) if i else None for i in n.input]
            if n.op_type in ALIAS_ONNX and std:
                alias[n.output[0]] = ins[0]
                continue
            if n.op_type == "Constant" and std:
                g.add_const(_onnx_constant(n), name=n.output[0], vid=n.output[0])
                static.add(n.output[0])
                continue
            attrs = {a.name: _onnx_attr(a) for a in n.attribute}
            outs = [o for o in n.output]
            seen = [o for o in outs if o and o in observed]
            all_static = all(i is None or i in static for i in ins)
            shape_only = std and n.op_type in SHAPE_ONLY_ONNX
            exact = std and (
                n.op_type in EXACT_ONNX or n.op_type in IEEE_ONNX
                or not any(_is_float(d) for d in
                           [dtype_of(i) for i in ins if i is not None]
                           + [np.asarray(observed[o]).dtype.name for o in seen]))
            random = n.op_type in RANDOM_ONNX
            if (shape_only or (all_static and exact)) and not random:
                for o in seen:
                    g.add_const(np.asarray(observed[o]), name=o, vid=o)
                    static.add(o)
                stats["folded"] += 1
                continue
            if all_static:
                stats["kept_on_constants"].append(op)
            for o in seen:
                arr = observed[o]
                g.add_value(Value(o, arr.shape, np.asarray(arr).dtype.name, name=o))
            nid = n.name or f"{n.op_type}_{idx}"
            while nid in used_ids:
                nid += "_"
            used_ids.add(nid)
            node = Node(nid, op, ins, seen, attrs, source=op)
            node.meta["output_slots"] = [k for k, o in enumerate(outs) if o and o in observed]
            node.meta["onnx_outputs"] = list(outs)
            g.add_node(node)
        g.outputs = [alias.get(o.name, o.name) for o in model.graph.output]
        g.meta["constants"] = stats
        g.validate()
        if check and program is not None:
            import torch
            with torch.no_grad():
                actual = program(*example_inputs)
            names = [o.name for o in model.graph.output]
            check_fidelity(g, [observed[o] for o in names], tensor_leaves(actual),
                           "ONNX graph vs the program")
        return g


# --------------------------------------------------------------------------------------------
# torch.compile (TorchInductor): the graph AOTAutograd hands to the compiler
# --------------------------------------------------------------------------------------------

def call_key(inputs) -> tuple:
    """Identifies the call a captured graph was compiled for: input shapes and dtypes."""
    return tuple((tuple(x.shape), str(x.dtype)) for x in inputs)


class AotGraphFrontend(Frontend):
    """A model compiled by `torch.compile`.

    `records` are the captures `compilers.compile_torch` made of the compilation itself, one
    per graph handed to Inductor's `inner_compile`:

        graph_module   copy of the functional ATen graph (after AOTAutograd and Inductor's
                       decompositions, before Inductor's post-grad passes and lowering)
        bindings       per placeholder: ("const", tensor) for parameters and buffers,
                       ("input", k) for the program's k-th input, ("literal", value),
                       or ("unknown", tensor)
        user_outputs   indices of the program's outputs among the graph outputs (or None)
        num_mutated    number of mutated inputs (must be 0)
        is_inference   compiled without autograd (must be True)
        graph_breaks   Dynamo graph breaks during the call that compiled it (must be 0)
        call           `call_key` of that call
        seq            number of that call (a model may be recompiled; the last one counts)

    The graph is the program's dataflow at the ATen level. What Inductor then does to it
    (fusion into generated kernels, reduction order, FMA contraction, kernel selection for
    matmul and convolution) is not in the graph: every node is labelled `backend: inductor`,
    and a later stage needs a model of that backend before it may bound the node.
    """

    def extract(self, records, path_config=None, example_inputs: Sequence = (), impl=None,
                name: str = "", program=None, check: bool = True):
        import torch
        impl = impl or {"backend": "inductor"}
        if len(records) != 1:
            raise ExtractionError(
                f"torch.compile produced {len(records)} graphs for these inputs; only a model "
                "compiled into a single graph (no graph breaks) is supported")
        rec = records[0]
        if rec.get("error"):
            raise ExtractionError(f"capture of the compiled graph failed: {rec['error']}")
        if not rec.get("is_inference", True):
            raise ExtractionError("the model was compiled with autograd enabled; compile and "
                                  "run it under torch.no_grad()")
        if rec.get("num_mutated"):
            raise ExtractionError(f"the compiled graph mutates {rec['num_mutated']} of its "
                                  "inputs (training-mode state?)")
        if rec.get("graph_breaks"):
            raise ExtractionError(f"{rec['graph_breaks']} Dynamo graph breaks: part of the "
                                  "model runs outside the compiled graph")
        bindings, dev = [], None
        for b in rec["bindings"]:
            if b[0] == "const":
                dev = dev or b[1].device
        for i, b in enumerate(rec["bindings"]):
            if b[0] == "const":
                bindings.append(Binding("const", b[1], name=f"p{i}"))
            elif b[0] == "input":
                x = example_inputs[b[1]]
                bindings.append(Binding("input", x.to(dev) if dev is not None else x, b[1]))
            elif b[0] == "literal":
                bindings.append(Binding("literal", b[1]))
            else:
                raise ExtractionError(f"placeholder {i} of the compiled graph is neither a "
                                      "parameter nor one of the program's inputs")
        g = TensorGraph(name or "torch_compile", "torch.compile")
        g.meta["impl"] = impl
        g.meta["captured"] = {"stage": "aot_forward_graph", "torch": torch.__version__}
        reference = fx_to_graph(rec["graph_module"], bindings, g, lambda n, name_, ops: impl,
                                rec.get("user_outputs") or None)
        _complete_inputs(g, example_inputs)
        g.validate()
        if check and program is not None:
            with torch.no_grad():
                actual = program(*example_inputs)
            check_fidelity(g, reference, tensor_leaves(actual),
                           "captured ATen graph vs the compiled program")
        return g


# --------------------------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------------------------

def extract_graph(program, example_inputs: Sequence, name: str = "",
                  check: bool = True) -> TensorGraph:
    """Source-level graph of `program` for the given example inputs.

    `check=False` skips the fidelity run of the real program (module docstring, item 5).
    """
    if isinstance(program, TensorGraph):
        return program
    try:
        import onnx
        if isinstance(program, onnx.ModelProto):
            return OnnxFrontend().extract(program, example_inputs=example_inputs, name=name)
    except ImportError:
        pass
    if isinstance(program, (str, os.PathLike)) and str(program).endswith(".onnx"):
        return OnnxFrontend().extract(program, example_inputs=example_inputs, name=name)
    compiler = getattr(program, "compiler", None)
    if compiler is not None and hasattr(program, "run"):
        return _extract_compiled(program, example_inputs, name, check)
    import torch
    if isinstance(program, torch.nn.Module):
        return TorchExportFrontend().extract(program, example_inputs=example_inputs, name=name,
                                             check=check)
    raise TypeError(f"cannot extract a graph from {type(program).__name__}")


def _extract_compiled(c, example_inputs, name, check=True):
    import torch
    meta = getattr(c, "meta", None) or {}
    model = getattr(c, "source_model", None)
    label = name or f"{c.compiler}"
    if c.compiler == "eager":
        return TorchExportFrontend().extract(model, example_inputs=example_inputs, name=label,
                                             program=c, check=check)
    if c.compiler in ("triton", "tilelang"):
        impl = {"backend": c.compiler, **dict(c.options)}
        trace = meta.get("kernel_trace")
        if trace is None:
            raise ExtractionError(f"compiled {c.compiler} program has no kernel_trace: cannot "
                                  "tell which operations ran on the kernel")
        dev = _model_device(model)
        with torch.no_grad():
            _, calls = trace(*[x.to(dev) for x in example_inputs])
        labels = KernelTraceLabels(calls, impl)
        g = TorchExportFrontend().extract(model, example_inputs=example_inputs, impl=impl,
                                          name=label, node_impl=labels, program=c, check=check)
        g.meta["kernel_trace"] = labels.stats
        return g
    if c.compiler == "onnxruntime_gpu":
        path = meta.get("ort_optimized_onnx")
        if not path or not os.path.isfile(path):
            raise ExtractionError("compiled ORT program has no saved optimized graph "
                                  f"({meta.get('ort_optimized_error', 'missing')})")
        dev = _model_device(model) if model is not None else None
        xs = [x.to(dev) for x in example_inputs] if dev is not None else list(example_inputs)
        g = OnnxFrontend().extract(path, example_inputs=xs, name=label, program=c, check=check)
        g.meta["impl"] = {"backend": "onnxruntime_gpu", **dict(c.options)}
        return g
    if c.compiler == "torch_compile":
        records = meta.get("aot_graphs")
        if records is None:
            raise ExtractionError("compiled torch_compile program has no captured graphs")
        dev = _model_device(model) if model is not None else None
        xs = tuple(x.to(dev) for x in example_inputs) if dev is not None \
            else tuple(example_inputs)
        key = call_key(xs)
        c(*xs)                          # compiles (or recompiles) for these shapes if needed
        recs = [r for r in records if r.get("call") == key]
        if not recs:
            raise ExtractionError("torch.compile did not compile a graph for these inputs")
        last = max(r.get("seq", 0) for r in recs)      # the compilation now in use
        recs = [r for r in recs if r.get("seq", 0) == last]
        impl = {"backend": "inductor", **dict(c.options)}
        return AotGraphFrontend().extract(recs, example_inputs=xs, impl=impl, name=label,
                                          program=c, check=check)
    raise Unimplemented(f"graph extraction for {c.compiler} artifacts is not implemented "
                        "(TensorRT engines expose no dataflow graph yet)")
