"""Frontends: turn a tensor program into a source-level `TensorGraph`.

Supported programs:

* `torch.nn.Module`: `torch.export`, functionalized (`run_decompositions({})`), ATen ops kept.
* ONNX `ModelProto` or a path to an `.onnx` file, including ONNX Runtime's optimized graphs with
  fused `com.microsoft` kernels.
* `Compiled` objects from the repository's `compilers.py` (duck-typed by `.compiler`):
  `eager` -> the source model; `triton` / `tilelang` -> the source model's graph with every
  GEMM-shaped op annotated with the kernel and its options; `onnxruntime_gpu` -> the optimized
  graph ONNX Runtime wrote for that session.
* A `TensorGraph`, returned unchanged.

`torch_compile` (Inductor) and `tensorrt` artifacts are not supported yet and raise
`Unimplemented`.

Shapes and dtypes come from running the program once on the example inputs. A value computed
only from constants and shapes (masks from `arange`, ONNX `Shape`/`Gather`/`Concat` chains) does
not depend on the input data, so it becomes a constant with the observed data. Shapes are fixed,
so every certificate is for the example's shapes.
"""
from __future__ import annotations

import operator
import os
import tempfile
from typing import Sequence

import numpy as np

from .graph import GraphError, Node, TensorGraph, Value
from .stubs import Frontend, Unimplemented

# ATen ops the triton / tilelang paths of compilers.py run on their own GEMM kernels.
KERNEL_SWAPPED_ATEN = {"aten.linear", "aten.matmul", "aten.mm", "aten.bmm", "aten.addmm",
                       "aten.conv1d", "aten.conv2d", "aten.convolution"}
RANDOM_ATEN = {"aten.rand", "aten.randn", "aten.randint", "aten.bernoulli", "aten.normal",
               "aten.native_dropout", "aten.uniform", "aten.rand_like", "aten.randn_like"}


class ExtractionError(GraphError):
    pass


def aten_name(target) -> str:
    """'aten.relu.default' -> 'aten.relu'."""
    s = str(target)
    parts = s.split(".")
    return ".".join(parts[:2]) if s.startswith("aten.") else s


# --------------------------------------------------------------------------------------------
# torch.export
# --------------------------------------------------------------------------------------------

class _TraceableHFMasks:
    """transformers builds attention masks with vmap on torch >= 2.6, which export cannot trace.
    Its pre-2.6 builder makes the same mask with plain tensor ops."""

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
    if isinstance(x, (list, tuple)):
        return [_literal(y) for y in x]
    return x


class TorchExportFrontend(Frontend):
    def extract(self, model, path_config=None, example_inputs: Sequence = (), impl=None,
                name: str = ""):
        import torch
        import torch.fx
        params = list(model.parameters()) + list(model.buffers())
        dev = params[0].device if params else torch.device("cpu")
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
        env, static = {}, set()
        state, consts = ep.state_dict, ep.constants
        feeds, user_i = [], 0
        placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
        for spec, ph in zip(ep.graph_signature.input_specs, placeholders):
            kind = spec.kind.name
            if kind == "USER_INPUT":
                x = inputs[user_i]
                g.add_value(Value(f"v:{ph.name}", tuple(x.shape), _np_dtype(x.dtype), "input",
                                  input_index=user_i, name=ph.name))
                feeds.append(x)
                user_i += 1
            elif kind in ("PARAMETER", "BUFFER", "CONSTANT_TENSOR"):
                t = state[spec.target] if spec.target in state else consts[spec.target]
                g.add_const(_np(t), name=spec.target, vid=f"v:{ph.name}")
                static.add(f"v:{ph.name}")
                feeds.append(t)
            else:
                raise ExtractionError(f"unsupported export input kind {kind}")
            env[ph.name] = f"v:{ph.name}"

        observed = {}

        class Recorder(torch.fx.Interpreter):
            def run_node(self, n):
                r = super().run_node(n)
                observed[n.name] = r
                return r

        with torch.no_grad():
            Recorder(gm).run(*feeds)

        def as_input(a, ins):
            if isinstance(a, torch.fx.Node):
                v = env.get(a.name)
                if not isinstance(v, str):
                    raise ExtractionError(f"argument {a.name} is not a single tensor")
                ins.append(v)
                return {"$in": len(ins) - 1}
            if isinstance(a, (list, tuple)) and any(isinstance(y, torch.fx.Node) for y in a):
                return [as_input(y, ins) for y in a]
            return _literal(a)

        for n in gm.graph.nodes:
            if n.op == "placeholder":
                continue
            if n.op == "output":
                outs = n.args[0]
                outs = outs if isinstance(outs, (list, tuple)) else [outs]
                for o in outs:
                    if not isinstance(o, torch.fx.Node) or not isinstance(env.get(o.name), str):
                        raise ExtractionError("only tensor outputs are supported")
                    g.outputs.append(env[o.name])
                continue
            if n.op != "call_function":
                raise ExtractionError(f"unsupported FX node {n.op} {n.target}")
            r = observed[n.name]
            if n.target is operator.getitem:
                base = env[n.args[0].name]
                env[n.name] = base[n.args[1]]
                continue
            results = list(r) if isinstance(r, (list, tuple)) else [r]
            if not any(isinstance(x, torch.Tensor) for x in results):
                continue          # asserts and other ops without tensor results
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
            is_static = (all(i in static for i in ins)
                         and aten_name(n.target) not in RANDOM_ATEN)
            out_ids = []
            for k, x in enumerate(results):
                if not isinstance(x, torch.Tensor):
                    out_ids.append(None)
                    continue
                vid = f"v:{n.name}" if len(results) == 1 else f"v:{n.name}.{k}"
                if is_static:
                    g.add_const(_np(x), name=n.name, vid=vid)
                    static.add(vid)
                else:
                    g.add_value(Value(vid, tuple(x.shape), _np_dtype(x.dtype), name=n.name))
                out_ids.append(vid)
            env[n.name] = out_ids[0] if len(results) == 1 and not isinstance(r, (list, tuple)) \
                else out_ids
            if is_static:
                continue
            node = Node(f"n:{n.name}", op, ins, [o for o in out_ids if o is not None], attrs,
                        source=op)
            node.meta["output_slots"] = [k for k, o in enumerate(out_ids) if o is not None]
            name_ = aten_name(n.target)
            node.meta["impl"] = (impl if impl and name_ in KERNEL_SWAPPED_ATEN
                                 else {"backend": "eager"})
            g.add_node(node)
        g.validate()
        return g


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


SHAPE_ONLY_ONNX = {"Shape", "Size"}
ALIAS_ONNX = {"MemcpyFromHost", "MemcpyToHost", "Identity"}


class OnnxFrontend(Frontend):
    def extract(self, model, path_config=None, example_inputs: Sequence = (), name: str = ""):
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
        static = set()
        for t in inits.values():
            g.add_const(numpy_helper.to_array(t), name=t.name, vid=t.name)
            static.add(t.name)
        alias = {}
        used_ids = set()
        for idx, n in enumerate(model.graph.node):
            op = f"{n.domain or 'ai.onnx'}::{n.op_type}"
            ins = [alias.get(i, i) if i else None for i in n.input]
            attrs = {a.name: _onnx_attr(a) for a in n.attribute}
            if n.op_type in ALIAS_ONNX and n.domain in ("", "ai.onnx"):
                alias[n.output[0]] = ins[0]
                if ins[0] in static:
                    static.add(n.output[0])
                continue
            if n.op_type == "Constant":
                val = next(iter(attrs.values()))
                g.add_const(np.asarray(val), name=n.output[0], vid=n.output[0])
                static.add(n.output[0])
                continue
            is_static = n.op_type in SHAPE_ONLY_ONNX or all(i is None or i in static for i in ins)
            outs = [o for o in n.output]
            if is_static:
                for o in outs:
                    if o and o in observed:
                        g.add_const(np.asarray(observed[o]), name=o, vid=o)
                        static.add(o)
                continue
            for o in outs:
                if o and o in observed:
                    arr = observed[o]
                    g.add_value(Value(o, arr.shape, np.asarray(arr).dtype.name, name=o))
            nid = n.name or f"{n.op_type}_{idx}"
            while nid in used_ids:
                nid += "_"
            used_ids.add(nid)
            node = Node(nid, op, ins, [o for o in outs if o and o in observed], attrs, source=op)
            node.meta["output_slots"] = [k for k, o in enumerate(outs) if o and o in observed]
            node.meta["onnx_outputs"] = list(outs)
            g.add_node(node)
        g.outputs = [alias.get(o.name, o.name) for o in model.graph.output]
        g.validate()
        return g


# --------------------------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------------------------

def extract_graph(program, example_inputs: Sequence, name: str = "") -> TensorGraph:
    """Source-level graph of `program` for the given example inputs."""
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
        return _extract_compiled(program, example_inputs, name)
    import torch
    if isinstance(program, torch.nn.Module):
        return TorchExportFrontend().extract(program, example_inputs=example_inputs, name=name)
    raise TypeError(f"cannot extract a graph from {type(program).__name__}")


def _extract_compiled(c, example_inputs, name):
    meta = getattr(c, "meta", None) or {}
    model = getattr(c, "source_model", None)
    label = name or f"{c.compiler}"
    if c.compiler == "eager":
        return TorchExportFrontend().extract(model, example_inputs=example_inputs, name=label)
    if c.compiler in ("triton", "tilelang"):
        impl = {"backend": c.compiler, **dict(c.options)}
        return TorchExportFrontend().extract(model, example_inputs=example_inputs, impl=impl,
                                             name=label)
    if c.compiler == "onnxruntime_gpu":
        path = meta.get("ort_optimized_onnx")
        if not path or not os.path.isfile(path):
            raise ExtractionError("compiled ORT program has no saved optimized graph "
                                  f"({meta.get('ort_optimized_error', 'missing')})")
        g = OnnxFrontend().extract(path, example_inputs=example_inputs, name=label)
        g.meta["impl"] = {"backend": "onnxruntime_gpu", **dict(c.options)}
        return g
    raise Unimplemented(f"graph extraction for {c.compiler} artifacts is not implemented "
                        "(Inductor kernels and TensorRT engines expose no dataflow graph yet)")
