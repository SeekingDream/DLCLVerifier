"""GPU compilation paths for evaluation. Use through `utils.load_compiler`.

    compile_fn = load_compiler("tensorrt")            # or an id, see COMPILERS
    compiled = compile_fn(model, example_inputs)      # example_inputs: one batch from the loader
    out = compiled(*inputs)                           # same call and output as model(*inputs)

Paths:

  0 eager            PyTorch eager, the unoptimized reference.
  1 torch_compile    TorchInductor. Fuses pointwise ops and reductions into generated Triton
                     kernels; with mode="max-autotune" it also picks Triton GEMM templates.
  2 onnxruntime_gpu  ONNX export, then ONNX Runtime CUDA execution provider with graph rewrites.
  3 tensorrt         ONNX export, then a TensorRT engine (layer fusion, kernel/tactic selection).
  4 triton           Every GEMM-shaped op (linear, matmul, bmm, addmm, conv1d/conv2d via im2col)
                     runs on a Triton kernel compiled by the Triton compiler.
  5 tilelang         The same ops run on a TileLang kernel compiled by TileLang.

With default options the triton and tilelang kernels both accumulate each dot product in k order
in one fp32 accumulator, so they agree bitwise; tile sizes (block_*) do not change that order.
split_k > 1 does: it sums split_k partial dot products. Use it to get distinct addition trees.

Scope (docs/SPECIFICATION.md): only the order of floating-point additions may change. Every path
therefore runs in float32 with TF32 off. TileLang's T.gemm uses TF32 tensor cores for float32
(relative error ~1e-3 measured on RTX 6000 Ada), so the TileLang kernel accumulates with explicit
fp32 multiply-adds instead. FMA contraction inside GPU kernels is not controlled.
"""
from __future__ import annotations

import collections
import hashlib
import os
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode

WORK_DIR = os.environ.get("DLCLV_WORK_DIR", os.path.join(os.path.dirname(__file__), "work_dir"))

COMPILERS = {0: "eager", 1: "torch_compile", 2: "onnxruntime_gpu", 3: "tensorrt",
             4: "triton", 5: "tilelang"}


def strict_fp32():
    """Turn off TF32 so float32 matmul and convolution stay float32."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


@dataclass
class Compiled:
    """A compiled model. Call it like the source model."""
    compiler: str
    options: dict
    run: Callable
    compile_seconds: float = 0.0
    artifact: str = ""        # path of the exported or built artifact, if any
    stats: object = None      # per-op swap counts for the kernel paths
    source_model: object = None   # the model this was compiled from (graph extraction)
    meta: dict = None             # evidence for graph extraction: the ONNX Runtime optimized
                                  # graph, the kernel paths' call trace, Inductor's input graphs

    def __call__(self, *inputs):
        with torch.no_grad():
            return self.run(*inputs)


def _device_of(model):
    for t in list(model.parameters()) + list(model.buffers()):
        return t.device
    return torch.device("cpu")


def _require_cuda(model, name):
    dev = _device_of(model)
    if dev.type != "cuda":
        raise NotImplementedError(f"{name} is a GPU path; model is on {dev}")
    return dev


def _warmup(compiled_run, example_inputs):
    t = time.time()
    with torch.no_grad():
        compiled_run(*example_inputs)
    torch.cuda.synchronize()
    return time.time() - t


# --------------------------------------------------------------------------------------------
# 0 eager, 1 torch_compile
# --------------------------------------------------------------------------------------------

def compile_eager(model, example_inputs):
    return Compiled("eager", {}, model, source_model=model, meta={})


class InductorCapture:
    """Records every graph that AOTAutograd hands to TorchInductor for one model.

    The record is made inside the compilation that produces the artifact, so the graph is the
    one Inductor compiled, not a second trace. `records` is read by
    `dlclv.frontends.AotGraphFrontend`, which documents the fields.
    """

    def __init__(self, model):
        self.params = list(model.parameters()) + list(model.buffers())
        self.records = []
        self.inputs, self.key, self.seq = (), None, 0
        self.dynamo = (None, [])     # Dynamo's graph module and the real inputs it passed

    @staticmethod
    def call_key(inputs):
        return tuple((tuple(x.shape), str(x.dtype)) for x in inputs)

    def begin(self, inputs):
        from torch._dynamo.utils import counters
        self.inputs, self.key = tuple(inputs), self.call_key(inputs)
        self.seq += 1
        self._n0 = len(self.records)
        self._breaks0 = sum(counters["graph_break"].values())

    def end(self):
        from torch._dynamo.utils import counters
        breaks = sum(counters["graph_break"].values()) - self._breaks0
        for r in self.records[self._n0:]:
            r["graph_breaks"] = breaks
        self.inputs = ()

    def _same(self, t, xs):
        for k, x in enumerate(xs):
            if t is x or (isinstance(x, torch.Tensor) and t.shape == x.shape and t.dtype == x.dtype
                          and t.device == x.device and t.numel() > 0
                          and t.data_ptr() == x.data_ptr()):
                return k
        return None

    def real_inputs(self, example_inputs):
        """The real tensors behind the (fake) example inputs of the AOT graph: the parameters
        and buffers of Dynamo's graph module, then the inputs Dynamo passed to the backend."""
        gm, xs = self.dynamo
        real = list(dict(gm.named_parameters(remove_duplicate=False)).values()) + \
            list(dict(gm.named_buffers(remove_duplicate=False)).values()) + list(xs)
        if len(real) != len(example_inputs):
            raise RuntimeError(f"{len(example_inputs)} graph inputs, {len(real)} real inputs")
        for r, e in zip(real, example_inputs):
            if isinstance(r, torch.Tensor) != isinstance(e, torch.Tensor) or (
                    isinstance(r, torch.Tensor) and (tuple(r.shape) != tuple(e.shape)
                                                     or r.dtype != e.dtype)):
                raise RuntimeError("graph inputs do not line up with the real inputs")
        return real

    def record(self, gm, example_inputs, kwargs):
        import torch.fx
        graph = torch.fx.Graph()
        graph.output(graph.graph_copy(gm.graph, {}))
        for n in graph.nodes:
            n.meta = {}                       # drop fake tensors; the frontend re-observes shapes
        out_meta = [n for n in gm.graph.nodes if n.op == "output"][0].meta
        static = set(kwargs.get("static_input_idxs") or ())
        bindings = []
        for i, t in enumerate(self.real_inputs(example_inputs)):
            if not isinstance(t, torch.Tensor):
                bindings.append(("literal", int(t)))
                continue
            k = self._same(t, self.inputs)
            if k is not None:
                bindings.append(("input", k))
            elif i in static or self._same(t, self.params) is not None:
                bindings.append(("const", t))
            else:
                bindings.append(("unknown", t))
        ctx = torch._guards.TracingContext.try_get()
        fw = getattr(ctx, "fw_metadata", None) if ctx is not None else None
        self.records.append({
            "graph_module": torch.fx.GraphModule(gm, graph),
            "bindings": bindings,
            "user_outputs": list(out_meta.get("user_visible_output_idxs") or []) or None,
            "num_mutated": int(getattr(fw, "num_mutated_inp_runtime_indices", 0) or 0),
            "is_inference": bool(kwargs.get("is_inference", True)),
            "call": self.key,
            "seq": self.seq,                  # which call compiled it (a later one: recompiled)
            "torch": torch.__version__,
        })

    def inner_compile(self, gm, example_inputs, **kwargs):
        from torch._inductor.compile_fx import compile_fx_inner
        try:
            self.record(gm, example_inputs, kwargs)
        except Exception as e:                # the frontend reports it; compilation goes on
            self.records.append({"error": f"{type(e).__name__}: {e}", "call": self.key,
                                 "seq": self.seq})
        return compile_fx_inner(gm, example_inputs, **kwargs)


def compile_inductor(model, mode="default", dynamic=False):
    """torch.compile(model, backend="inductor", mode=mode) with the compiled graphs captured.

    Returns (run, capture). The backend is Inductor's own wrapper; the only change is that
    `compile_fx` receives an `inner_compile` that records its arguments and then calls
    Inductor's `compile_fx_inner`.
    """
    cap = InductorCapture(model)

    class Backend(torch._TorchCompileInductorWrapper):
        def __call__(self, model_, inputs_):
            from torch._inductor.compile_fx import compile_fx
            cap.dynamo = (model_, list(inputs_))
            return compile_fx(model_, inputs_, inner_compile=cap.inner_compile,
                              config_patches=self.config)

    fn = torch.compile(model, backend=Backend(mode, None, dynamic), dynamic=dynamic)

    def run(*inputs):
        cap.begin(inputs)
        try:
            # max-autotune and reduce-overhead use CUDA graphs, which reuse output buffers.
            torch.compiler.cudagraph_mark_step_begin()
            out = fn(*inputs)
            return out.clone() if isinstance(out, torch.Tensor) else out
        finally:
            cap.end()

    return run, cap


def compile_torch(model, example_inputs, mode="default", dynamic=False):
    _require_cuda(model, "torch_compile")
    torch._dynamo.reset()
    run, cap = compile_inductor(model, mode, dynamic)
    secs = _warmup(run, example_inputs)
    return Compiled("torch_compile", {"mode": mode, "dynamic": dynamic}, run, secs,
                    source_model=model, meta={"aot_graphs": cap.records})


# --------------------------------------------------------------------------------------------
# ONNX export shared by 2 and 3
# --------------------------------------------------------------------------------------------

def _model_key(model, example_inputs, extra=""):
    task = getattr(model, "task", None)
    name = f"{task.model}" if task is not None else f"{type(model).__name__}_{id(model)}"
    sig = ";".join(f"{tuple(x.shape[1:])}:{x.dtype}" for x in example_inputs) + extra
    return f"{name}_{hashlib.sha1(sig.encode()).hexdigest()[:10]}"


class _TraceableMasks:
    """During ONNX export, build HF attention masks without vmap.

    transformers uses a vmap-based mask builder on torch >= 2.6, which the TorchScript ONNX
    exporter cannot trace (RuntimeError: _Map_base::at). Its pre-2.6 builder makes the same mask
    with plain tensor ops.
    """

    def __enter__(self):
        try:
            import transformers.masking_utils as mu
        except ImportError:
            self.mu = None
            return self
        self.mu, self.saved = mu, mu.sdpa_mask
        mu.sdpa_mask = mu.sdpa_mask_older_torch
        return self

    def __exit__(self, *exc):
        if self.mu is not None:
            self.mu.sdpa_mask = self.saved


def export_onnx(model, example_inputs, opset=17):
    """Export to ONNX with a dynamic batch dimension. Reuses an earlier export of the same model."""
    key = _model_key(model, example_inputs, f"opset{opset}")
    d = os.path.join(WORK_DIR, "onnx", key)
    path = os.path.join(d, "model.onnx")
    names = [f"input_{i}" for i in range(len(example_inputs))]
    if not os.path.isfile(path):
        os.makedirs(d, exist_ok=True)
        tmp = os.path.join(d, "tmp.onnx")
        with torch.no_grad(), _TraceableMasks():
            torch.onnx.export(
                model, tuple(example_inputs), tmp, input_names=names, output_names=["output"],
                dynamic_axes={n: {0: "batch"} for n in names + ["output"]},
                opset_version=opset, do_constant_folding=True, dynamo=False)
        os.replace(tmp, path)   # external weight files keep the names written at export
    return path, names


# --------------------------------------------------------------------------------------------
# 2 onnxruntime_gpu
# --------------------------------------------------------------------------------------------

_ORT_LEVELS = {"disable": "ORT_DISABLE_ALL", "basic": "ORT_ENABLE_BASIC",
               "extended": "ORT_ENABLE_EXTENDED", "all": "ORT_ENABLE_ALL"}


def compile_ort(model, example_inputs, graph_optimization="all", conv_algo_search="EXHAUSTIVE"):
    import onnxruntime as ort
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
    dev = _require_cuda(model, "onnxruntime_gpu")
    t = time.time()
    path, names = export_onnx(model, example_inputs)
    so = ort.SessionOptions()
    so.graph_optimization_level = getattr(ort.GraphOptimizationLevel, _ORT_LEVELS[graph_optimization])
    so.log_severity_level = 3
    # Save the graph ONNX Runtime actually runs (after its fusions) for graph extraction.
    optimized = os.path.join(os.path.dirname(path),
                             f"ort_optimized_{graph_optimization}.onnx")
    so.optimized_model_filepath = optimized
    so.add_session_config_entry("session.optimized_model_external_initializers_file_name",
                                f"ort_optimized_{graph_optimization}.weights")
    so.add_session_config_entry("session.optimized_model_external_initializers_min_size_in_bytes",
                                "1024")
    dev_id = dev.index if dev.index is not None else torch.cuda.current_device()
    providers = [("CUDAExecutionProvider", {"device_id": dev_id, "use_tf32": 0,
                                             "cudnn_conv_algo_search": conv_algo_search})]
    sess = ort.InferenceSession(path, so, providers=providers)
    if sess.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError(f"ORT fell back to {sess.get_providers()}")
    import numpy as np
    np_types = {torch.float32: np.float32, torch.int64: np.int64, torch.int32: np.int32}

    def run(*inputs):
        io = sess.io_binding()
        xs = [x.to(dev).contiguous() for x in inputs]
        for n, x in zip(names, xs):
            io.bind_input(n, "cuda", dev_id, np_types[x.dtype], list(x.shape), x.data_ptr())
        io.bind_output("output", "cpu")
        sess.run_with_iobinding(io)
        return torch.from_numpy(io.copy_outputs_to_cpu()[0]).to(dev)

    secs = time.time() - t + _warmup(run, example_inputs)
    opts = {"graph_optimization": graph_optimization, "conv_algo_search": conv_algo_search}
    meta = {"onnx": path, "ort_optimized_onnx": optimized if os.path.isfile(optimized) else None}
    return Compiled("onnxruntime_gpu", opts, run, secs, artifact=path, source_model=model,
                    meta=meta)


# --------------------------------------------------------------------------------------------
# 3 tensorrt
# --------------------------------------------------------------------------------------------

def compile_tensorrt(model, example_inputs, optimization_level=3, workspace_gb=8):
    import tensorrt as trt
    dev = _require_cuda(model, "tensorrt")
    t = time.time()
    onnx_path, names = export_onnx(model, example_inputs)
    bs = example_inputs[0].shape[0]
    key = f"bs{bs}_opt{optimization_level}_trt{trt.__version__}"
    plan = os.path.join(os.path.dirname(onnx_path), f"{key}.plan")
    logger = trt.Logger(trt.Logger.ERROR)
    if not os.path.isfile(plan):
        builder = trt.Builder(logger)
        network = builder.create_network(0)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(onnx_path):
            errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError("TensorRT ONNX parse failed: " + "; ".join(errs))
        config = builder.create_builder_config()
        config.clear_flag(trt.BuilderFlag.TF32)   # float32 only; FP16/INT8 are never set
        config.builder_optimization_level = optimization_level
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
        profile = builder.create_optimization_profile()
        for i, x in enumerate(example_inputs):
            shape = tuple(x.shape)
            profile.set_shape(network.get_input(i).name, (1, *shape[1:]), shape, shape)
        config.add_optimization_profile(profile)
        engine_bytes = builder.build_serialized_network(network, config)
        if engine_bytes is None:
            raise RuntimeError("TensorRT engine build failed")
        with open(plan, "wb") as f:
            f.write(engine_bytes)
    with open(plan, "rb") as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    ctx = engine.create_execution_context()
    to_torch = {trt.float32: torch.float32, trt.int64: torch.int64, trt.int32: torch.int32}
    out_dtype = to_torch[engine.get_tensor_dtype("output")]

    def run(*inputs):
        if inputs[0].shape[0] > bs:
            raise ValueError(f"engine built for batch <= {bs}")
        xs = [x.to(dev).contiguous() for x in inputs]
        for n, x in zip(names, xs):
            ctx.set_input_shape(n, tuple(x.shape))
            ctx.set_tensor_address(n, x.data_ptr())
        out = torch.empty(tuple(ctx.get_tensor_shape("output")), dtype=out_dtype, device=dev)
        ctx.set_tensor_address("output", out.data_ptr())
        stream = torch.cuda.current_stream(dev)
        if not ctx.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        stream.synchronize()
        return out

    secs = time.time() - t + _warmup(run, example_inputs)
    opts = {"optimization_level": optimization_level, "max_batch": bs}
    return Compiled("tensorrt", opts, run, secs, artifact=plan, source_model=model,
                    meta={"onnx": onnx_path})


# --------------------------------------------------------------------------------------------
# 4 triton, 5 tilelang: GEMM-shaped ops lowered to DSL kernels
# --------------------------------------------------------------------------------------------

def _as_pair(v):
    return tuple(v) if isinstance(v, (tuple, list)) else (v, v)


class GemmMode(TorchFunctionMode):
    """Routes float32 CUDA GEMM-shaped ops to `bmm(a, b) -> a @ b` for 3-D a [Bt,M,K], b [Bt,K,N].

    Bias is added after the kernel, which rounds the same way as a fused epilogue. Ops that do not
    match (other dtypes, 1-D matmul, string padding) run unchanged and are counted as fallbacks.
    """

    def __init__(self, bmm):
        super().__init__()
        self.bmm = bmm
        self.stats = collections.Counter()
        self.log = None        # a list while a call trace is being recorded (kernel_trace)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = _HANDLED.get(func)
        if name is not None:
            out = getattr(self, "_" + name)(*args, **kwargs)
            if self.log is not None:
                ts = [t for t in list(args) + list(kwargs.values()) if isinstance(t, torch.Tensor)]
                self.log.append({"op": name, "handled": out is not NotImplemented,
                                 "shapes": [tuple(t.shape) for t in ts],
                                 "dtypes": [str(t.dtype).replace("torch.", "") for t in ts]})
            if out is not NotImplemented:
                self.stats[name] += 1
                return out
            self.stats["fallback_" + name] += 1
        return func(*args, **kwargs)

    @staticmethod
    def _ok(*ts):
        return all(t.dtype == torch.float32 and t.is_cuda for t in ts)

    def _matmul(self, a, b):
        if not self._ok(a, b) or a.dim() < 2 or b.dim() < 2:
            return NotImplemented
        if b.dim() == 2:
            out = self.bmm(a.reshape(1, -1, a.shape[-1]), b.unsqueeze(0))
            return out.view(*a.shape[:-1], b.shape[-1])
        batch = torch.broadcast_shapes(a.shape[:-2], b.shape[:-2])
        M, K, N = a.shape[-2], a.shape[-1], b.shape[-1]
        a3 = a.expand(*batch, M, K).reshape(-1, M, K)
        b3 = b.expand(*batch, K, N).reshape(-1, K, N)
        return self.bmm(a3, b3).view(*batch, M, N)

    def _mm(self, a, b):
        if a.dim() != 2 or b.dim() != 2:
            return NotImplemented
        return self._matmul(a, b)

    def _bmm(self, a, b):
        if a.dim() != 3 or b.dim() != 3:
            return NotImplemented
        return self._matmul(a, b)

    def _addmm(self, bias, a, b, beta=1, alpha=1):
        if beta != 1 or alpha != 1 or not self._ok(bias):
            return NotImplemented
        out = self._mm(a, b)
        return NotImplemented if out is NotImplemented else out + bias

    def _linear(self, x, weight, bias=None):
        if not self._ok(x, weight) or x.dim() < 1:
            return NotImplemented
        out = self.bmm(x.reshape(1, -1, x.shape[-1]), weight.t().unsqueeze(0))
        out = out.view(*x.shape[:-1], weight.shape[0])
        return out if bias is None else out + bias

    def _conv2d(self, x, w, bias=None, stride=1, padding=0, dilation=1, groups=1):
        if isinstance(padding, str) or not self._ok(x, w) or x.dim() != 4:
            return NotImplemented
        stride, padding, dilation = _as_pair(stride), _as_pair(padding), _as_pair(dilation)
        N, _, H, W = x.shape
        OC, Cg, kh, kw = w.shape
        G, K = groups, Cg * kh * kw
        Ho = (H + 2 * padding[0] - dilation[0] * (kh - 1) - 1) // stride[0] + 1
        Wo = (W + 2 * padding[1] - dilation[1] * (kw - 1) - 1) // stride[1] + 1
        cols = F.unfold(x, (kh, kw), dilation=dilation, padding=padding, stride=stride)
        L = cols.shape[-1]
        cols = cols.view(N, G, K, L).permute(1, 2, 0, 3).reshape(G, K, N * L)
        out = self.bmm(w.reshape(G, OC // G, K), cols)              # [G, OC/G, N*L]
        out = out.view(G, OC // G, N, L).permute(2, 0, 1, 3).reshape(N, OC, Ho, Wo)
        return out if bias is None else out + bias.view(1, -1, 1, 1)

    def _conv1d(self, x, w, bias=None, stride=1, padding=0, dilation=1, groups=1):
        if isinstance(padding, str) or x.dim() != 3:
            return NotImplemented
        one = lambda v: v[0] if isinstance(v, (tuple, list)) else v
        out = self._conv2d(x.unsqueeze(2), w.unsqueeze(2), bias, (1, one(stride)),
                           (0, one(padding)), (1, one(dilation)), groups)
        return out if out is NotImplemented else out.squeeze(2)


_HANDLED = {
    torch.matmul: "matmul", torch.Tensor.matmul: "matmul", torch.Tensor.__matmul__: "matmul",
    torch.mm: "mm", torch.Tensor.mm: "mm", torch.bmm: "bmm", torch.Tensor.bmm: "bmm",
    torch.addmm: "addmm", F.linear: "linear", F.conv2d: "conv2d", F.conv1d: "conv1d",
}


def _compile_kernel_path(name, bmm, model, example_inputs, options):
    _require_cuda(model, name)
    mode = GemmMode(bmm)

    def run(*inputs):
        with mode:
            return model(*inputs)

    def kernel_trace(*inputs):
        """Run once and return (output, calls): one record per intercepted GEMM-shaped call, in
        program order, saying whether the kernel ran it or it fell back to PyTorch."""
        mode.log = []
        try:
            out = run(*inputs)
        finally:
            calls, mode.log = mode.log, None
        return out, calls

    secs = _warmup(run, example_inputs)
    return Compiled(name, options, run, secs, stats=mode.stats, source_model=model,
                    meta={"kernel_trace": kernel_trace})


# ---- Triton ----

def triton_bmm(a, b, block_m=64, block_n=64, block_k=32, split_k=1, num_warps=4, num_stages=3):
    """a [Bt,M,K] @ b [Bt,K,N] in float32 (IEEE dot, no TF32). Any strides, including 0.

    split_k > 1 writes split_k partial sums and adds them in a fixed order, so it stays
    deterministic.
    """
    import triton
    import kernels
    Bt, M, K = a.shape
    N = b.shape[2]
    if b.shape[0] != Bt:
        b = b.expand(Bt, -1, -1)
    c = torch.empty((Bt, split_k, M, N), device=a.device, dtype=torch.float32)
    k_per = triton.cdiv(triton.cdiv(K, split_k), block_k) * block_k
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n), Bt * split_k)
    kernels.triton_bmm_kernel()[grid](
        a, b, c, M, N, K, k_per, *a.stride(), *b.stride(), *c.stride(),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, SPLIT_K=split_k,
        num_warps=num_warps, num_stages=num_stages)
    return c[:, 0] if split_k == 1 else c.sum(1)


def compile_triton(model, example_inputs, block_m=64, block_n=64, block_k=32, split_k=1,
                   num_warps=4, num_stages=3):
    opts = dict(block_m=block_m, block_n=block_n, block_k=block_k, split_k=split_k,
                num_warps=num_warps, num_stages=num_stages)
    return _compile_kernel_path("triton", lambda a, b: triton_bmm(a, b, **opts), model,
                                example_inputs, opts)


# ---- TileLang ----

def tilelang_bmm(a, b, block_m=64, block_n=64, block_k=32, num_stages=2, threads=128, split_k=1):
    """a [Bt,M,K] @ b [Bt,K,N] in float32. Compiled once per (N, K, layout, config).

    split_k > 1 zero-pads K to a multiple of split_k, computes split_k partial products of
    K/split_k terms each, and adds them in a fixed order.
    """
    import kernels
    Bt, M, K = a.shape
    N = b.shape[2]
    if b.shape[0] != Bt:
        b = b.expand(Bt, -1, -1)
    if split_k > 1:
        Kc = -(-K // split_k)
        pad = Kc * split_k - K
        if pad:
            a = F.pad(a, (0, pad))
            b = F.pad(b, (0, 0, 0, pad))
        a = a.reshape(Bt, M, split_k, Kc).permute(0, 2, 1, 3).reshape(Bt * split_k, M, Kc)
        b = b.reshape(Bt * split_k, Kc, N)
        return tilelang_bmm(a, b, block_m, block_n, block_k, num_stages, threads).view(
            Bt, split_k, M, N).sum(1)
    bt_ = b.transpose(1, 2)
    trans_b = bt_.is_contiguous() and not b.is_contiguous()
    b_arg = bt_ if trans_b else b.contiguous()
    kern = kernels.tilelang_bmm_kernel(N, K, trans_b, block_m, block_n, block_k, num_stages, threads)
    return kern(a.contiguous(), b_arg)


def compile_tilelang(model, example_inputs, block_m=64, block_n=64, block_k=32, num_stages=2,
                     threads=128, split_k=1):
    import logging
    logging.getLogger("TileLang").setLevel(logging.WARNING)
    opts = dict(block_m=block_m, block_n=block_n, block_k=block_k, num_stages=num_stages,
                threads=threads, split_k=split_k)
    return _compile_kernel_path("tilelang", lambda a, b: tilelang_bmm(a, b, **opts), model,
                                example_inputs, opts)


BACKENDS = {"eager": compile_eager, "torch_compile": compile_torch,
            "onnxruntime_gpu": compile_ort, "tensorrt": compile_tensorrt,
            "triton": compile_triton, "tilelang": compile_tilelang}
