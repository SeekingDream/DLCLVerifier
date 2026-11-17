"""Fail-closed reconstruction of a tensor program as explicit eager PyTorch arithmetic.
See docs/RECONSTRUCTION.md for scope, execution assumptions and what is (not) guaranteed.

The reconstructed program A' is an ordered ATen graph (FX is only the SSA container) in which
every node is one of:

* an explicit primitive (`PRIMITIVES`): one IEEE operation, cast or data movement;
* a lowered kernel: a library or DSL kernel call replaced by a function built only from
  `dlclv.primitives` (explicit accumulation order, FMA, partial accumulators), supplied by a
  `dlclv.lowering.Lowering` together with its evidence, and executed under `PrimitiveGuard`;
* in `mode='eager-kernel'` only, a preserved library call (`EAGER_KERNELS`): the same opaque
  kernel as the original, not reconstructed.

An in-place call on a fresh, otherwise unused intermediate (`INPLACE`) is run as its
out-of-place overload and then classified as above.

No decomposition, constant folding, reassociation or generic `sum`/`matmul` substitution is
performed. Anything else raises `UnsupportedReconstruction`.
"""
from __future__ import annotations

import collections
import copy
import operator
import platform
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from .bitwise import bitwise_equal


class UnsupportedReconstruction(RuntimeError):
    """The implementation's arithmetic semantics are unavailable."""


class ExecutionMismatch(RuntimeError):
    """Execution is outside the captured domain/configuration."""


@dataclass(frozen=True)
class ExecutionConfig:
    torch_version: str
    machine: str
    build: str
    threads: int
    interop_threads: int
    default_dtype: str
    mkldnn_enabled: bool
    mkldnn_deterministic: Optional[bool]
    deterministic_algorithms: bool
    deterministic_warn_only: bool
    float32_matmul_precision: str
    subnormals_preserved: bool = True      # measured: a float32 subnormal survives x * 1
    device: str = "cpu"
    accelerator: Optional[tuple] = None    # CUDA: GPU, CUDA/cuDNN versions, TF32 and cuDNN flags

    @classmethod
    def current(cls, device="cpu"):
        import torch
        try:
            cpu_autocast = torch.is_autocast_enabled('cpu')
        except TypeError:                              # older PyTorch
            cpu_autocast = torch.is_autocast_cpu_enabled()
        if cpu_autocast or torch.is_autocast_enabled():
            raise ExecutionMismatch('autocast must be disabled')
        device = torch.device(device)
        tiny = torch.tensor([2.0 ** -140], dtype=torch.float32, device=device)
        accelerator = None
        if device.type == 'cuda':
            accelerator = (torch.cuda.get_device_name(device), str(torch.version.cuda),
                           str(torch.backends.cudnn.version()),
                           ('matmul.allow_tf32', torch.backends.cuda.matmul.allow_tf32),
                           ('cudnn.allow_tf32', torch.backends.cudnn.allow_tf32),
                           ('cudnn.enabled', torch.backends.cudnn.enabled),
                           ('cudnn.benchmark', torch.backends.cudnn.benchmark),
                           ('cudnn.deterministic', torch.backends.cudnn.deterministic))
        return cls(str(torch.__version__), platform.platform() + platform.machine(),
                   torch.__config__.show(), torch.get_num_threads(),
                   torch.get_num_interop_threads(), str(torch.get_default_dtype()),
                   torch.backends.mkldnn.enabled, getattr(torch.backends.mkldnn, 'deterministic', None),
                   torch.are_deterministic_algorithms_enabled(),
                   torch.is_deterministic_algorithms_warn_only_enabled(),
                   torch.get_float32_matmul_precision(),
                   bool((tiny * 1.0 == tiny).item() and tiny.item() != 0.0),
                   device.type, accelerator)


# Exact overloads, not operator family names. add(alpha != 1) is rejected below.
# Keeping ATen overloads also preserves scalar promotion and each cast boundary.
PRIMITIVES = frozenset('''
aten.add.Tensor aten.add.Scalar aten.sub.Tensor aten.sub.Scalar
aten.mul.Tensor aten.mul.Scalar aten.neg.default
aten.div.Tensor aten.div.Scalar
aten._to_copy.default aten.to.dtype aten.clone.default aten.detach.default aten.alias.default
aten.view.default aten.reshape.default aten.permute.default aten.transpose.int
aten.t.default aten.squeeze.default aten.squeeze.dim aten.unsqueeze.default
aten.slice.Tensor aten.select.int aten.expand.default aten.flatten.using_ints
aten.cat.default
'''.split())


# Explicit opt-in: these calls preserve library kernels, not their internal
# accumulation instructions. Never describe this mode as primitive lowering.
EAGER_KERNELS = frozenset('''
aten.convolution.default aten.conv2d.default
aten.relu.default aten.relu6.default aten.hardtanh.default aten.cat.default
aten.max_pool2d_with_indices.default aten.max_pool2d.default
aten._native_batch_norm_legit_no_training.default aten.batch_norm.default
aten.mean.dim aten.adaptive_avg_pool2d.default aten.addmm.default aten.linear.default
aten.matmul.default aten.mm.default aten.bmm.default aten.mv.default aten.dot.default
aten.sum.dim_IntList
'''.split())
MODES = ('primitive', 'eager-kernel')

# In-place calls that newer torch.export keeps (`relu_`, `out += identity`). Each is executed
# as its out-of-place overload, which runs the same kernel into a fresh tensor. That is the
# same program only when nothing else can observe the mutation, so it is admitted only if the
# mutated operand is a fresh intermediate (not an input, constant or view) with no other user.
INPLACE = {'aten.relu_.default': 'aten.relu.default', 'aten.add_.Tensor': 'aten.add.Tensor',
           'aten.hardtanh_.default': 'aten.hardtanh.default',
           'aten.relu6_.default': 'aten.relu6.default'}
ALIASING = frozenset('''
aten.view.default aten.reshape.default aten.permute.default aten.transpose.int aten.t.default
aten.squeeze.default aten.squeeze.dim aten.unsqueeze.default aten.slice.Tensor aten.select.int
aten.expand.default aten.flatten.using_ints aten.alias.default aten.detach.default
aten._to_copy.default aten.to.dtype aten.contiguous.default
'''.split())


def _spec(x, *, device='cpu', allow_integer=False):
    import torch
    if type(x) not in (torch.Tensor, torch.nn.Parameter):
        raise ExecutionMismatch('only ordinary tensor inputs/intermediates are supported')
    if x.device.type != device or x.layout != torch.strided:
        raise ExecutionMismatch(f'only strided {device} tensors are supported here')
    dtypes = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
    if allow_integer:
        dtypes += (torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8, torch.bool)
    if x.dtype not in dtypes:
        raise ExecutionMismatch(f'unsupported tensor dtype: {x.dtype}')
    return tuple(x.shape), x.dtype, tuple(x.stride()), x.storage_offset()


class ReconstructedProgram:
    """Callable explicit program with conditional evidence, never a certificate.

    `lowerings` are consulted for every call, in program order, during the capture run on the
    example inputs; a call one of them lowers is executed as explicit primitives from then on.
    """

    def __init__(self, gm, example_inputs, *, source, assumptions=(), mode='primitive',
                 lowerings: Sequence = ()):
        import torch
        from .lowering import LoweringFailed
        if mode not in MODES:
            raise ValueError(f'unknown reconstruction mode: {mode}')
        self.mode = mode
        devices = {x.device.type for x in example_inputs if isinstance(x, torch.Tensor)}
        if len(devices) > 1:
            raise ExecutionMismatch('example inputs are on different devices')
        self.device = devices.pop() if devices else 'cpu'
        if self.device not in ('cpu', 'cuda'):
            raise ExecutionMismatch(f'unsupported device: {self.device}')
        self.config = ExecutionConfig.current(self.device)
        if not self.config.subnormals_preserved:
            raise ExecutionMismatch('subnormals are flushed to zero (FTZ/DAZ); the primitives '
                                    'assume gradual underflow')
        self.input_specs = tuple(_spec(x, device=self.device) for x in example_inputs)
        self._gm = copy.deepcopy(gm)
        self._gm.graph.lint()
        self._lowerings = list(lowerings)
        self._lowered = {}                 # node name -> LoweredOp
        self._preserved = set()
        self._functional = {}              # in-place node name -> out-of-place overload
        self._ops = collections.Counter()  # primitive operations executed inside lowered kernels
        self._node_specs = {}
        try:
            self._execute(example_inputs, capture=True)
            for low in self._lowerings:
                low.finish()
        except LoweringFailed as exc:
            raise UnsupportedReconstruction(f'kernel lowering failed: {exc}') from exc
        failures = {}
        for low in self._lowerings:
            failures.update(getattr(low, 'failures', None) or {})
        lowered = [{'node': name, 'target': str(self._targets[name]), 'kernel': op.kernel,
                    'arithmetic': op.description, 'level': op.level, 'evidence': op.evidence}
                   for name, op in self._lowered.items()]
        levels = sorted({op.level for op in self._lowered.values()})
        extra = []
        for op in self._lowered.values():
            extra += [a for a in op.assumptions if a not in extra]
        self.report = {
            'status': 'conditional', 'source': source, 'universal_certificate': False,
            'mode': mode, 'device': self.device,
            'preserved_kernels': sorted(self._preserved),
            'lowered_kernels': lowered,
            'lowering_levels': levels,
            'lowering_failures': failures,
            'lowered_primitive_ops': dict(sorted(self._ops.items())),
            'assumptions': list(assumptions) + [
                'Same device, PyTorch binary, dispatch and floating-point environment.',
                'Eager execution only; no autocast, compiler, dispatch overrides or concurrent state changes.',
                'Rounding mode stays round-to-nearest-even (not introspectable here); gradual '
                'underflow was measured on this device when the program was built and is '
                're-checked on every call.',
                'Each explicit primitive is one correctly rounded IEEE 754 operation in its dtype, '
                'and separate eager calls are never contracted into a fused operation.',
                'Source artifact has the stated eager ATen semantics; weights are frozen snapshots.']
            + extra,
            'excluded_inputs': ['Inputs for which the original or the reconstruction produces a '
                                'NaN: NaN sign and payload are hardware specific.'],
            'empirical': None,
        }
        if self._preserved:
            self.report['assumptions'].append(
                'Library kernels are preserved as opaque eager calls, not lowered to arithmetic. '
                'Identical dispatch must select deterministic kernels for the fixed tensor layouts.')

    # ---- admission of one node ----

    def _functionalize(self, n):
        """An in-place call of `INPLACE` on a fresh, otherwise unused intermediate: record and
        return its out-of-place overload (the same kernel into a fresh tensor)."""
        import torch
        name = str(n.target)
        src = n.args[0] if n.args else None
        if not isinstance(src, torch.fx.Node) or src.op != 'call_function' \
                or str(src.target) in ALIASING or src.target is operator.getitem \
                or len(src.users) != 1:
            raise UnsupportedReconstruction(
                f'{n.name}: {name} mutates a value that may be visible elsewhere')
        family, overload = INPLACE[name].split('.')[1:]
        self._functional[n.name] = getattr(getattr(torch.ops.aten, family), overload)
        return self._functional[n.name]

    def _admit(self, n):
        """Static check of a call that no lowering took. Returns True if it is a preserved
        library kernel, False if it is an explicit primitive; raises otherwise."""
        import torch
        mode = self.mode
        name = str(self._functional.get(n.name, n.target))
        kernel = mode == 'eager-kernel' and name in EAGER_KERNELS
        # A projection of a preserved kernel's tuple, or of a lowered kernel's (whose elements
        # that were not reconstructed are None and fail in `call_function` when projected).
        projection = n.target is operator.getitem and (
            mode == 'eager-kernel' or (n.args and getattr(n.args[0], 'name', None)
                                       in self._lowered))
        if projection and (len(n.args) != 2 or type(n.args[1]) is not int or n.kwargs):
            raise UnsupportedReconstruction('only fixed tuple/list projections are supported')
        if n.op != 'call_function' or not (name in PRIMITIVES or kernel or projection):
            why = [(getattr(low, 'failures', None) or {}).get(n.name) for low in self._lowerings]
            why = '; '.join(w for w in why if w)
            raise UnsupportedReconstruction(
                f'{n.name}: {n.target}: need lowered arithmetic/kernel semantics; '
                'opaque reductions, GEMM, FMA and hardware MMA are unsupported'
                + (f' (lowering attempted: {why})' if why else ''))
        if name in ('aten.add.Tensor', 'aten.add.Scalar',
                    'aten.sub.Tensor', 'aten.sub.Scalar'):
            if n.kwargs.get('alpha', 1) != 1:
                raise UnsupportedReconstruction(f'{n.name}: scaled add may fuse')
        if name in ('aten._to_copy.default', 'aten.to.dtype'):
            dev = n.kwargs.get('device')
            if dev is not None and torch.device(dev).type != self.device:
                raise UnsupportedReconstruction('device conversion is unsupported')
        if kernel:
            self._preserved.add(name)
        return kernel

    @property
    def code(self):
        """Reviewable program: the ATen graph in dependency order, then the explicit arithmetic
        of every lowered call. Execute through this object for domain guards."""
        text = self._gm.code
        if self._lowered:
            text += '\n# Lowered kernels (executed as dlclv.primitives, not as the call above):\n'
            for name, op in self._lowered.items():
                text += f'#   {name} = {self._targets[name]} [{op.level}] {op.kernel}\n'
                text += f'#       {op.description}\n'
        return text

    def _execute(self, inputs, capture=False):
        import torch
        from torch.utils import _pytree
        from .lowering import LoweringFailed
        from .primitives import NonPrimitiveOperation, PrimitiveGuard
        owner = self
        if capture:
            self._targets = {}

        class Runner(torch.fx.Interpreter):
            def call_function(self, target, args, kwargs):
                if target is operator.getitem and type(args[0]) not in (tuple, list):
                    raise UnsupportedReconstruction('getitem requires a tuple/list kernel result')
                result = super().call_function(target, args, kwargs)
                if target is operator.getitem and result is None:
                    raise UnsupportedReconstruction(
                        'the program uses a kernel result that the lowering does not '
                        'reconstruct (e.g. max-pool indices, batch-norm statistics)')
                return result

            def run_node(self, n):
                if n.op in ('placeholder', 'output', 'get_attr'):
                    result = super().run_node(n)
                else:
                    if capture:
                        owner._targets[n.name] = n.target
                        op, target = None, n.target
                        if n.op == 'call_function' and str(target) in INPLACE:
                            target = owner._functionalize(n)
                        if n.op == 'call_function' and target is not operator.getitem:
                            args, kwargs = self.fetch_args_kwargs_from_env(n)
                            for low in owner._lowerings:
                                op = low.lower(n.name, target, args, kwargs)
                                if op is not None:
                                    owner._lowered[n.name] = op
                                    break
                        if op is None:
                            owner._admit(n)
                    op = owner._lowered.get(n.name)
                    if op is None and n.name in owner._functional:
                        args, kwargs = self.fetch_args_kwargs_from_env(n)
                        result = owner._functional[n.name](*args, **kwargs)
                    elif op is None:
                        result = super().run_node(n)
                    else:
                        args, kwargs = self.fetch_args_kwargs_from_env(n)
                        if op.check is not None:
                            try:
                                op.check(*args, **kwargs)
                            except LoweringFailed as exc:
                                raise ExecutionMismatch(str(exc)) from exc
                        try:
                            with PrimitiveGuard(owner._ops if capture else None):
                                # The graph was exported around the library call, whose
                                # result is contiguous; later views rely on that layout.
                                result = op.fn(*args, **kwargs)
                                if isinstance(result, tuple):
                                    result = tuple(r if r is None else r.contiguous()
                                                   for r in result)
                                else:
                                    result = result.contiguous()
                        except NonPrimitiveOperation as exc:
                            raise UnsupportedReconstruction(f'{n.name}: {exc}') from exc
                specs = []
                for x in _pytree.tree_leaves(result):
                    if isinstance(x, torch.Tensor):
                        # Constants may be integer buffers (e.g. num_batches_tracked).
                        specs.append(_spec(x, device=owner.device, allow_integer=(
                            owner.mode == 'eager-kernel' or n.op == 'get_attr')))
                    elif x is not None:
                        raise ExecutionMismatch(f'{n.name}: non-tensor results unsupported')
                if capture:
                    owner._node_specs[n.name] = specs
                elif owner._node_specs[n.name] != specs:
                    raise ExecutionMismatch(f'{n.name}: intermediate layout/dtype changed')
                return result

        with torch.no_grad():
            return Runner(self._gm).run(*inputs)

    def __call__(self, *inputs):
        import torch
        if torch._dynamo.is_compiling():
            raise ExecutionMismatch('compiled reconstruction is unsupported')
        if ExecutionConfig.current(self.device) != self.config:
            raise ExecutionMismatch('execution configuration changed')
        if tuple(_spec(x, device=self.device) for x in inputs) != self.input_specs:
            raise ExecutionMismatch('input shape/dtype/stride/offset changed')
        return self._execute(inputs)

    def compare(self, reference: Callable, samples):
        """Finite sample evidence only. Does not change conditional status.

        `bitwise_equal` is strict (dtype, shape and every bit, NaN payloads included).
        `nan_bits_only` reports whether every difference found was between two NaNs: such a
        sample is outside the supported domain rather than evidence of a wrong order."""
        import torch
        from torch.utils import _pytree
        count, matched, nan_only = 0, True, True
        for inputs in samples:
            with torch.no_grad():
                a, sa = _pytree.tree_flatten(reference(*inputs))
                b, sb = _pytree.tree_flatten(self(*inputs))
            same = sa == sb and len(a) == len(b)
            for x, y in zip(a, b) if same else ():
                if bitwise_equal(x, y):
                    continue
                same = False
                if x.shape != y.shape or x.dtype != y.dtype or not x.is_floating_point():
                    nan_only = False
                else:
                    differ = (x != y) | (torch.signbit(x) != torch.signbit(y))
                    both_nan = torch.isnan(x) & torch.isnan(y)
                    nan_only &= not bool((differ & ~both_nan).any())
            if sa != sb or len(a) != len(b):
                nan_only = False
            matched &= same
            count += 1
        result = {'samples': count, 'bitwise_equal': matched if count else None,
                  'is_proof': False}
        if count and not matched:
            result['nan_bits_only'] = nan_only
        self.report['empirical'] = result
        return result


class ReconstructionFrontend:
    """Dispatch by representation. Adapters must return a ReconstructedProgram and state
    lowering assumptions.

    Built in: eager `nn.Module`, explicit eager ATen `GraphModule`, and the repository's
    `Compiled` wrappers `eager`, `triton` and `tilelang` (the latter two through
    `lowering.DslGemmLowering`, which reads the kernel path's own call trace) and `tvm`
    (through `tir_lowering`, which interprets the scheduled TIR of the lowered module).

    External adapters consume actual lowered artifacts, not a guessed implementation
    of a high-level operator. Register by format name; unknown formats fail closed.
    """

    def __init__(self):
        self.adapters = {}

    def register(self, format: str, adapter: Callable):
        if format in ('pytorch', 'aten') or format in self.adapters:
            raise ValueError(f'adapter already registered: {format}')
        self.adapters[format] = adapter

    def reconstruct(self, program, example_inputs, *, format='pytorch', mode='primitive',
                    lowerings: Sequence = (), probe_kernels: bool = False,
                    kernel_probe: bool = True, validation_trials: int = 5):
        import torch
        from .lowering import (DslGemmLowering, LoweringFailed, ProbedConvLowering,
                               ProbedLinearLowering, ProbedSumLowering)
        if mode not in MODES:
            raise ValueError(f'unknown reconstruction mode: {mode}')
        if format in self.adapters:
            result = self.adapters[format](program, example_inputs)
            if not isinstance(result, ReconstructedProgram):
                raise TypeError('adapter must return ReconstructedProgram')
            if result.mode != mode:
                raise UnsupportedReconstruction('adapter returned a different reconstruction mode')
            return result
        if format not in ('pytorch', 'aten'):
            raise UnsupportedReconstruction(
                f'{format}: no lowered-kernel adapter; operator graphs do not specify arithmetic')
        lowerings = list(lowerings)
        if probe_kernels:
            from .eager_lowering import eager_lowerings
            lowerings += [ProbedLinearLowering(validation_trials),
                          ProbedConvLowering(validation_trials),
                          ProbedSumLowering(validation_trials)]
            lowerings += eager_lowerings(validation_trials)
        reference, source, assumptions = program, 'strict torch.export', []
        if hasattr(program, 'compiler'):
            if program.compiler in ('triton', 'tilelang'):
                try:
                    lowerings.insert(0, DslGemmLowering(program, example_inputs, kernel_probe,
                                                        validation_trials))
                except LoweringFailed as exc:
                    raise UnsupportedReconstruction(str(exc)) from exc
                source = (f'strict torch.export of the source model; {program.compiler} kernel '
                          f'calls from the path\'s trace, lowered from kernels.py')
                assumptions.append(
                    f'The {program.compiler} path runs the source model\'s Python with only the '
                    'traced GEMM-shaped calls replaced; every other operation is the PyTorch '
                    'eager call of the exported graph.')
            elif program.compiler == 'tvm':
                return self._tvm(program, example_inputs, mode, validation_trials)
            elif program.compiler != 'eager':
                raise UnsupportedReconstruction(
                    f'{program.compiler}: captured source/AOT graph is not generated kernel '
                    'semantics; no lowering of its kernels (fusion, reduction order, FMA '
                    'contraction, library tactics) is available')
            program = program.source_model
        if format == 'aten':
            if not isinstance(program, torch.fx.GraphModule):
                raise UnsupportedReconstruction('aten requires an explicit FX GraphModule')
            return ReconstructedProgram(program, example_inputs, source='explicit eager ATen',
                                        mode=mode, lowerings=lowerings)
        if not isinstance(program, torch.nn.Module) or hasattr(program, '_orig_mod'):
            raise UnsupportedReconstruction('requires an eager nn.Module')
        # Strict export, with no decompositions or constant folding. A captured graph
        # is conditional on export preserving source behavior, not proof of Python.
        try:
            # Strict export traces with Dynamo, whose per-code-object cache fills up when many
            # programs are exported in one process; every export here starts from a clean one.
            torch._dynamo.reset()
            ep = torch.export.export(copy.deepcopy(program), tuple(example_inputs), strict=True)
        except Exception as exc:
            raise UnsupportedReconstruction(f'strict export failed: {exc}') from exc
        if any('MUTATION' in str(s.kind) for s in ep.graph_signature.output_specs):
            raise UnsupportedReconstruction('input/buffer mutation is unsupported')
        result = ReconstructedProgram(ep.module(), example_inputs, source=source, mode=mode,
            lowerings=lowerings,
            assumptions=['Strict export faithfully represents the source for the fixed input domain; '
                         'Python side effects and export specializations are not proved.']
            + assumptions)
        evidence = result.compare(reference, [example_inputs])
        if not evidence['bitwise_equal']:
            raise UnsupportedReconstruction(
                'export/reconstruction differs in output bits on example'
                + (' (NaN sign/payload only)' if evidence.get('nan_bits_only') else ''))
        return result


def _tvm(self, program, example_inputs, mode, validation_trials):
    """A `compilers.compile_tvm` program: the Relax `main` of the lowered module gives the
    graph, and every `call_tir` is executed from its scheduled TIR (`dlclv.tir_lowering`)."""
    from .lowering import LoweringFailed
    from .tir_lowering import TirLowering, build_tvm_graph
    if mode != 'primitive':
        raise UnsupportedReconstruction('a tvm program has no eager kernels to preserve; use '
                                        "mode='primitive'")
    try:
        gm = build_tvm_graph(program, example_inputs)
    except LoweringFailed as exc:
        raise UnsupportedReconstruction(f'tvm: {exc}') from exc
    result = ReconstructedProgram(
        gm, example_inputs, mode=mode, lowerings=[TirLowering(program, validation_trials)],
        source='TVM lowered module: Relax main (call_tir bindings) and scheduled TIR functions, '
               f'options {program.options}',
        assumptions=['The lowered module kept by compile_tvm is the one TVM generated code '
                     'from; its remaining passes (dataflow lowering, memory planning, VM code '
                     'generation) do not change arithmetic.'])
    evidence = result.compare(program, [example_inputs])
    if not evidence['bitwise_equal']:
        raise UnsupportedReconstruction(
            'tvm: reconstruction differs from the compiled program in output bits on the example'
            + (' (NaN sign/payload only)' if evidence.get('nan_bits_only') else ''))
    return result


ReconstructionFrontend._tvm = _tvm


def reconstruct(program, example_inputs, *, format='pytorch', mode='primitive', lowerings=(),
                probe_kernels=False, kernel_probe=True, validation_trials=5):
    """Build a guarded explicit program A' for `program` at the example's shapes.

    mode='primitive' (default): every call must be an explicit primitive or be lowered to
        explicit primitives; anything opaque is rejected.
    mode='eager-kernel': calls in `EAGER_KERNELS` that no lowering takes are preserved as the
        same opaque library kernels (reported in `report['preserved_kernels']`).

    lowerings: `dlclv.lowering.Lowering` objects tried on every call, in order.
    probe_kernels: also try to lower PyTorch eager `linear` and one-dimensional `sum` calls
        from behaviour (FPRev + bitwise validation; evidence level "validated").
    kernel_probe / validation_trials: for `triton` / `tilelang` compiled programs, check the
        kernel order on the device (FPRev chain, FMA probe, bitwise validation per bmm shape).

    The result is compared with the original on the example input. It is never a certificate
    of equivalence for all inputs: see `report` for assumptions and evidence levels.
    """
    return ReconstructionFrontend().reconstruct(
        program, example_inputs, format=format, mode=mode, lowerings=lowerings,
        probe_kernels=probe_kernels, kernel_probe=kernel_probe,
        validation_trials=validation_trials)
