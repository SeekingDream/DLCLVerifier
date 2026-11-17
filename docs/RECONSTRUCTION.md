# Reconstruction frontend

`dlclv.reconstruction` rebuilds the computation a tensor program actually performs as an
explicit eager PyTorch program A': one ATen call per IEEE operation, cast or data movement.
The target is **bitwise** equality with the original A under a stated execution
configuration, not numerical closeness. It is separate from the drift certifier and never
treats a small bound as equality.

```python
from dlclv.reconstruction import reconstruct

# A program on the repository's Triton path (compilers.py, kernels.py), on the GPU.
compiled = utils.load_compiler("triton", split_k=4)(model, inputs)
aprime = reconstruct(compiled, inputs)                        # or mode="eager-kernel"
aprime(*inputs)                                               # same bits as compiled(*inputs)
print(aprime.code)       # ATen graph in dependency order + the arithmetic of each lowered kernel
print(aprime.report)     # evidence level per kernel, assumptions, executed primitive ops
aprime.compare(compiled, more_inputs)                         # finite-sample evidence only
```

## What A' is made of

Every node of A' is one of three things. The report says which.

| Node | What runs | Listed in |
| --- | --- | --- |
| Explicit primitive (`PRIMITIVES`) | one `add`/`sub`/`mul`/`div`/`neg`, cast, or indexing/reshape call | the graph (`code`) |
| Lowered kernel | a function built only from `dlclv.primitives`, replacing a library or DSL kernel call | `report["lowered_kernels"]` |
| Preserved kernel (`mode="eager-kernel"` only) | the same opaque library call as A; **not reconstructed** | `report["preserved_kernels"]` |

`mode="primitive"` (default) admits only the first two and rejects everything else.
`mode="eager-kernel"` additionally keeps the calls in `EAGER_KERNELS` (any ReLU, pooling,
batch norm, GEMM, convolution or reduction that no lowering took) as they are.

No decomposition, constant folding or reassociation is performed, and no generic
`torch.sum`/`torch.matmul` is ever substituted for a kernel. An in-place call that newer
`torch.export` keeps (`relu_`, `out += identity`) runs as its out-of-place overload, and only
when the mutated value is a fresh intermediate with no other user.

## Primitives (`src/dlclv/primitives.py`)

| Primitive | Arithmetic |
| --- | --- |
| `fma32(a, b, c)` | float32 fused multiply-add with one rounding |
| `chain_dot(a, b, ks, fp)` | `acc = +0; for k in ks: acc = fma(a_k, b_k, acc)` (or `acc + fl(a_k * b_k)`) |
| `tree_dot(tree, a, b, fused, zero_init, bias, zero_root)` | products accumulated along an arbitrary binary tree; per product: fused into its node or rounded first |
| `tree_sum(tree, x, dim, zero_init)` | one addition per tree node |
| `im2col(...)` | `F.unfold` as zero padding and strided slices |
| `two_sum(a, b)` | error-free addition |

**FMA.** PyTorch has no fused multiply-add, and `(a.double()*b.double()+c.double()).float()`
rounds twice. `fma32` computes the product exactly in float64 (48 bits fit), adds with TwoSum
to get the exact error, moves the float64 sum one ulp towards the exact value when it is
inexact and its last bit is even (round to odd), and casts. Float64 has more than two extra
bits, so the cast rounds the exact `a*b+c` correctly (Boldo and Melquiond 2008). It is checked
against exact rational arithmetic, including the ties where the float64 route is wrong, e.g.
`fma(24 + 3·2^-13, 8 - 2^-13, 2^30)` is `2^30 + 128`; two roundings give `2^30 + 256`.

**Signed zeros** are part of the bits and are modelled: an accumulator that starts at `+0`
turns an exact `-0` product into `+0`; a product that underflows keeps `-0`; a kernel step on
masked (zero) operands, `fma(+0, +0, acc)`, turns `-0` into `+0`.

**`PrimitiveGuard`.** Lowered kernels run under a dispatch guard that raises on any ATen call
outside the allowlists (`ROUNDING_OPS`, `EXACT_FAMILIES`). A lowering therefore cannot call a
library reduction, `addcmul`, a scaled add, or move data between devices. Besides `+ - * /`
and casts, the rounding list has `sqrt` (a basic IEEE operation, correctly rounded on x86 and
in CUDA), which only the TIR interpreter uses; the exact list includes gather and scatter by
index and integer index arithmetic. The operations that
did run are counted in `report["lowered_primitive_ops"]`.

## Kernel lowerings (`src/dlclv/lowering.py`)

A lowering inspects one call with its concrete operands and returns explicit arithmetic plus
its evidence, or declines. It never guesses.

| Lowering | Kernel | Where the order comes from | Level |
| --- | --- | --- | --- |
| `DslGemmLowering` | Triton / TileLang GEMM of `kernels.py`, as called by `compilers.GemmMode`: `linear`, `matmul`, `mm`, `bmm`, `addmm`, `conv1d`/`conv2d` (im2col), split-K | kernel source, then checked on the device | `known` (or `source` with `kernel_probe=False`) |
| `ProbedLinearLowering` | PyTorch eager `linear` (BLAS, cuBLAS), float32 | behaviour only | `validated` |
| `ProbedConvLowering` | PyTorch eager `conv1d` / `conv2d` (cuDNN, CPU), float32 | behaviour only | `validated` |
| `ProbedSumLowering` | PyTorch eager `sum` / `mean` over given dimensions, `adaptive_avg_pool2d` to 1x1 | behaviour only | `validated` |
| `ClampLowering` | `relu`, `relu6`, `hardtanh`, `clamp` with scalar bounds | candidates, selected by behaviour | `validated` |
| `MaxPoolLowering` | `max_pool2d` without `ceil_mode` | kernel source on CUDA, then behaviour | `known` on CUDA, `validated` on CPU |
| `BatchNormLowering` | inference `batch_norm` with affine parameters, float32 | candidates, selected by behaviour | `validated` |

All but the first are library kernels and are enabled with `reconstruct(..., probe_kernels=True)`.
The last three are in `src/dlclv/eager_lowering.py`. With them a CNN on the Triton or TileLang
path (convolutions and the classifier on the DSL kernel, everything else a PyTorch library
kernel) is reconstructed in `mode="primitive"` with nothing preserved.

### Triton and TileLang GEMM

Read from the lowered kernel, not from the operator graph:

- **Which calls ran on the kernel**: the path's own call trace (`meta["kernel_trace"]`),
  matched to graph nodes in program order by the kernel call's operand shapes. A call that
  fell back to PyTorch is left alone. A kernel call without a node, or a node without a call,
  is an error.
- **Operands**: rebuilt as `GemmMode` builds them (reshapes, expansion, im2col for convolution).
- **Accumulation**: per output element one float32 accumulator, `acc = +0`, then
  `acc = fma(a[m,k], b[k,n], acc)` for k increasing. This is `tl.dot(a, b, acc)` over K blocks
  in Triton and `Cl[i,j] += As[i,k] * Bs[k,j]` in TileLang.
- **Masked tail**: when the kernel runs steps past the end of the k range on zero operands, one
  `fma(+0, +0, acc)` is applied.
- **Split-K**: the k ranges follow each path's own formula (`lowering.k_ranges`); each range has
  its own accumulator; the partial sums are stacked and added in the order of the library
  `sum(1)` the path calls, which is probed (below).
- **Bias**: a separate float32 addition after the kernel.

With `kernel_probe=True` (default) the kernel itself (`meta["bmm"]`) is consulted before the
lowering is accepted: FPRev masks must reveal exactly the tree above, the two-term product probe
must report FMA (without split-K; with split-K two products can land in different partial
sums, so fusion is checked by the validation only), and the explicit function must reproduce
the kernel bit for bit on `validation_trials` adversarial inputs at every distinct kernel
shape of the program.

### Library kernels, from behaviour

`ProbedLinearLowering`, `ProbedConvLowering` and `ProbedSumLowering` have no source to read.
They use FPRev (arXiv:2411.00442) to rebuild the summation tree at the exact shapes, a product
probe for which products are fused, signed-zero probes for `+0` accumulators, and then bitwise
validation at every output element. If any step fails the call is not lowered; the reason is
in `report["lowering_failures"]` and in the rejection message.

- **Sums and averages.** The reduced dimensions are moved last and flattened row-major; the
  tree is over those leaves. Positions that use different orders (vector lanes and the scalar
  tail) get one tree per position class when there are at most 64 leaves. `mean` and
  `adaptive_avg_pool2d(x, 1)` are that sum, `+0` where the accumulator starts at zero, and one
  division by the number of elements; the division form is validated, not assumed.
- **Convolution.** The model is an explicit im2col (padding contributes products with `+0`)
  and, per output element, one tree over the `K = C/groups * kh * kw` products of its output
  channel, the same tree at every position, then the bias. Masks go in the weights and are
  read at the output positions whose window lies inside the input; all of them must agree.
  Validation covers every output element, borders included, with random weights.
  cuDNN's implicit-GEMM algorithms (small channel counts, strided, 1x1, grouped and dilated
  layers on the tested GPU) are a k-sequential FMA chain and are lowered. The algorithm cuDNN
  picks for 3x3 stride-1 layers with 16 or more channels is **not** a float32 FMA tree of the
  products (FPRev finds multi-term nodes, or two unrounded products under one node), so those
  calls are refused. A layer whose input is smaller than its kernel has no interior position
  and is refused too. `cudnn.benchmark = True` is refused: the algorithm would be chosen by
  timing.

### Library kernels without an accumulation order

ReLU, clamps, max pooling and inference batch norm have no reduction, but the operator's
definition still does not fix their bits. Each lowering states a few explicit candidate
programs, runs the kernel on adversarial inputs (plus all `-0`, all `+0`, and for clamps one
ulp either side of each bound), discards every candidate that differs in one bit, and lowers
the call only if one survives. Candidates that survive together are listed in the evidence as
`indistinguishable`.

| Kernel | What the kernel decides | Observed with PyTorch 2.6 |
| --- | --- | --- |
| `relu`, `relu6`, `hardtanh`, `clamp` | the sign of a zero result: on a tie the kernel returns its operand or the bound | `relu(-0.0)` is `-0` on x86 CPU and `+0` on CUDA |
| `max_pool2d` | which of two equal maxima is returned (`+0` against `-0`) | first maximum of a row-major scan from `-inf`, on both; padding is `-inf` |
| `batch_norm` (inference) | association, folding to `x*alpha + beta`, and which multiply-add is fused | CUDA, native and cuDNN: `fma(fl(fl(x - mean) * w), invstd, b)`. x86 CPU: `fma(x, alpha, fma(-mean, alpha, b))` with `alpha = fl(invstd * w)` |

The textbook `(x - mean) / sqrt(var + eps) * w + b` matches neither batch-norm kernel.

`invstd` is `rsqrt(running_var + eps)`, a library function that is not correctly rounded on
every device and has no explicit primitive. It depends only on the frozen running variance,
never on the program input, so it is **read from the kernel** once (the call with mean 0,
weight 1, bias 0 on ones returns it exactly in every candidate form) and kept as a
per-channel constant. The lowered call checks on every execution that the statistics and
affine parameters are the tensors it was built for (`ExecutionMismatch` otherwise). This is
the one place where a constant of A' is not recomputed from explicit primitives.

## TVM: reconstruction from the lowered TIR (`src/dlclv/tir_lowering.py`)

`compilers.compile_tvm` (compiler 6, `tvm`) imports the strict `torch.export` graph into TVM
Relax, legalizes every operator to a TIR function, schedules each loop nest for the GPU with
`dlight`, and lets TVM generate and compile CUDA kernels. No cuDNN or cuBLAS call is involved.
It keeps the module TVM generates code from in `compiled.meta["tvm_lowered"]`.

That module is the lowered representation the reconstruction reads. It never looks at an
operator name:

- **Graph.** The Relax `main` is a list of `call_tir` bindings. `build_tvm_graph` turns it
  into the ordered graph of A' (inputs, constants as frozen buffers, one node per `call_tir`).
- **Kernels.** A scheduled TIR function is an explicit loop nest over buffers. `run_primfunc`
  interprets it statement by statement with `dlclv.primitives` operations: loads and stores
  are gathers and scatters by computed index, `+ - * /`, `sqrt`, comparisons and selections
  are single calls, and a serial loop is a Python loop in increasing order. So a convolution
  is `acc = +0; for c, kh, kw: acc = acc + pad(x)[...] * w[...]` on one float32 accumulator
  per output element, in exactly the loop order of the TIR; batch norm is the expression TVM
  wrote, `(x - mean) / sqrt(var + eps) * w + b`; max pooling starts from `-FLT_MAX` (not
  `-inf`), as its TIR says.
- **GPU-thread loops** run as one tensor operation over all their iterations. That equals
  running the iterations one by one only if none reads what another wrote. It is checked when
  the lowering is built: within one thread region a block writes distinct elements, and a
  read of a buffer written in the region is the block's own element. Out-of-bounds loads
  outside their `if_then_else` guard are rejected.

Two things the TIR does not determine are stated as a `Model` and checked per kernel:

| Convention | Candidates | Observed (nvcc 12.8, RTX 6000 Ada) |
| --- | --- | --- |
| FMA contraction of `a * b + c` by the CUDA compiler | contracted to one `fma32`, or not | contracted |
| `max(a, b)` / `min(a, b)` on equal operands (`+0` against `-0`) | first operand, second operand, or `+0` above `-0` whatever the order | `+0` above `-0` (IEEE 754-2019 `maximum`) |

The contraction rule is syntactic: an addition or subtraction with exactly one multiplication
as a direct operand is one FMA. An addition of two products is ambiguous and is refused. Each
TIR function is built on its own with the same code generator and nvcc, and the candidates
are run against it on the call's own operands, on adversarial operands, and (for `max`/`min`)
on zeros of random sign, all `-0` and all `+0`. One candidate must reproduce the kernel bit
for bit. A kernel whose inputs never tie takes the convention the program's other kernels
showed; the candidates it could not tell apart are listed as `indistinguishable_models`.
`evidence["kernel_source_identical_to_program"]` records that every CUDA kernel of the
standalone build appears verbatim in the program's CUDA source.

Evidence level `known`: order and operations come from the lowered IR (static), the two
compiler conventions are corroborated on the device (finite).

Not supported, and refused with the reason:

- `schedule="reduction"`: dlight's cross-thread reduction blocks (`T.init`) are lowered to
  their final order by a later TIR pass, after the module that is kept.
- `schedule="default"` (tiled GEMM with shared-memory staging and vectorized loops): the TVM
  bundled with TileLang 0.1.15 cannot build it for CUDA at all.
- Transcendental intrinsics (`exp`, `tanh`, `erf`, ...), buffers that are not float32,
  symbolic shapes, `call_tir` with packed shape arguments, Relax control flow.

## Supported scope

| Representation | Reconstruction |
| --- | --- |
| Eager `nn.Module`, repository `eager` wrapper (CPU or CUDA) | Strict `torch.export`; primitives; with `probe_kernels=True` the library kernels in the table above; otherwise rejected or preserved by mode |
| Explicit eager ATen `GraphModule` (`format="aten"`) | Same, the graph itself is A |
| `triton` / `tilelang` compiled program | GEMM-shaped kernel calls lowered to explicit FMA chains; the rest as for eager. CNNs with batch norm, ReLU, max pooling and global average pooling reconstruct fully in primitive mode |
| `torch_compile` (TorchInductor) | **Unsupported.** The captured AOT graph precedes fusion and code generation; the generated kernels are not parsed |
| ONNX file, `onnxruntime_gpu` | **Unsupported.** No lowering of ORT, cuBLAS or cuDNN kernels |
| `tensorrt` | **Unsupported.** Engine tactics are opaque |
| `tvm` compiled program (`compilers.compile_tvm`, `schedule="fallback"`, `fuse` on or off) | Every kernel executed from its scheduled TIR; primitive mode only, nothing preserved |
| `tvm` with `schedule="reduction"`; TIR with transcendental intrinsics, tiled or vectorized schedules, dtypes other than float32 | **Unsupported**, with the reason |
| Other formats | **Unsupported**; unknown formats fail closed |
| `TensorGraph`, an FPRev tree on its own | Not accepted as implementation evidence |

Fixed per reconstruction: input and intermediate shapes, dtypes, strides, storage offsets, and
the weights (a frozen snapshot). Tensors are strided float16, bfloat16, float32 or float64 on
one device. The FMA primitive and the kernel lowerings are float32; `ProbedSumLowering`,
`ClampLowering` and `MaxPoolLowering` also handle float64.

**Excluded inputs.** Inputs for which A or A' produces a NaN. A generated NaN has a
hardware-specific sign and payload (x86 SSE produces a negative quiet NaN, GPUs a positive
one), so NaN bits are outside every claim. `compare` reports `nan_bits_only` when the only
differences found were between two NaNs. Infinities, subnormals and signed zeros are in scope.

Unsupported in primitive mode and never approximated: transcendental functions, scaled adds,
`addcmul`, tensor-core MMA (TF32, fp16), random operations, mutation of inputs or buffers,
data-dependent control flow, training-mode batch norm, average pooling other than global,
`max_pool2d` with `ceil_mode` or indices, and any reduction, GEMM or convolution without an
accepted lowering.

## Execution assumptions

Captured when A' is built and checked on every call (`ExecutionConfig`); a change raises
`ExecutionMismatch`:

- PyTorch version and build string, host platform, device type;
- thread counts, default dtype, autocast off, MKLDNN flags, deterministic-algorithm settings,
  float32 matmul precision;
- on CUDA: GPU model, CUDA and cuDNN versions, TF32 flags, cuDNN `enabled`, `benchmark`,
  `deterministic`;
- gradual underflow: measured on the device (a subnormal survives `x * 1`).

Assumed, not checked:

1. Each explicit primitive is the correctly rounded IEEE 754 operation in its dtype (round to
   nearest, ties to even), on CPU and on CUDA, and separate eager calls are never contracted.
   The rounding mode cannot be read from PyTorch.
2. Strict `torch.export` represents the source module on the fixed input domain. Agreement on
   the example cannot validate Python specializations for other inputs.
3. For a kernel path: it runs the source model's Python with only the traced GEMM-shaped calls
   replaced.
4. Per lowering, the statement in `report["assumptions"]` (below).
5. Preserved kernels are deterministic for the fixed layouts and settings.

Custom dispatch modes, tensor subclasses, concurrent state changes and compiler wrappers are
outside the contract. The contract concerns output bits, not aliasing, autograd or side effects.

## Guarantees and evidence

Every result has `status="conditional"` and `universal_certificate=False`. Nothing here is a
machine-checked proof, and no number of passing samples changes that.

| Claim | Basis | Holds for |
| --- | --- | --- |
| `fma32` equals the IEEE fused operation | Proof on paper (exact product, TwoSum, round to odd), given assumption 1; tested against rational arithmetic | all finite float32 operands |
| A' of a primitive-only graph equals that graph | Induction over identical ordered calls on identical bits, given assumption 1 | all admitted inputs |
| A lowered Triton/TileLang call equals the kernel (level `known`) | Kernel source gives a value-independent loop; the DSL compiler's lowering of it to one FMA chain is **assumed**, corroborated by FPRev, the FMA probe and bitwise validation on this GPU | all admitted inputs, **if** the assumption holds; the checks themselves are finite |
| Same, level `source` | Kernel source only | as above, uncorroborated |
| A lowered TVM kernel equals the compiled kernel (level `known`) | The scheduled TIR gives every operation and the order of every reduction. **Assumed:** TVM's code generator and nvcc implement the TIR as written, the kept module is the one code was generated from, and the two conventions above hold for all inputs; they are corroborated by bitwise validation of every kernel on this GPU | all admitted inputs, **if** the assumptions hold; the checks themselves are finite |
| A lowered library call equals the kernel (level `validated`) | Dynamic only: FPRev assumes a fixed, value-independent order and does not prove it | the tested inputs; for all inputs it is an **assumption** |
| A clamp, max-pool or batch-norm lowering | A finite set of stated candidates, all but one (or a bitwise-indistinguishable group) eliminated on adversarial inputs. Max pooling on CUDA is also read from the kernel source | the tested inputs; that the kernel is one of the candidates at every element and value is an **assumption** |
| The batch-norm `invstd` constants | Read from the kernel, not reconstructed | the frozen statistics, checked on every call |
| The split-K combine order | Dynamic only (it is a library `sum`) | as the previous row |
| A preserved kernel | It is the same call | conditional on determinism and identical dispatch |
| `compare(...)` | Finite samples | exactly the samples compared |

`report["lowering_levels"]` summarizes the levels a reconstruction relies on. A reader who
accepts only static evidence should treat `validated` lowerings and every split-K program as
unknown order, as the certifier's strict policy does (`docs/ABSTRACT_INTERPRETATION.md`, 4.2).

## Remaining limitations

- **No machine-code evidence.** The FMA chain of the DSL kernels is not read from SASS or PTX.
  A compiler version that emits another order would be caught only by the device checks.
- **Shapes are fixed.** A lowering is established per kernel shape; other shapes need a new
  reconstruction, and the kernel's order may differ there.
- **Probes can fail.** BLAS kernels whose order depends on the output position are not
  lowered (observed on CPU for several batch sizes), nor is a CPU `sum` over a middle
  dimension whose order no per-position tree reproduces. Elementwise transcendental functions,
  windowed average pooling and training-mode batch norm have no lowering; in primitive mode
  programs that use them are rejected.
- **cuDNN convolution is only partly covered.** The 3x3 stride-1 layers that dominate ResNet
  style networks run an algorithm with no float32 FMA-tree model and are refused, so an
  *eager* or cuDNN-backed CNN is not reconstructed end to end; on the Triton and TileLang
  paths the same layers run on the DSL kernel and are. This is also what blocks TorchInductor
  and ONNX Runtime on CNNs: both call cuDNN for convolution (and ONNX Runtime's default
  `EXHAUSTIVE` algorithm search picks by timing), so an adapter for their graphs would stop at
  the same kernels.
- **Non-contiguous operands** of the probed reductions and batch norm are not lowered.
- **TVM coverage.** Only the serial-reduction schedules are interpreted, and the CUDA source
  TVM emits is compared as text, not the machine code nvcc makes from it. The interpreter is
  slow: one gather, one FMA and one scatter per reduction step.
- **Cost.** A lowered GEMM runs K steps of about 14 elementwise float64 calls. ResNet18 at
  64×64 takes minutes to build and validate; `tree_dot` keeps up to K/2 partial tensors.
- **Trace alignment.** Kernel calls are matched to graph nodes by operand shapes in program
  order. A GEMM that reaches PyTorch without a Python-level call (for example inside `einsum`)
  has no trace record and makes the reconstruction fail.
- **Cross-device claims.** A' runs on the device of its inputs. The explicit arithmetic gives
  the same bits on CPU and CUDA in the tests, which supports assumption 1 but does not prove it.
- **Export.** Whatever `torch.export` specializes is specialized in A'.

## Extension contract

`reconstruct(..., lowerings=[...])` accepts `Lowering` objects; `lower(node, target, args,
kwargs)` returns a `LoweredOp(fn, kernel, description, level, evidence, assumptions)` or `None`.
A new lowering must read the lowered representation or generated kernel, state its level
truthfully, and build `fn` from `dlclv.primitives` only (the guard enforces the last part).
Hardware semantics with no primitive, such as tensor-core accumulation, need a new validated
primitive first.

`ReconstructionFrontend.register(format, adapter)` adds a whole-program adapter for another
representation; it must return a `ReconstructedProgram`. Registration proves nothing about
the adapter.

## Tests

```bash
pytest tests/test_primitives.py tests/test_lowering.py tests/test_eager_lowering.py \
       tests/test_reconstruction.py tests/test_reconstruction_utils.py      # CPU (+ CUDA if present)
CUDA_VISIBLE_DEVICES=7 pytest tests/test_lowering_gpu.py tests/test_tvm_gpu.py   # real kernels
CUDA_VISIBLE_DEVICES=7 python test/test_reconstruction_models.py --tasks 2 \
    --device cuda --compiler triton --image-size 64 --samples 4 --image-edges \
    --output results/reconstruction_triton_resnet18_64.json
```

- `test_primitives.py`: `fma32` against exact rational arithmetic on random values, wide
  exponents, cancellation, subnormals, overflow, signed zeros and float32 ties where two
  roundings are wrong; chains and trees against a rational emulator; order and fusion are
  shown to change the bits; the guard rejects non-primitive calls.
- `test_lowering.py` (CPU): the kernel is a loop-by-loop transcription of `kernels.py`
  evaluated in rational arithmetic behind the repository's `GemmMode`. Covers trace matching,
  im2col, masked tails, split-K, the report, and rejection of a kernel that rounds products
  separately or visits K blocks in reverse. Library `sum` and `linear` are probed for real
  and skip where this machine's kernel has no establishable order.
- `test_eager_lowering.py` (CPU, and CUDA when present): ReLU, `relu6`, `hardtanh`, `clamp`,
  max pooling, batch norm, `mean`, global average pooling, sums over several dimensions and
  convolutions, each compared bitwise on adversarial inputs and on all `-0` / all `+0`. Also
  checks that the two clamp tie conventions and the two max-pool scan orders differ exactly on
  signed zeros, that the textbook batch-norm formula gives other bits than the kernel, that
  changed batch-norm statistics are caught, and that unsupported forms are refused.
- `test_tvm_gpu.py`: an MLP and a residual CNN block compiled by TVM with and without operator
  fusion, rebuilt from their TIR and compared bitwise; the TVM path differs in bits from eager
  PyTorch and from the same TIR read without FMA contraction; the `max` conventions differ on
  signed zeros and the kernel's is selected; unsupported schedules, modes and intrinsics are
  refused.
- `test_lowering_gpu.py`: the real Triton and TileLang kernels (K with and without a masked
  tail, split-K 1 to 4, a CNN with strided, grouped and dilated convolutions), cuBLAS `linear`
  and CUDA `sum`, on inputs with wide exponents, cancellation, subnormals, underflow and signed
  zeros. Also checks that the explicit program gives the same bits on the CPU.

### Recorded results

On server3 (RTX 6000 Ada, PyTorch 2.6.0+cu124, Triton 3.2.0, TileLang 0.1.15), 2026-11-16:

- The reconstruction suites above passed with one skip (CPU BLAS `linear` for one shape:
  order not established).
- Pretrained ResNet18 on the Triton path at 64×64, `mode="eager-kernel"`: 21 kernel calls
  lowered at level `known`, bitwise agreement on 8 inputs (4 random, black, gray, white,
  checkerboard). Batch norm, ReLU, max pooling and adaptive average pooling were preserved, not
  reconstructed. See `results/reconstruction_triton_resnet18_64.json`.

- Pretrained ResNet18 at 64×64 in `mode="primitive"` with `probe_kernels=True`, on the Triton
  path and on the TileLang path: 60 kernel calls lowered each (21 GEMMs at level `known`, the
  batch norms, ReLUs, max pooling and global average pooling from behaviour at level
  `validated`, max pooling at `known`), **nothing preserved**, bitwise agreement on the same 8
  inputs. See `results/reconstruction_triton_resnet18_64_primitive.json` and
  `results/reconstruction_tilelang_resnet18_64_primitive.json`.
- `tests/test_eager_lowering.py` and `tests/test_tvm_gpu.py` passed on CPU and GPU; skips are
  kernels whose order the probe could not establish on that machine (reported, not lowered).

On the Mac (PyTorch 2.2.2, CPU), eager SqueezeNet 1.1, MobileNetV2 and ResNet18 in
`eager-kernel` mode with nothing lowered:
`results/reconstruction_utils_eager_kernel_cpu.json` (14 inputs each, from the earlier
revision of this frontend).

All of these are finite-sample agreement under the assumptions above.
