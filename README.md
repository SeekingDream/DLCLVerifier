# DLCLVerifier

Certifies limits on silent numerical drift when a tensor program is optimized by a DL compiler
or kernel DSL. The repository has the models, compilers and differential testing used for the
evaluation, and a prototype certifier (`src/dlclv`, section "Certifier"). No baseline is
integrated yet.

## Reconstruction frontend

`dlclv.reconstruction` rebuilds what a program actually computes as explicit eager PyTorch
arithmetic (one call per addition, multiplication, cast or indexing step) with the goal of
bitwise equality with the original, not closeness.

```python
from dlclv.reconstruction import reconstruct
aprime = reconstruct(utils.load_compiler("triton")(model, inputs), inputs, probe_kernels=True)
print(aprime.code, aprime.report)      # primitive mode: no opaque kernel left, or it raises
```

- Triton and TileLang GEMM kernels (`kernels.py`) are lowered from the kernel source to one
  FMA chain per output element in k order, with split-K partial sums, masked tails and signed
  zeros; the kernel is then checked on the GPU (FPRev, FMA probe, bitwise validation).
- A float32 FMA with a single rounding is built from float64 operations (`primitives.fma32`).
- PyTorch library kernels are rebuilt from behaviour with `probe_kernels=True`: `linear`,
  `sum` / `mean` / global average pooling, cuDNN's GEMM-style convolutions, ReLU and clamps,
  max pooling and inference batch norm. That evidence is dynamic only.
- A CNN on the Triton or TileLang path (pretrained ResNet18 included) is reconstructed with
  no opaque kernel left.
- A `tvm` program is reconstructed from its lowered representation: the Relax `main` gives
  the graph and every kernel is executed from its scheduled TIR, statement by statement
  (`tir_lowering.py`). The order of every reduction is read from the TIR; FMA contraction and
  the `max`/`min` tie convention are checked against each compiled kernel.
- Everything else is rejected in `mode="primitive"` or kept as the same opaque library call in
  `mode="eager-kernel"`. cuDNN's 3x3 stride-1 convolution algorithm has no float32 FMA-tree
  model and is refused; TorchInductor, ONNX Runtime and TensorRT programs are unsupported.
- Results are conditional on stated assumptions and never a certificate for all inputs.

See [scope, assumptions and evidence levels](docs/RECONSTRUCTION.md).

## Scope (drift certifier)

The target is drift caused by **non-associativity of floating-point addition**, such as
`(a+b)+c ≠ a+(b+c)`: the same operands summed in a different order. FMA contraction, precision
changes, merged constants and elementary-function replacement are out of scope
(`docs/SPECIFICATION.md`). The bound to beat is the tree-depth bound for both orders:

```
|ŝ_T₁ − ŝ_T₂| ≤ Σ [γ(d_i) + γ(d′_i)]·|x_i|     γ(k) = ku/(1 − ku)
```

## Setup

```bash
conda create -n DLCLVerifier python=3.11 -c conda-forge --override-channels
conda activate DLCLVerifier
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-eval.txt
```

Gated sources need `huggingface-cli login`: ImageNet (`ILSVRC/imagenet-1k`) and the Llama models.
ImageNet can also come from a local folder: `DLCLV_IMAGENET_DIR=/path/to/imagenet/val`.

## Usage

```python
import utils

model, loader, domain = utils.load_data_model(task_id=6, data_num=100)   # distilbert / sst2
inputs, labels = next(iter(loader))

compiled = utils.load_compiler("tensorrt")(model, inputs)                 # called like the model
max_err = utils.diff_testing(model, compiled, test_num=256, input_domain=domain)
```

- `load_data_model(task_id, data_num)` returns the model (eval mode, float32), a DataLoader of
  exactly `data_num` seeded samples yielding `(inputs, labels)`, and the `InputDomain`. Call the
  model as `model(*inputs)`.
- `load_compiler(compiler, **options)` returns `compile_fn(model, example_inputs)`. Options are
  each path's knobs, e.g. `load_compiler("triton", split_k=4)`.
- `diff_testing(prog_1, prog_2, test_num, input_domain)` runs both programs on seeded random
  points of the domain and returns the maximum absolute output error. `return_details=True` adds
  the relative error and the witness batch. Sampling gives a lower bound, never a certificate.

## Certifier

`src/dlclv` bounds `|prog_1(x) - prog_2(x)|` over a whole input domain, not just sampled points:

```python
from dlclv.certifier import TensorProgramCertifier
cert = TensorProgramCertifier(model, compiled, domain, {"epsilon": 1e-4,
                                                         "assume_sum_of_products_conv": True})
cert.match().summary()          # which operations correspond (fusions, unmatched regions)
res = cert.certify()            # verified | bounded | inconclusive (with a reason)
print(res.summary(), res.assumptions)
```

- Two references are supported:
  - `{"reference": "eager"}`: bounds `|prog_2 - eager model|`. Program 1 must be the eager
    model. The accumulation order of each of its kernels is either known or rebuilt from the
    kernel by a probe (`probe_eager_orders`, run automatically); an operation whose order
    cannot be established is bounded over every order and listed in `result.notes`
    (`require_known_orders=True` returns `inconclusive` instead).
  - `{"reference": "exact"}`: bounds `|prog_2 - F|`, where `F` is the eager model's graph in
    exact real arithmetic. It does not depend on the eager kernels, so it stays finite where
    their orders are unknown (convolutions).
- PyTorch eager is the reference: `prog_1` is the eager model in every pair.
- Graphs (`src/dlclv/frontends.py`) come from `torch.export` for the eager model, ONNX files,
  ONNX Runtime's optimized graph, and the `triton`, `tilelang`, `onnxruntime_gpu` and
  `torch_compile` paths. TensorRT is not extractable yet.
  - `triton` / `tilelang`: an operation is labelled with the kernel only when the path's own
    call trace says the kernel ran it; a call that fell back to PyTorch keeps the eager label.
  - `torch_compile`: the graph is the ATen graph AOTAutograd handed to Inductor, captured during
    the compilation that built the artifact. Inductor's kernels (fusion, reduction order, FMA
    contraction) have no floating-point model yet, so `certify()` returns `inconclusive`
    (`unsupported_path`) and claims no bound.
  - Every extraction runs the real program once and compares it with the extracted graph
    (`graph.meta["fidelity"]`); a graph that does not reproduce its program is an error.
  - A value computed from constants is folded only when its bits cannot depend on the
    implementation. Random values and programs that mutate state are rejected or kept as nodes.
- The two graphs are matched into units of corresponding operations, e.g. add+ReLU vs a fused
  kernel.
- The difference is bounded per element by differential back-substitution (`method="backsub"`,
  the default). Accumulations use a known order (Triton/TileLang kernels, or FPRev evidence) or
  a bound over every admissible order, with node magnitudes taken from signed partial sums.
- `cert.probe_eager_orders()` rebuilds the accumulation tree of the eager linear layers from
  the kernel itself (FPRev masks, then bit-exact emulation) and uses it in place of the
  any-order bound. The evidence is dynamic only, so each order used is listed in
  `result.assumptions`.
- `utils.search_max_diff` searches the domain for a large difference (box vertices, gradient
  signs, hill climbing): a stronger lower bound than uniform sampling, never a certificate.
- `guarantee="formal"` is a worst-case bound; `"probabilistic"` assumes independent mean-zero
  rounding errors and never yields `verified`.
- Design and current precision: `docs/ABSTRACT_INTERPRETATION.md`. On ResNet18 at one sample:
  pairs that agree bitwise certify at 0; against exact arithmetic the Triton and TileLang
  programs certify formally at 0.07 to 0.5 for a 224×224 input (observed 3e-6), while programs
  with unknown accumulation orders (eager, ONNX Runtime) still get vacuous bounds.

## Tasks

| ID | Model | Dataset | Input domain |
|---|---|---|---|
| 0 | `squeezenet_1_1` | `imagenet_val` | pixels in [0, 1] |
| 1 | `mobilenet_v2` | `imagenet_val` | pixels in [0, 1] |
| 2 | `resnet18` | `imagenet_val` | pixels in [0, 1] |
| 3 | `all_minilm_l6_v2` | `sst2` | token ids, length 128 |
| 4 | `resnet50` | `imagenet_val` | pixels in [0, 1] |
| 5 | `whisper_tiny_encoder` | `librispeech` | log-mel hull of the samples |
| 6 | `distilbert_sst2` | `sst2` | token ids, length 128 |
| 7 | `vit_b_16` | `imagenet_val` | pixels in [0, 1] |
| 8 | `smollm2_135m` | `tinystories` | token ids, length 128 |
| 9 | `smollm2_135m` | `open_orca` | token ids, length 128 |
| 10 | `gpt2` | `tinystories` | token ids, length 128 |
| 11 | `gpt2` | `open_orca` | token ids, length 128 |
| 12 | `bert_large_squad` | `squad_1_1` | token ids, length 384 |
| 13 | `qwen3_0_6b` | `open_orca` | token ids, length 128 |
| 14 | `llama_3_2_1b` | `open_orca` | token ids, length 128 (gated) |
| 15 | `llama_3_1_8b` | `cnn_dailymail` | token ids, length 128 (gated) |
| 16 | `quantized_mobilenet_v2_int8` | `imagenet_val` | pixels in [0, 1] (CPU only) |

Image models normalize inside the model, so their domain is in pixel space. LLMs are compared on
the logits of one forward pass. Not supported yet: `vnncomp_small_nets`, `text_classification_v2`.

## Compilers

All paths run float32 with TF32 off, so outputs differ only by addition order.

| ID | Name | What optimizes | Main options |
|---|---|---|---|
| 0 | `eager` | nothing (reference) | — |
| 1 | `torch_compile` | TorchInductor: fusion, generated Triton kernels | `mode` |
| 2 | `onnxruntime_gpu` | ONNX Runtime CUDA EP graph rewrites and kernels | `graph_optimization` |
| 3 | `tensorrt` | TensorRT layer fusion and tactic selection | `optimization_level` |
| 4 | `triton` | linear, matmul, conv as a Triton GEMM | `block_*`, `split_k`, `num_stages` |
| 5 | `tilelang` | linear, matmul, conv as a TileLang GEMM | `block_*`, `split_k`, `num_stages` |
| 6 | `tvm` | TVM Relax: every operator as a scheduled TIR loop nest compiled to CUDA, no cuDNN/cuBLAS | `schedule`, `fuse` |

- TileLang's `T.gemm` uses TF32 for float32, so its kernel accumulates with explicit fp32 FMA.
- With default options `triton` and `tilelang` agree bitwise: both sum each dot product in k
  order, and tile sizes do not change that. `split_k > 1` gives a different addition tree.
- FMA contraction inside GPU kernels is not controlled.
- `tvm` uses the TVM build bundled with TileLang (or an installed `tvm`) and needs an `nvcc`
  with its headers; `compile_tvm` puts a matching `/usr/local/cuda-*` toolkit on `PATH` itself.
  `schedule="fallback"` (default) keeps every reduction a serial loop in TIR order;
  `schedule="default"` does not build for CUDA with TileLang 0.1.15's TVM.

## Tests

On server3 (RTX 6000 Ada, GPU 7) all checks pass, except where gated data or weights are missing.

```bash
CUDA_VISIBLE_DEVICES=7 python test/test_load_data_model.py --data-num 8
CUDA_VISIBLE_DEVICES=7 python test/test_load_compiler.py    # tasks x compilers, vs eager and fp64
CUDA_VISIBLE_DEVICES=7 python test/test_diff_testing.py
CUDA_VISIBLE_DEVICES=7 python test/test_certifier_models.py # certify, then check vs diff_testing
CUDA_VISIBLE_DEVICES=7 python test/test_certifier_models.py --tasks 2 --radius 0   # one sample
CUDA_VISIBLE_DEVICES=7 python test/test_layer_bounds.py     # rounding bound vs real kernels
CUDA_VISIBLE_DEVICES=7 python test/test_linear.py           # one linear layer, eager vs each path
CUDA_VISIBLE_DEVICES=7 python test/test_lenet5.py           # LeNet-5, eager vs each path
CUDA_VISIBLE_DEVICES=7 python test/test_fprev_gpu.py        # GEMM accumulation orders (FPRev)
pytest                                                      # src/dlclv: matching, bounds, FPRev,
                                                            # reconstruction (GPU kernels if CUDA)
CUDA_VISIBLE_DEVICES=7 python test/test_reconstruction_models.py --tasks 2 --device cuda \
    --compiler triton --image-size 64 --mode primitive --probe-kernels   # ResNet18, bitwise
```

## Baselines

Must beat: tree-depth bound (Higham), order-agnostic bound (Rump), random order testing, FPTaylor,
Gappa, Daisy, Satire, z3 bit-precise. To show the gap (no ε bound): Alive2, mlir-tv. None is
integrated yet; see `baselines/registry.yaml` and `baselines/INTEGRATION_PLAN.md`.

## Layout

```
utils.py        tasks, models, data, input domains, load_compiler, diff_testing
compilers.py    compilation paths
kernels.py      Triton and TileLang GEMM kernels
test/           GPU tests for the above
configs/        compilation paths, benchmarks, datasets, domains, experiment protocol
docs/           specification, evaluation plan, pilot results, references
baselines/      baseline registry and integration plan
src/dlclv/      certifier: graph extraction, matching, FP error model, FPRev, abstract interpretation;
                reconstruction: reconstruction.py, lowering.py, eager_lowering.py,
                tir_lowering.py, primitives.py
pilot/          earlier exploratory CPU measurement (docs/PILOT_RESULTS.md)
tests/          tests for src/dlclv
```

## Rules

- `status` and `integration` fields must stay truthful: `implemented`, `pilot` or `unimplemented`.
- Every reported outcome goes through `check_result`.
- Sampling never yields `verified`. A loose bound never yields `refuted`.
