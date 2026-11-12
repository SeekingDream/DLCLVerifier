# DLCLVerifier

Certifies limits on silent numerical drift when a tensor program is optimized by a DL compiler
or kernel DSL. The repository has the models, compilers and differential testing used for the
evaluation, and a prototype certifier (`src/dlclv`, section "Certifier"). No baseline is
integrated yet.

## Scope

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

- `{"reference": "exact"}` bounds `|prog_2 - exact arithmetic on prog_1's graph|` instead. Use it
  for "model vs compiled": the eager model's own summation order is unknown, and bounding it
  makes the two-program bound vacuous.
- Graphs come from `torch.export`, ONNX files, ONNX Runtime's optimized graph, and the `eager`,
  `triton`, `tilelang` and `onnxruntime_gpu` paths. TensorRT and `torch.compile` are not
  extractable yet.
- The two graphs are matched into units of corresponding operations, e.g. add+ReLU vs a fused
  kernel.
- The difference is bounded per element by differential back-substitution (`method="backsub"`,
  the default). Accumulations use a known order (Triton/TileLang kernels, or FPRev evidence) or
  a bound over every admissible order, with node magnitudes taken from signed partial sums.
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

- TileLang's `T.gemm` uses TF32 for float32, so its kernel accumulates with explicit fp32 FMA.
- With default options `triton` and `tilelang` agree bitwise: both sum each dot product in k
  order, and tile sizes do not change that. `split_k > 1` gives a different addition tree.
- FMA contraction inside GPU kernels is not controlled.

## Tests

On server3 (RTX 6000 Ada, GPU 7) all checks pass, except where gated data or weights are missing.

```bash
CUDA_VISIBLE_DEVICES=7 python test/test_load_data_model.py --data-num 8
CUDA_VISIBLE_DEVICES=7 python test/test_load_compiler.py    # tasks x compilers, vs eager and fp64
CUDA_VISIBLE_DEVICES=7 python test/test_diff_testing.py
CUDA_VISIBLE_DEVICES=7 python test/test_certifier_models.py # certify, then check vs diff_testing
CUDA_VISIBLE_DEVICES=7 python test/test_certifier_models.py --tasks 2 --radius 0   # one sample
CUDA_VISIBLE_DEVICES=7 python test/test_layer_bounds.py     # rounding bound vs real kernels
CUDA_VISIBLE_DEVICES=7 python test/test_fprev_gpu.py        # GEMM accumulation orders (FPRev)
pytest                                                      # src/dlclv: matching, bounds, FPRev
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
src/dlclv/      certifier: graph extraction, matching, FP error model, FPRev, abstract interpretation
pilot/          earlier exploratory CPU measurement (docs/PILOT_RESULTS.md)
tests/          tests for src/dlclv
```

## Rules

- `status` and `integration` fields must stay truthful: `implemented`, `pilot` or `unimplemented`.
- Every reported outcome goes through `check_result`.
- Sampling never yields `verified`. A loose bound never yields `refuted`.
