# DLCLVerifier evaluation scaffold

Evaluation plan and scaffold for formally verifying numerical consistency between a model and
its compiled form. **The verifier does not exist yet.** This repository has no verification
results.

## Scope

The target is the numerical difference caused by **non-associativity of floating-point addition**,
such as `(a+b)+c ≠ a+(b+c)`: the same operands added in a different order by a compiler, vectorizer
or thread partition. FMA contraction, precision changes, merged constants and elementary-function
replacement are out of scope. See `docs/SPECIFICATION.md`.

The bound to beat is the tree-depth bound for both orders, combined by the triangle inequality:

```
|ŝ_T₁ − ŝ_T₂| ≤ Σ [γ(d_i) + γ(d′_i)]·|x_i|     γ(k) = ku/(1 − ku)
```

## Status

| Component | Status |
|---|---|
| Property draft (`docs/SPECIFICATION.md`) | Proposal for review |
| Evaluation plan (`docs/EVALUATION_PLAN.md`) | Written |
| Configs (`configs/`) | Written and validated for structure and dataset references |
| Outcome rules (`src/dlclv/outcomes.py`) | **Implemented, tested** |
| Config validation and status report (`dlclv`) | **Implemented, tested** |
| Pilot (`pilot/`, `docs/PILOT_RESULTS.md`) | **Ran locally; exploratory evidence only** |
| Relational analyzer, frontends, certificate checker, falsifier | **UNIMPLEMENTED** (`stubs.py`) |
| All baseline adapters | **UNIMPLEMENTED** (`baselines/INTEGRATION_PLAN.md`) |
| Experiment runners | **UNIMPLEMENTED**; `run_experiment` raises |
| Mutation and minifloat soundness suites | **UNIMPLEMENTED** |

## Models

No model has been exported or verified yet. Parameter counts are approximate.

### Vision, text and speech

| Model | Tier | Parameters | Datasets | Source |
|---|---|---|---|---|
| `vnncomp_small_nets` | Core | — | `mnist`, `cifar10` | VNN-COMP benchmark repos, pin exact files |
| `squeezenet_1_1` | Core | 1.2M | `imagenet_val` | torchvision |
| `mobilenet_v2` | Core | 3.5M | `imagenet_val` | torchvision |
| `text_classification_v2` | Core | — | `imdb` | [TensorFlow Lite](https://storage.googleapis.com/download.tensorflow.org/models/tflite/text_classification/text_classification_v2.tflite) |
| `resnet18` | Core | 11.7M | `imagenet_val` | torchvision |
| `all_minilm_l6_v2` | Core | 22.7M | `sst2` | [link](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |
| `resnet50` | Ambitious | 25.6M | `imagenet_val` | torchvision; MLPerf Inference workload |
| `whisper_tiny_encoder` | Ambitious | 37.8M | `librispeech` | [link](https://huggingface.co/openai/whisper-tiny) |
| `distilbert_sst2` | Ambitious | 67.0M | `sst2` | [link](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english) |
| `vit_b_16` | Ambitious | 86.6M | `imagenet_val` | [link](https://huggingface.co/google/vit-base-patch16-224) |
| `bert_large_squad` | Ambitious | 335.1M | `squad_1_1` | [link](https://huggingface.co/google-bert/bert-large-uncased-whole-word-masking-finetuned-squad) |
| `quantized_mobilenet_v2_int8` | Ambitious | 3.5M | `imagenet_val` | Quantized MobileNetV2; pin at export |

### LLMs

Scope: one forward pass at a fixed sequence length, compared on logits, with no KV-cache variation or sampling. A certificate would not cover multi-step generation. The input domain is `D6_embedding_ball`, so it covers perturbed embeddings, not all token sequences.

The core suite includes a SmolLM2 decoder block (T2). All full LLMs are ambitious.

| Model | Tier | Parameters | Datasets | Source |
|---|---|---|---|---|
| `smollm2_135m` | Ambitious | 134.5M | `tinystories`, `open_orca` | [link](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) |
| `gpt2` | Ambitious | 137.0M | `tinystories`, `open_orca` | [link](https://huggingface.co/openai-community/gpt2) |
| `qwen3_0_6b` | Ambitious | 751.6M | `open_orca` | [link](https://huggingface.co/Qwen/Qwen3-0.6B) |
| `llama_3_2_1b` | Ambitious | 1.24B | `open_orca` | [link](https://huggingface.co/meta-llama/Llama-3.2-1B) |
| `llama_3_1_8b` | Ambitious | 8B | `cnn_dailymail` | MLPerf llama3.1-8b |

Gated models need license acceptance. Operators (T0), reordering kernels (T1), components (T2) and synthetic sweeps are in `configs/benchmarks.yaml`.

## Compilers

`pilot` means the path was only measured by the exploratory script, with no verifier support.

| Path | System | Target | Tier | Status | Stages |
|---|---|---|---|---|---|
| `xla_cpu_default` | XLA via JAX | x86-64 CPU, AVX2+FMA pinned, then AVX-512 | core | pilot | stablehlo → hlo_unoptimized → hlo_optimized → llvm_ir → native |
| `ort_cpu_graph` | ONNX Runtime CPU execution provider | x86-64 CPU | core | pilot | onnx → onnx_optimized → mlas_kernels |
| `torch_inductor_cpu` | TorchInductor (torch.compile) CPU | x86-64 CPU | core_stretch | unimplemented | fx_graph → inductor_lir → generated_cpp → native |
| `iree_llvm_cpu` | IREE | llvm-cpu | extension | unimplemented | stablehlo_or_torch_or_tosa → flow_stream_hal_mlir → llvm_ir → native |
| `tensorrt_llm_gpu` | TensorRT | NVIDIA GPU | extension | unimplemented | torchscript_or_onnx → tensorrt_engine |
| `xla_gpu` | XLA GPU / Inductor Triton | NVIDIA GPU | extension | unimplemented | hlo_optimized → llvm_nvptx_or_ptx |
| `openvino_cpu_f32` | OpenVINO CPU plugin | x86-64 CPU | extension | unimplemented | ir → compiled_model |
| `executorch_xnnpack` | ExecuTorch or LiteRT with XNNPACK | Arm NEON | extension | unimplemented | edge_dialect → delegate_graph → xnnpack_kernels |

Flags and variants are in `configs/compilers.yaml`.

## Datasets

No dataset has been downloaded. Datasets give realistic centers and ranges for input domains. A result covers only its declared domain, not the data distribution.

| Dataset | Kind | Source | Notes |
|---|---|---|---|
| `mnist` | image | VNN-COMP benchmark instances where available | — |
| `cifar10` | image | VNN-COMP benchmark instances where available | — |
| `imagenet_val` | image | ILSVRC 2012 validation; MLPerf ResNet-50 dataset | registration and ImageNet terms required |
| `imdb` | text | [link](https://huggingface.co/datasets/stanfordnlp/imdb) | IMDB attribution for text_classification_v2 rests on an IREE sample notebook |
| `sst2` | text | [link](https://huggingface.co/datasets/stanfordnlp/sst2) | — |
| `squad_1_1` | text | [link](https://huggingface.co/datasets/squad) | MLPerf BERT dataset |
| `tinystories` | llm_prompts | [link](https://huggingface.co/datasets/roneneldan/TinyStories) | — |
| `open_orca` | llm_prompts | [link](https://huggingface.co/datasets/Open-Orca/OpenOrca) | MLPerf llama2-70b dataset |
| `cnn_dailymail` | llm_prompts | MLPerf llama3.1-8b instructions | — |
| `librispeech` | audio | MLPerf whisper instructions | — |

## Baselines

The only baseline run so far is the z3 trial in the pilot. All adapters are unimplemented; see `baselines/INTEGRATION_PLAN.md`.

### Main: must beat

| Baseline | Evidence | Repository |
|---|---|---|
| Tree-depth bound (Higham) | Formal bound | Reimplement |
| Order-agnostic bound (Rump) | Formal bound | Reimplement |
| Random order testing | Empirical | Reimplement |
| FPTaylor | Formal bound | [link](https://github.com/soarlab/FPTaylor) |
| Gappa | Formal bound | [link](https://gappa.gitlabpages.inria.fr/) |
| Daisy | Formal bound | [link](https://github.com/malyzajko/daisy) |
| Satire | Formal, first-order | [link](https://github.com/arnabd88/Satire) |
| z3 bit-precise | Formal bound | [link](https://github.com/Z3Prover/z3) |

### Shows the gap: no ε bound

| Baseline | Evidence | Repository |
|---|---|---|
| Alive2 | Formal, bounded | [link](https://github.com/AliveToolkit/alive2) |
| mlir-tv | Formal equivalence, no ε | [link](https://github.com/aqjune/mlir-tv) |

### Context

| Baseline | Evidence | Repository |
|---|---|---|
| APEG / Sardana | Abstract interpretation; no bound between two orders | Not located |
| NumFuzz | Formal bound | [link](https://github.com/Athena-Types/NumFuzz) |
| PRECiSA | Formal bound | [link](https://github.com/nasa/PRECiSA) |
| FLiT | Empirical | [link](https://github.com/PRUNERS/FLiT) |
| Verificarlo | Empirical | [link](https://github.com/verificarlo/verificarlo) |
| TAO thresholds | Empirical | Not located |
| Higham–Mary | Probabilistic | Reimplement |
| VCFloat / LAProof | Formal bound | [link](https://github.com/VeriNum/vcfloat) |

Out of scope, kept in `baselines/registry.yaml` in case the scope widens: Bean, TensorRight, Volta, arm-tv, α,β-CROWN, ReluDiff / NeuroDiff, Marabou, NNSmith, WhiteFox, MLIRSmith, TorchLean.

## Use

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pytest
PYTHONPATH=src python -m dlclv validate
PYTHONPATH=src python -m dlclv status   # item-by-item implementation labels
```

To rerun the pilot, install `pilot/requirements.lock.txt`. Those pins need a platform that has
those wheels; the recorded run used Python 3.11 on x86-64 macOS:

```bash
python pilot/run_pilot.py --out /path/to/output.json --smt-timeout 90
```

## Models and data (`utils.py`)

`load_data_model(task_id, data_num)` returns `(model, dataloader, input_domain)` for one
(model, dataset) task; `utils.TASKS` lists the 17 tasks. Install `requirements-eval.txt`, then:

```bash
CUDA_VISIBLE_DEVICES=7 python test/test_load_data_model.py --data-num 8
```

ImageNet (`ILSVRC/imagenet-1k`) and the Llama models are gated: run `huggingface-cli login`, or
set `DLCLV_IMAGENET_DIR` to a local ImageNet validation folder in ImageFolder layout.
`vnncomp_small_nets` and `text_classification_v2` are not supported yet.

## Layout

```
docs/       specification, evaluation plan, pilot results, references
configs/    compilation paths, benchmarks, datasets, domains, experiment protocol
baselines/  baseline registry and integration plan
src/dlclv/  outcome rules, config checks, unimplemented interfaces
pilot/      exploratory measurement script
results/    recorded pilot output
tests/      tests for the implemented parts
```

## Rules

- `status` and `integration` fields must stay truthful: `implemented`, `pilot` or `unimplemented`.
- Every reported outcome goes through `check_result`.
- Sampling never yields `verified`. A loose bound never yields `refuted`.
