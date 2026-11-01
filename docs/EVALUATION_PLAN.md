# Evaluation plan

Status: plan only. Pilot numbers are exploratory (`PILOT_RESULTS.md`). Citations are in
`REFERENCES.md`. Facts about XLA and Inductor come from `main` on 2026-11-01 and need re-pinning.

## 0. Constraints that shape everything

**Deadline.** OSDI '27 abstracts are due December 1, 2026 and papers December 8, 2026, 5:59 pm EST.
Papers are limited to 12 pages plus references. The CFP says placeholder abstracts are desk
rejected, and that a first-pages-only screening stage may be added. Those dates are 30 and 37
days after this plan was written, and the repository had no code when it was written.

**Decision.** Build for December 8, with a go/no-go gate on November 15 (section 5). The
evaluation below is deliberately smaller than the ideal one. If the gate fails, submit to a later
venue with the full plan rather than submit a weak result. A strong OSDI paper needs a working
system on real paths, and no evaluation design guarantees an award.

**Assumption.** The repository holds no verifier yet, so this plan assumes one design: a
relational analyzer over tensor graphs that emits checkable certificates. If your technique
differs, sections 1–2 still hold, but ablations A1–A7 need to change.

## 1. The core claim

Neighboring work leaves a specific gap, based on the abstracts and sections read:

| Work | Decides | Does not provide |
|---|---|---|
| Volta | Equivalence of GPU kernels modeling tensor elements as reals | Any rounding bound; its authors argue float-sound equivalence would reject many kernels |
| Tensor Core paper | Bitwise equality within reduction-order classes | A bound across classes |
| TAO | Per-operator worst-case bounds plus empirical thresholds | A sound end-to-end bound |
| TorchLean | Lean semantics and verified fragments | Correctness of `torch.compile`-style optimization |
| mlir-tv | Refinement with abstract FP as uninterpreted functions | Any `ε` |
| Satire, FPTaylor, Daisy | `|float − real|` for one program | A relational bound for compiled pairs |

**Claim to test:** sound `ε` certificates for real compiled pairs at model scale, tight enough to
be useful, with discrepancies explained when the check fails.

A search did not find a system doing exactly this, but that is not proof that none exists. Repeat
the search before submitting, because several of these preprints are weeks old.

The pilot says where the difficulty lies:

- Domain-wide worst-case bounds were about 10⁶–10⁷ above observed differences (P1, P2).
- Bit-precise SMT could not prove one FMA site in 90 s (P6).
- The largest differences came from reductions and kernels, not from graph rewrites (P1, P2).

So tightness on realistic domains is the main risk and the main contribution. Scalability alone
would be a weak result if the bound were vacuous.

## 2. Compilers and optimization systems

### Comparison

| System and path | Adoption | Optimization diversity | Hardware | Verification feasibility | Role |
|---|---|---|---|---|---|
| **XLA CPU via JAX**: StableHLO → HLO → LLVM IR | High; JAX default backend | Algebraic simplifier, fusion, inline polynomial math, FMA contraction | x86, Arm | **Medium-high**: inspectable stages, HLO spec. Library fusions (YNN/XNN) must be pinned or modeled. | **Core 1** |
| **ONNX Runtime CPU**: ONNX → optimized ONNX | High, 22k stars | Levels Basic/Extended/Layout. Conv+BN fusion observed in P3. | x86, Arm | **High at L0**, since graphs are dumpable. MLAS kernels axiomatized. | **Core 2** |
| **TorchInductor CPU**: FX → C++ | Very high; `torch.compile` default | Fusion, vectorization, tiling | x86, Arm, GPU via Triton | **Medium**: needs a generated-code frontend; effective compiler flags are order-sensitive | **Core stretch** |
| **IREE llvm-cpu** | Medium, 4k stars | MLIR pipeline | CPU, GPU, embedded | **High transparency** but pass and microkernel configuration unverified here | Extension 1 |
| **XLA GPU, Inductor/Triton** | Very high | Partitioning, split-K, fused epilogues | NVIDIA, AMD | **Low for formal claims**: cuBLAS and vendor math lack specifications | Extension |
| **TensorRT-LLM** | High for serving | FP8 and engine-level fusion | NVIDIA | **Black-box only**; `verified` is unreachable | Empirical only |
| **OpenVINO CPU** | Medium-high, 11k stars | Graph transforms, oneDNN JIT, documented hint-based precision | Intel | **Low-medium**: JIT kernels | Extension |
| **ExecuTorch or LiteRT + XNNPACK** | High on edge | Delegation, quantization | Arm, mobile | **Medium** for int8; float kernels are hand-written | Quantization extension |
| **Apache TVM** | Active repository, used by NNSmith | Relay-style passes | Many | Medium; lower priority than the above | Fuzz target only |

### Exact configurations

| Path | Baseline configuration | Variants | Why |
|---|---|---|---|
| XLA CPU | `xla_cpu_enable_fast_math=false`, `xla_cpu_enable_fast_min_max=true`, `xla_allow_excess_precision=true`, XNNPACK and YNN at defaults | `fast_math=true`; `xla_allow_excess_precision=false`; library fusions off, with flags pinned to the release | Source defaults; `contract` is set on add/sub/mul, so FMA fusion occurs without fast math |
| ORT | `ORT_DISABLE_ALL` as reference stage; `BASIC`, `EXTENDED`, `ALL` as targets; 1 thread | 2, 4, 8 threads | Level ablation separates fusion from layout and kernel effects |
| Inductor | `TORCHINDUCTOR_USE_FAST_MATH=0`, unsafe-math flag 0, `ffp-contract=off`, `TORCHINDUCTOR_DETERMINISTIC=1` | `ffp-contract=fast`; unsafe math on | Default list also carries signed-zero and trapping flags whose net effect depends on driver order, so capture it |

Target CPU: pin one x86-64 machine with AVX2+FMA and record CPUID-based dispatch. Add AVX-512
only after the first target is finished.

Trust level: L0 for ORT, L1 for XLA and Inductor. Say explicitly that L1 trusts LLVM's backend.
arm-tv covers LLVM→AArch64 for floating point and Neon, so an L2 demonstration on Arm is a
credible extension, but only after the artifact is located and tested.

## 3. Benchmarks

| Tier | Contents | Purpose | Stress | Feasibility |
|---|---|---|---|---|
| **T0 operators** | 24 operators, including dot, reductions, conv, softmax, norms, casts | Semantics, soundness anchors | Reduction order, math functions | Core, days |
| **T1 fused kernels** | 12 rewrites, including FMA, reassociation, Conv+BN, Gemm, attention, polynomial math | Pass-level claims | Constant folding, cancellation | Core, about 1–2 weeks |
| **T2 components** | ResNet block, inverted residual, MHA, encoder block | Composition | Skip connections, non-Lipschitz cancellation | Core, depends on T1 |
| **T3 models** | VNN-COMP small nets, SqueezeNet, MobileNetV2, text_classification_v2, ResNet-18, one small transformer | Real workload evidence | 10⁶–10⁷ parameters | Core; ResNet-18 and transformer are the gate |
| **T4 ambitious** | ResNet-50, ViT-B/16, BERT-large, small LLM decoder, Whisper tiny, int8 MobileNetV2 | Headline scale | 10⁷–10⁸ parameters | Extension |
| **Synthetic** | Depth and width sweeps | Separate growth rates | Controlled | Core |

Why these are representative:

- SqueezeNet, MobileNet and text_classification_v2 allow alignment with mlir-tv's evaluation.
- VNN-COMP nets connect to the neural-network verification community and to α,β-CROWN.
- MLPerf Inference models give T4 an industry-standard basis.
- Softmax, polynomial math and mixed-precision folds are where discrepancy is plausible, based on
  the known semantics. Measure them first.

Workloads that expose discrepancies:

- Large reductions with mixed signs (P1).
- Normalization folds with small `σ`, whose reciprocal scales differences.
- Softmax with large logits, near-tie max.
- f32→bf16→f32 removal under `xla_allow_excess_precision`.
- Near-zero outputs (P3), subnormal activations (P4).
- Fast-math variants, as positive controls.

Weights must be trained, not random. Random-weight pilots understate magnitude structure.

## 4. Baselines

Guarantee classes are defined in `baselines/registry.yaml` and enforced by `check_result`.

| Baseline | Reference | Availability | Guarantee | Adaptation | Limitations |
|---|---|---|---|---|---|
| FPTaylor | TOPLAS 2018 | MIT, pushed 2024-09 | **Formal bound** | FPCore-style expansion; triangle | Single expression, small scale |
| Gappa | Project site | Public, proof-producing | **Formal bound** | Generate hints | Manual hints may be needed |
| Daisy | TACAS 2018 | Apache-2.0, 2026-09 | **Formal bound** | Input translation | Small programs |
| Satire | SC 2020 | MIT, 2025-07 | **Formal, first-order** | Straight-line expansion | First-order only, straight-line |
| NumFuzz | PLDI 2024 | GPL-3.0, mirror | **Formal bound by type system** | Port to its language | No overflow or underflow, positive reals (paper) |
| Bean | PLDI 2025 | No public artifact located | **Formal, backward** | None yet | Different property |
| Alive2 | PLDI 2021 | MIT, 2026-10 | **Formal, bounded refinement** | Run on LLVM IR pairs | No `ε`; bounded unrolling; intra-procedural |
| mlir-tv | CAV 2022 | Apache-2.0, 2025-03 | **Formal under FP abstraction** | Old solver and MLIR | No `ε`; dialect subset; 30 s timeouts in the paper |
| TensorRight | POPL 2025 | Apache-2.0, 2026-10 | **Formal, rewrite rules** | Map rewrites to rules | Rule-level, no rounding bound found |
| Volta | PACMPL 2026 | MIT, 2026-10 | **Formal over reals** | PTX inputs | Reals, statically known shapes; GPU only |
| z3/cvc5 bit-precise | Solvers | Available | **Formal, exact** | Direct encoding | P6 timeouts |
| α,β-CROWN + difference graph | arXiv 2103.06624 | Public, 2026-10 | **Formal for real semantics only** | `M − C` graph, inject rounding intervals | Sound for FP only if rounding is modeled |
| ReluDiff, NeuroDiff | arXiv 2001.03662, 2009.09943 | Artifact not located | **Formal for real semantics only** | ReLU-only pairs | Same-architecture focus |
| NNSmith | ASPLOS 2023 | Apache-2.0, 2025-01 | **Empirical** | Add `ε` oracle | No guarantee |
| WhiteFox | OOPSLA 2024 | Apache-2.0, 2025-08 | **Empirical** | Same | Needs LLM access, no guarantee |
| MLIRSmith, MLIRod | ASE 2023, ISSTA 2024 | Locate artifact | **Empirical** | MLIR paths only | Crash-focused |
| FLiT, pLiner, Verificarlo | HPDC 2019, SC 2020 | Public | **Empirical** | Numeric variability on kernels | Sampling; localization rather than proof |
| TAO thresholds | EuroSys 2026 | No code located | Per-op formal; **thresholds empirical** | Reimplement | Per-operator bounds not propagated |
| Higham–Mary | SISC 2019 | Reimplement | **Probabilistic** | Formula-based | Mean-zero independence model |
| TorchLean, VCFloat, LAProof | 2026 preprint; repos | MIT, LGPL, MIT | **Machine-checked, fragments** | Cross-check small lemmas | Not compiler optimization correctness |

Must-have set for the paper: FPTaylor or Gappa, Satire, Daisy, Alive2, mlir-tv or published-data
comparison, bit-precise SMT, α,β-CROWN difference graph, NNSmith with `ε` oracle, Higham–Mary or
TAO-style thresholds. See `baselines/INTEGRATION_PLAN.md`.

## 5. Implementation order and gates

| Dates | Deliverable | Exit test |
|---|---|---|
| Nov 2–8 | Spec freeze; `R_ieee`; exact-rational replay; extract XLA and ORT pairs; T0 | Replay computes exact distance for T0 on both paths |
| Nov 9–15 | Relational analyzer for dot, reduction, conv; certificate checker; T1 | **Gate below** |
| Nov 16–22 | T2; falsifier; mutation suite; baselines 1–4 | No violations; mutants classified |
| Nov 23–29 | T3 and scale sweep; remaining baselines; abstract and first pages | ResNet-18 or transformer attempt finished, either way |
| Nov 30–Dec 3 | Abstract Dec 1; final reruns; artifact freeze Dec 3 | All tables produced by script |
| Dec 4–8 | Paper, figures, threats to validity | Submit Dec 8 |

**Go/no-go on November 15. All are required:**

1. A real pair (not hand-written) from XLA or ORT gets a checked certificate.
2. T1 median tightness is below 10³ on a local domain. The threshold is a judgment call, set
   because `ε ≈ 10⁻⁴` on binary32 needs roughly that to be useful. Revise it if your application
   `ε` differs, but do it before seeing the results.
3. No soundness violation is open.

Likely bottlenecks, largest first:

1. **Loose bounds through deep networks.** Worst-case error multiplies along layers. Mitigations:
   local domains, relational correlation, norm-based summaries.
2. **Opaque kernels.** A verified graph means little if a dot goes through a closed kernel. Pin
   library use and say which kernels are axiomatized.
3. **ReLU, min/max and branches.** If `M` and `C` take different branches, relational correlation
   breaks. Use continuity or Lipschitz arguments.
4. **Stage matching.** Op identity is lost across fusion. Use compiler metadata first, then
   real-equivalence checking.
5. **Elementary-function assumptions.** Polynomial approximations need a sound method-error bound.
6. **Memory at tensor granularity.** Certificates must not grow with scalar operation count.
7. **Baseline porting.** Cap effort per baseline. If mlir-tv takes more than 3 days, drop the
   rerun and label the comparison as using published results.

## 6. Experimental design

### Research questions

| RQ | Question | Experiment |
|---|---|---|
| RQ1 | What fraction of real pairs gets a verified, refuted or inconclusive outcome? | E1 |
| RQ2 | How tight are certified bounds? | E2 |
| RQ3 | Is the tool sound, and does it find real discrepancies? | E3, E4 |
| RQ4 | How do runtime and memory scale? | E5 |
| RQ5 | How does it compare with existing methods? | E6 |
| RQ6 | How do domain, precision and configuration change results? | E7 |
| RQ7 | Which design choices are responsible? | E8 |

### Metrics

| Property | Metric | Reporting rule |
|---|---|---|
| Coverage | Counts per outcome and cause, for each tier and path | Denominator is all attempts, including unsupported and timed-out cases |
| Tightness | `ε_cert`, best `ε_obs`, ratio `ε_cert/ε_obs` | True worst case is in `[ε_obs, ε_cert]`. Ratio is an upper bound on looseness and depends on falsifier strength. |
| Usefulness | `ε_cert` against output scale and decision margin | Report whether labels are provably preserved |
| Runtime | Median, IQR across 5 runs, per-stage split | Cactus plot; timeouts stay in the distribution as censored values, never dropped |
| Memory | Peak RSS, certificate size | Same set as runtime |
| Check cost | Certificate checker time | Reported separately from analyzer time |

### Sensitivity

| Factor | Levels |
|---|---|
| Domain | D0–D5; radius sweep; number of partitions |
| Precision | binary32, binary64; bf16 and binary16 in extension |
| Optimization | ORT levels, XLA flag variants, Inductor variants, library fusions on and off |
| Environment | FTZ/DAZ on and off, 1–8 threads, AVX2 versus AVX-512 |
| Tolerance | The `ε` grid plus the certified minimum |

Report outcome transitions, for example how many benchmarks move from verified to inconclusive.

### Ablations

| ID | Removed | Expected question |
|---|---|---|
| A1 | Relational analysis | Is correlation needed over a triangle bound? |
| A2 | Shared-subexpression correlation | Does cancellation matter? |
| A3 | Tensor-level summaries | Does scale survive scalar expansion? |
| A4 | Order-agnostic reductions | Does fixing one tree lose coverage? |
| A5 | Domain partitioning | Tightness versus time |
| A6 | Falsifier guidance | Does `ε_obs` weaken, and do cases become inconclusive? |
| A7 | SMT or exact fallback | Value of local precision |

### Scalability

- Sweep depth and width separately (`configs/benchmarks.yaml`) and fit growth on log axes.
- Sweep reduction fan-in from 2³ to 2²⁰.
- Report per-stage runtime for extraction, matching, bounding and checking.
- Report model scale three ways: graph nodes, parameters and scalar FLOPs.
- Limits: 1 hour per query, 4 hours per benchmark, 128 GB. Check these against the actual
  machine, then freeze them.

### Soundness validation

No testing proves soundness, so use independent layers:

1. **Exact falsification.** For each verified result, attack with random, corner and
   gradient-guided inputs, and compute distances in exact rational arithmetic on the real
   artifact. One violation blocks the claim until root-caused.
2. **Exhaustive ground truth.** Run on a small minifloat format and on small binary16 domains,
   where all inputs can be enumerated. Certified bounds must be at or above the true maximum.
3. **Mutation.** Seed target changes: wrong merged constant, dropped scale, swapped operands,
   FTZ on, narrowed precision, altered polynomial coefficient, plus benign reorderings. Exact
   replay classifies each mutant before the tool runs.
4. **Certificate checking.** Use a small independent rational-arithmetic checker. Aim to mechanize
   key lemmas in Lean or Coq after submission.
5. **Cross-tool agreement.** On T0, FPTaylor and Gappa bounds should not fall below `ε_obs`. On
   disagreement, investigate; do not vote.
6. **Assumption checks.** Compare library kernels and math functions against their declared error
   assumptions on dense samples. A failed assumption invalidates dependent certificates.

### Reporting outcomes

| Outcome | Requires | Must not be conflated with |
|---|---|---|
| `verified` | Sound analysis, checked certificate, assumptions digest, `ε_cert ≤ ε` | "No counterexample found" |
| `refuted` | Witness replayed on the real artifact with exact distance above `ε` | A bound above `ε` |
| `inconclusive` + cause | `bound_above_epsilon`, `timeout`, `out_of_memory`, `unsupported_operator`, `unsupported_path`, `exceptional_unresolved`, `solver_unknown` | Either of the above |
| `error` | Tool failure | Inconclusive |

Further rules:

- A refutation at `ε` is a claim about that `ε`, and may simply show that the tolerance was
  unrealistic. It is not automatically a compiler bug. Reserve "bug" for behavior that violates a
  documented contract, confirmed by maintainers where possible.
- Report separately the empirical pass rate of testing tools. It is not a verification rate.
- Keep fast-math discrepancies separate: they are allowed by the flag, so they are not bugs.
- Do not choose `ε` after seeing results. Fix the grid and any application-derived value in
  advance, and always publish the certified minimum.

## 7. What the paper needs

| Priority | Experiment | Why |
|---|---|---|
| Essential | E1 on real paths | Shows a system, not a calculus |
| Essential | E2 tightness | Separates it from vacuous bounds |
| Essential | E3 soundness | A verifier paper cannot survive doubt here |
| Essential | E5 scale | Systems contribution |
| Essential | E6 with honest failures | Establishes the need |
| Essential | E8 ablation | Attributes the result to design choices |
| High value | E4 real discrepancies | Most persuasive to systems readers |
| Important | E7 sensitivity | Answers configuration questions |
| Optional | GPU, quantization, L2 | Only after everything above |

Twelve pages hold about four evaluation figures: outcome matrix, tightness distribution, scaling,
baseline comparison. The motivating example should be a measured production-path case, and the
first pages must carry the full claim in case first-pages screening is used.

Evidence that could make this a standout paper, without any guarantee:

- Certified bounds on a production path at real model scale with tightness within about one order of
  magnitude on realistic domains. Achieving this is an open question, not an expectation.
- Maintainer-confirmed discrepancy findings that the tool's certificates explain.
- A checked certificate format that compiler developers could run in CI.

## 8. Threats to validity

- **Internal:** tool bugs (mitigated by section 6), and old pilot versions.
- **Construct:** `ε_obs` depends on the falsifier, and `ε` may not reflect application tolerance.
- **External:** few paths, one CPU family, trained-weight selection.
- **Trusted base:** LLVM backend, kernels and hardware at the reported level.
- **Conclusion:** timing noise across runs and timeouts that censor distributions.
