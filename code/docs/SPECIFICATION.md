# Verification property (draft v0)

Status: **proposal for team review**. No component enforces it yet.

## Property

Let `M` be the model program, `C` the compiled artifact, `X` the input domain and
`ε` a per-output tolerance vector. `⟦·⟧(x)` is the **set** of results allowed by the semantics.

```
∀x ∈ X, ∀m ∈ ⟦M⟧_σ(x), ∀c ∈ ⟦C⟧_τ(x):
    finite(m) ∧ finite(c) ∧ ∀ output tensor t: ‖m_t − c_t‖∞ ≤ ε_t
```

Subtraction is exact rational subtraction, never floating-point subtraction.
`σ` and `τ` are explicit environments: format, rounding mode, FTZ/DAZ, ISA features, thread
count and library versions. Every certificate records them.

## Reference semantics for `M`

"What `M` means" must be fixed before anything can be verified. Eager PyTorch is not a neutral
reference: in pilot P2, PyTorch differed from JAX and ONNX Runtime on all 20,000 inputs, while
compilation inside each system changed nothing. Comparing `C` to an arbitrary eager run would
blame the compiler for kernel-library differences.

| Reference | Definition | Use |
|---|---|---|
| `R_ieee` (primary) | Sequential interpreter in a canonical topological and reduction order, using IEEE 754 binary formats. Basic operations round to nearest ties-to-even. | Headline results |
| `R_stage` | The compiler's own previous stage, as in translation validation. | Per-pass results and localization |
| `R_exec` | Eager execution with pinned kernels, modeled as an artifact. | "Compared with what users run today" |

Elementary functions (`exp`, `tanh`, `log`) have no IEEE correctly-rounded mandate. `R_ieee`
declares each function's reference as the real function plus a stated error assumption, and
that assumption is counted in the trusted base.

## Norm

- Primary: absolute `ℓ∞` per output tensor, with one `ε` per tensor. Outputs of different scales
  are not mixed.
- Secondary: guarded relative, `|m−c| ≤ ε_r·max(|m|, τ)`, with `τ` declared in advance.
- Derived only: `ℓ2 ≤ √n·ℓ∞`.
- ULP distance: descriptive only. Pilot P3 saw a 1.2M ULP maximum on values whose absolute
  distance was 4.8e-7, because near-zero values make ULP unstable.
- Argmax or label agreement needs a margin precondition: prove `2ε` is below the top-1 margin on
  the domain. A small `ε` alone does not preserve discrete decisions.

## Input domain

`X` is a finite description from `configs/domains.yaml`: a point, `ℓ∞` ball or box, or a finite
integer set, with exact rational endpoints. All points are floating-point values of the input
format. Subnormal inputs are included by default. Sampled inputs are never called a domain.

Results for a local domain are claims about that domain only, and must not be generalized to
the data distribution.

## Numerical formats

| Tier | Formats | Rule |
|---|---|---|
| Core | binary32, binary64 | IEEE arithmetic, round-to-nearest ties-to-even |
| Extension | binary16, bfloat16, mixed precision | Conversions are explicit rounding steps. Instructions with special semantics, such as `VDPBF16PS` DAZ/FTZ and non-IEEE environment handling on AMX, are modeled per instruction. |
| Extension | int8, int16 quantized | Exact fixed-point semantics; the distance to a float model is intended error, not rounding error. |
| Excluded | x87 extended, non-default rounding modes, unspecified GPU math | Result is `unsupported_path`. |

## Exceptional values

| Case | Rule |
|---|---|
| NaN or ±Inf on one side only, reachable in `X` | `refuted` once a witness replays |
| NaN or ±Inf on both sides | Not verified under the default property. Reported as `both_exceptional`, so a broken model cannot pass trivially. An opt-in class-agreement property may be reported separately. |
| Cannot exclude exceptional values | `inconclusive/exceptional_unresolved` |
| Overflow in an intermediate value | Must be proved absent, or handled as above |
| Fast-math flags (`nnan`, `ninf`) | `C` is only defined where the assumptions hold, so absence must be a proved obligation on `X` |
| `+0` versus `−0` | Distance 0. If sign can reach a sign-sensitive operation (`atan2`, division, `copysign`), it must be tracked. Compilers may ignore signed zeros. |
| Subnormals, FTZ/DAZ | Environment parameter. Analysis covers the configured mode. If the mode is unknown, both behaviors are allowed. |
| Operation domain errors (`log(x≤0)`, division by 0) | Handled as exceptional-value generation |

## Nondeterminism

| Source | Treatment |
|---|---|
| Thread count and partitioning | Set-valued reduction semantics when any partition is allowed. Otherwise thread count is a pinned parameter. Pilot P5 showed `torch.sum` changing with thread count. |
| Reduction order | Any tree permitted by the code or flags. Use order-agnostic reasoning. |
| FMA contraction | Each contractable site may be fused or not unless the binary resolves it. |
| ISA dispatch | Pin features; a certificate is valid only for the recorded features. |
| Fast-math reassociation | Refinement over the set of allowed rewrites, or treat the emitted artifact as ground truth. |
| Randomness | Disabled in inference paths, or seeded with the state treated as an input. |
| Autotuners and heuristics | Pin configuration. Record the chosen plan. |
| Uninitialized memory, data races | Out of scope, listed in the trusted base. |

## Trusted base levels

Each result reports the level it reached, since the phrase "verified compiler output" would
otherwise hide what was trusted.

| Level | Checked pair | Trusted |
|---|---|---|
| L0 | Graph to optimized graph | All kernels, codegen, hardware |
| L1 | Graph to LLVM IR | LLVM backend, library kernels, hardware |
| L2 | Graph to machine code | Lifter or ISA model, hardware |

Also trusted at every level: the semantics, numeric assumptions, certificate checker and
extraction tooling. The evaluation should say so.
