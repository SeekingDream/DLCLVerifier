# Baseline integration plan

Status: **every adapter is unimplemented**. The only baseline run so far is a z3 trial in pilot
P6. The machine-readable list is `registry.yaml`. Clone third-party tools into `third_party/`,
which git ignores, and pin commits before the evaluation freeze.

## Common contract

Each adapter implements `dlclv.stubs.BaselineAdapter` and returns a `dlclv.outcomes.Result`.
`check_result` rejects records that overclaim:

- `empirical`, `probabilistic`: never `verified`.
- `formal_real_semantics`: never `verified` for floating-point execution.
- `formal_equality`: can be `verified` only at `ε = 0`. A pass at `ε > 0` is reported as the
  separate statement "proved equal under its abstraction".
- `formal_first_order`: never `verified`, unless the adapter adds a sound remainder bound.

## How single-program tools answer a relational question

FPTaylor, Gappa, Daisy, Satire and NumFuzz bound `|P_float(x) − P_real(x)|`. Two adaptations exist:

1. **Triangle**: `ε = e(M) + e(C)`, plus a real-equivalence obligation discharged separately. This
   is the fair default.
2. **Difference expression**: analyze `M − C` as one expression where the tool's input language
   allows it. This helps only if the tool correlates shared variables.

Report which one was used. The real-equivalence step can use Volta-style or TensorRight-style
reasoning. Without it, the triangle bound is unsupported.

## Order of work

| Order | Baseline | Work | Expected blocker |
|---|---|---|---|
| 1 | FPTaylor, Gappa, Daisy | Emit FPCore or tool input from T0/T1 scalar expansion | Hundreds of operators may be the practical limit |
| 2 | Satire | Same expansion, straight-line code | First-order only, reduction fan-in |
| 3 | z3 bit-precise | Promote pilot P6 to a runner, add cvc5 | Timeouts, per P6 |
| 4 | Alive2 | Lower pairs to LLVM IR, compare exact refinement | No `ε` query; expected failure on any rounding change is the point of the comparison |
| 5 | mlir-tv | Pin its required MLIR and solver; convert benchmarks to supported ops | Port effort; if over 3 days, cite published results instead and say so |
| 6 | α,β-CROWN | Build `M − C` difference graph, add rounding as an interval perturbation | Models real semantics unless rounding is injected |
| 7 | NNSmith | Generate graphs for core paths, compare outputs | Add an `ε` oracle |
| 8 | TAO-style thresholds, Higham–Mary | Reimplement per-operator bounds and profiles | TAO has no located code; state "our reimplementation" |
| 9 | Optional tools | Only after the must tier is complete | |

Do not report numbers from any baseline's paper as if they were produced on this benchmark set.
