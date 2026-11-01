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

Report which one was used. For an in-scope pair, real equality follows from both trees adding the
same operands, which the frontend must check.

## Order of work

Scope is addition reordering. Entries with `role: out_of_scope` are not scheduled.

| Order | Baseline | Work | Expected blocker |
|---|---|---|---|
| 1 | Tree-depth and order-agnostic bounds | Extract both trees; compute `Σ[γ(d_i)+γ(d′_i)]|x_i|` in exact rational arithmetic over the domain | Trees hidden inside kernels |
| 2 | Random order testing | Sample orders and inputs; replay exactly | Lower bound only |
| 3 | FPTaylor, Gappa, Daisy | Per-order bounds, then `T₁ − T₂` as one expression | Hundreds of operators may be the limit |
| 4 | Satire | Same, straight-line expansion | First-order only |
| 5 | z3, cvc5 | Promote pilot P6 to a runner | Timeouts, per P6 |
| 6 | Alive2 | Run on LLVM `reassoc` pairs | No `ε` query; associativity not modeled in its e-graph |
| 7 | mlir-tv | Pin required MLIR and solver; run reductions | Port effort; if over 3 days, cite published results and say so |
| 8 | Context tools | Only after the above | |

Do not report numbers from any baseline's paper as if they were produced on this benchmark set.
