# Pilot results (exploratory, not paper data)

Script: `pilot/run_pilot.py`. Raw output: `results/pilot/pilot_intel_macos.json`. Pinned
packages: `pilot/requirements.lock.txt`. Seed 20261031.

Machine: Intel Core i9-9880H, macOS, CPU only. JAX 0.4.38, PyTorch 2.2.2, ONNX Runtime 1.23.2,
z3 5.3.3. **These are old builds of JAX and PyTorch**, chosen because they still ship for this
platform, so they predate the XLA and Inductor sources described in the plan.

All "observed" values are lower bounds on the true worst case. No verifier was involved. The
naive bounds use float64 without outward rounding, so they are estimates, not certificates.

| ID | Question | Measured | What it implies |
|---|---|---|---|
| P1 | Does summation order matter? | `n=10⁶` binary32, 20 trials: the 4 systems never all agreed. Largest difference 0.051 (sequential vs JAX); the non-sequential systems differed by at most 0.0011. First-order order-agnostic bound: 4.76e4. | Worst-case bound exceeds observation by about 10⁶. Reductions decide tightness. |
| P2 | Does compilation change a 4-hidden-layer MLP? | 20,000 inputs in `[0,1]^64`: JAX eager = JAX jit = ORT no-opt = ORT all-opt bit for bit, and PyTorch eager = traced. PyTorch vs the others differed on all inputs, max 2.53e-7. Naive domain bound: 3.07. | The compiler changed nothing here, the kernel library did, so the reference must be fixed. The naive bound is off by about 1.2×10⁷ and is useless domain-wide. |
| P3 | Does a real graph rewrite change values? | ORT Conv+BN: optimized graph is `Conv`, `ReorderOutput`. 75.3% of elements differ, max 4.8e-7, median 2 ULP among differing elements, max 1.2M ULP. | Real change from a production path. Enable-all also changes layout and kernel, so the cause needs per-level ablation. |
| P4 | What does FTZ do to metrics? | PyTorch FTZ on subnormals: absolute 2e-40, relative 1.0, 100% flushed. ORT `session.set_denormal_as_zero` did not change the output. | Relative metrics need a floor. The ORT result is **unresolved**: the option may not reach this code path, so it shows nothing about safety. |
| P5 | Do threads matter? | ORT MLP: no change across 1, 2, 4, 8 threads or 5 repeats. `torch.sum` on 5M floats: a different value at each of 1, 2, 4, 8 threads, spread 9.8e-4, stable across repeats. | Thread count is part of the semantics for reductions. Here it behaved as a deterministic function of the partition. |
| P6 | Is bit-precise SMT viable? | z3, 90 s limit, `[-1,1]` inputs. `fma(a,b,c)` vs `(a*b)+c`: refuted at 2⁻²⁵ in 8 s, **unknown** at 2⁻²². 4-term reassociation: refuted at 2⁻²⁶ in 14 s, **unknown** at 2⁻²⁰. | Tight refutation works. Proving even one 3-input site timed out. SMT is a local tool at best. |

## Relevance to the reassociation scope

P1 and P5 measure addition order directly. P2 may include accumulation-order effects, but kernel
differences were not isolated. P3 is out of scope, since it changes constants. P6's FMA query is
out of scope; its 4-term query is in scope.

## Limits of the pilot

- One machine, one model per experiment, random weights.
- P6 compares in float64 with a rounded difference, and used one solver with default tactics.
  A better encoding may do better. It suggests the bottleneck but does not prove it.
- The `formal`-looking P6 outcomes are pilot labels and produce no certificate.
- None of these numbers should appear in the paper as evaluation results.
