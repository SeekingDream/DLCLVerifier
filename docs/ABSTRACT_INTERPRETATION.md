# Abstract interpretation stage

Status: **implemented** (`src/dlclv/fpmodel.py`, `fprev.py`, `absint.py`, `backsub.py`; section 8). It fills
`interpretation.UnitInterpreter.transfer`, which `TensorProgramCertifier.certify()` calls once per
matched unit in dependency order (`interpretation.propagate`).

## 1. Contract

Each boundary tensor `t` of a unit carries a triple `[lo_t, hi_t, e_t]`:

- every element of `t` in program 1 **and** in program 2 lies in `[lo_t, hi_t]`, for every input
  in the domain;
- for the same input, the two programs' values of each element differ by at most `e_t`
  (exact real subtraction).

`transfer(unit, input triples) -> output triples` must keep both properties. Graph inputs start
at `[domain lo, domain hi, 0]`. Constants start at `[min, max, 0]`: matching pairs constants
only when their bytes are identical.

Triples are per tensor, so one number bounds every element. A transfer function can still use
finer information: per-column weight magnitudes from the constants (`UnitContext.match`) and
per-element accumulation orders. It then reduces the result to a single triple.

## 2. Floating-point model

Used by both strategies. For one operation that rounds an exact result `z` to `fl(z)`:

| Setting | Local error `rho(Z)`, where `Z >= abs(z)` |
|---|---|
| Round to nearest (RN, ties to even), result normal | `|fl(z) - z| <= ½ ulp(Z) <= u·Z`, `u = 2^-p` (fp32: `2^-24`) |
| Directed rounding, or rounding mode unknown | `ulp(Z)` |
| Addition and subtraction, gradual underflow | the RN bound holds exactly: subnormal addition is exact |
| Multiplication or FMA that may underflow | add `eta = ½·2^(emin-p+1)` (fp32: `2^-150`) |
| FTZ/DAZ (flush to zero) | add `2^emin` (fp32: `2^-126`) per op whose result or input may be subnormal |
| FMA `fl(a·b + c)` | one rounding of the exact `a·b + c`; the product itself is exact |
| `fl(a·b)` then `fl(p + c)` | two roundings: product leaf and addition node |
| Accumulator precision `p_acc` differs from operand precision | use `u_acc` at accumulation nodes and `u_op` at product leaves |
| Multi-term fused node (Tensor Core: aligned and truncated, per FPRev §5.2) | not IEEE. Needs a declared per-architecture model; otherwise unsupported (TF32 and fp16 are out of scope anyway) |

`rho` is nondecreasing in `Z`. Both strategies rely on that.

**Summation identity.** In a reduction tree, every internal node `k` computes
`ŝ_k = ŝ_l + ŝ_r + δ_k` with `|δ_k| <= rho(M_k)`, where `M_k` bounds `|ŝ_l + ŝ_r|`. Summing over
the tree gives exactly

    ŝ - s = Σ_leaves δ_leaf + Σ_nodes δ_k        (δ_leaf = product rounding; 0 with FMA or plain sums)

All rounding errors add up exactly; nothing compounds multiplicatively. `M_k` must also bound
the computed operands. Use `M_k = (1 + γ_{m_k-1}) · W_k`, where `W_k` is the sum of the leaf
magnitude bounds under `k` and `γ_m = m·u/(1-m·u)` (Higham). This holds for any tree, so it does
not create a dependence on the order.

## 3. Shared transfer for an accumulation unit

Each output element is a reduction `s = Σ_i x_i` over leaves `x_i = a_i·b_i` (dot, matmul,
convolution after im2col), or over plain summands (sum, mean, softmax denominator, layer-norm
statistics). Program `j` evaluates the leaves at its own inputs `x^j`, with its own tree `T_j`
and its own FP model `F_j`.

    |ŝ1(x¹) - ŝ2(x²)|  <=  |s(x¹) - s(x²)|  +  |ŝ1(x¹) - s(x¹)|  +  |ŝ2(x²) - s(x²)|
                           \_ propagated P _/   \___ rounding R1 __/   \___ rounding R2 __/

- **`P` (propagated error).** Exact real arithmetic on the incoming triples.
  - Weight `w_i` is a matched constant, so it is identical in both programs: `P <= Σ_i |w_i|·e_a`.
  - Both factors are variables (attention `q·k`):
    `P <= Σ_i (A_i·e_b + B_i·e_a + e_a·e_b)`, where `A = max(|lo_a|, |hi_a|)` and likewise `B`.
- **`R_j` (rounding error of program `j`).** Depends only on `T_j`, `F_j` and the leaf magnitude
  bounds `X_i >= |x_i^j|`. Section 4 bounds it when `T_j` is known (Strategy 1), section 5 when
  it is not (Strategy 2).
- **Relational case.** When both trees are known **and** the incoming errors are 0, compute
  `D` (section 4.3) in place of `R1 + R2`.
- **Output range.** `[lo, hi] = [real lo - max(R1, R2), real hi + max(R1, R2)]`, where
  `[real lo, real hi]` encloses `s` over the input box. Interval arithmetic gives it, per column
  with constant weights.

Other unit kinds use the same split `e_out <= P + R1 + R2`:

- elementwise ops: `P <= sup |f'| · e_in` over `[lo, hi]`;
- `relu`, `max` and layout ops: exact, so `R = 0`;
- ops whose two implementations are the same IEEE function (same op and FMA decisions, which
  matching plus kernel evidence can show) and whose `e_in = 0`: `e_out = 0`.

Inside softmax and layer norm, the reductions use the strategies below; the nonlinear parts use
the derivative bound. Elementary functions (`exp`, `tanh`, `erf`) need a declared ULP assumption
for each implementation (specification, "Reference semantics"); otherwise see section 7.

## 4. Strategy 1: known or inferred accumulation order

### 4.1 Where the order comes from

| Kernel | Source of the order | Evidence level |
|---|---|---|
| `triton` / `tilelang` paths in `compilers.py` | Our own kernel source. The TileLang kernel is an explicit `T.serial(k)` FMA loop per output element: a sequential chain over k. Triton `tl.dot(..., input_precision="ieee")` is lowered to an FMA loop; confirm in the PTX/SASS. `split_k > 1` adds a `torch.sum` over S partial sums, whose order comes from FPRev. | static (source plus PTX check) |
| Inductor-generated Triton reductions | generated source, plus `tl.sum`'s lowering (warp-shuffle tree) | static, after reading the lowering |
| cuBLAS, cuDNN, ONNX Runtime, TensorRT kernels | opaque | FPRev, then validation (4.2) |

Observed in this repository: the default `triton` and `tilelang` paths give bitwise-identical
outputs on every model. That is consistent with both being the same sequential FMA chain. It
corroborates the static order; it does not prove it.

### 4.2 When an inferred order may be treated as fixed

FPRev (Xie et al., arXiv:2411.00442) builds the summation tree from `l(i,j) = n - Sum(A^{i,j})`
on masked inputs `A^{i,j}` (`+M` at `i`, `-M` at `j`, ones elsewhere; fp32 uses `M = 2^127`). It
needs between `n-1` and `n(n-1)/2` kernel calls. It supports w-ary fused nodes. Its correctness
proof assumes an order that is "uniquely determined by the given implementation on specific
hardware", and it excludes randomized or value-dependent orders. So FPRev **assumes** data
independence rather than proving it. The paper also reports orders that change with `n` (NumPy)
and with hardware (BLAS on different CPUs and GPUs).

The certifier therefore records an inferred order as **fixed** only when all of the following
hold. Each item goes into the certificate's assumptions digest.

1. **Pinned configuration.** The order is revealed at exactly the certified shapes and the
   certified launch: GPU model and SM count, driver, library versions, kernel binary hash, grid,
   split-K factor, workspace. It also records the selection decision: cuBLAS algorithm,
   TensorRT tactic, ORT cuDNN algorithm. ORT's `EXHAUSTIVE` search must be pinned or recorded.
   At run time a different kernel invalidates the certificate.
2. **Determinism.** The kernel gives bitwise-identical results across repeated runs. The SASS has
   no float `ATOM`/`RED` in the reduction. Deterministic-algorithm settings are on.
3. **Data independence.** *Static:* the reduction's control flow and addressing do not depend on
   values (true by construction for our kernels; read from SASS for opaque ones). *Dynamic
   corroboration:* a bit-exact emulator of the inferred tree, using the FP model in item 4,
   reproduces the kernel's output bit for bit on random and adversarial domain samples (wide
   exponent spread, cancellation, subnormals, near-overflow).
4. **FP model.** FMA versus separate multiply and add, accumulator precision and rounding mode
   come from SASS (`FFMA` vs `FMUL`+`FADD`; any `HMMA` means TF32/fp16, which is out of scope) and
   from probe inputs that separate the alternatives. FPRev reveals the tree only.
5. **Position classes.** The order is the same for every output element, or FPRev is run once per
   class of element positions (edge tiles, split-K remainders), with item 3 checked on all of
   them.
6. **FPRev preconditions.** `M` and `n` are representable (`n <= 2^24 + 1` for fp32
   accumulation; otherwise use the paper's ModifiedFPRev).

With items 1, 2, 4, 5, 6 and the static part of 3, the order is `known`. With the dynamic part
of 3 only, it is `validated`. A strict policy treats `validated` as unknown and uses Strategy 2;
a permissive policy accepts it as a recorded assumption in the trusted base. If any other item
fails, use Strategy 2 for that kernel.

### 4.3 Algorithms

    KnownTreeBound(T, X, F):              # R_j for a known tree; X[i] = leaf magnitude bound
        for node k of T in post-order:
            if k is a leaf i:
                W[k] = X[i]
                E[k] = 0 if F.fma or F.plain_sum else rho_op(X[i])   # product rounding
            else:                                                     # children c1..cr (r = 2,
                W[k] = Σ W[c]                                         #  or w if fused)
                M = (1 + γ(leaves(k) - 1)) · W[k]
                E[k] = Σ E[c] + rho_acc(M)
        return E[root]

    RelationalDiff(T1, T2, X, F1, F2):    # D, used only when every incoming e_i = 0
        S = maximal subtrees present in both T1 and T2 with the same leaf set, the same shape,
            the same FMA decisions and the same precision
        T1', T2' = T1, T2 with each subtree in S replaced by one shared leaf
        # Same inputs, same IEEE operations => bitwise-equal value in both programs, so the
        # shared leaf is a common real number. By the summation identity, ŝ_j = s' + Σ δ over
        # the nodes of T_j', with s' the exact sum of the (shared) leaves.
        return KnownTreeBound(T1', X', F1) + KnownTreeBound(T2', X', F2)
               # X' of a shared leaf = W of that subtree, times (1 + γ)

    TransferKnown(unit, triples):
        P = Propagated(unit, triples)
        if every incoming e = 0 and both trees are known:
            err = RelationalDiff(T1, T2, X, F1, F2)
        else:
            err = P + KnownTreeBound(T1, X, F1) + KnownTreeBound(T2, X, F2)
        return [range(unit, triples, err), err]

**Scaling.** Within a kernel, every output element in one position class shares the same tree.
Leaf bounds differ only through constant weights. So `KnownTreeBound` runs once per class, as a
vectorized pass over all columns: `X` has shape `[K, N]`. That costs `O(K·N)` per matmul, the
same as one matmul. Sequential chains reduce to prefix sums of `W`.

**Soundness.**
- Each `δ_k` is bounded by `rho(M_k)`, and `M_k` bounds the operand magnitude by induction on
  the tree.
- The summation identity makes `ŝ_j - s` the exact sum of the local errors.
- The triangle inequality combines `P`, `R1` and `R2`.
- `RelationalDiff` adds one fact: IEEE operations are deterministic functions. Equal operands,
  operation, rounding mode and FMA decision give equal results. So shared subtrees cancel
  exactly. This needs `e = 0` on their leaves: with different inputs they are not bitwise equal.
- What is assumed is exactly the order evidence of section 4.2 and the FP model, both recorded.

## 5. Strategy 2: unknown accumulation order

### 5.1 Admissible orders

The admissible set `𝒯` is described by a small grammar, from whatever is known:

    Order := Known(T)                  # Strategy 1 result
           | AnyTree(children)         # any full binary tree over the children, any leaf order
           | Depth(children, D)        # any tree of depth <= D (bounded parallel reduction)
           | Chunks([Order...], comb)  # disjoint known chunks, combined by `comb`
    FP     := fma | separate | either  # `either` when FMA cannot be established

The default when nothing is known is `AnyTree(all leaves)` with `FP = either`. Typical
refinements:
- `Chunks(sequential FMA chains over k-ranges, AnyTree)` for split-K with an unknown final
  combine;
- `Depth(..., ceil(n/T) + ceil(log2 T))` for a reduction over at most `T` threads.

Each refinement needs evidence of the kind in section 4.2, but weaker: it constrains the order
without fixing it.

### 5.2 Bound over all admissible orders

**Lemma.** Take any full binary tree on leaves with magnitudes `x_(1) <= ... <= x_(n)` and total
`W`. List the weights of its `n-1` internal nodes in decreasing order: `s_1 >= s_2 >= ...`. Then

    s_j <= W - (x_(1) + ... + x_(j-1))      for j = 1..n-1,

with equality for every `j` at once in the sequential tree that adds leaves in decreasing
magnitude.

*Proof.*
1. Let `U` be the internal nodes of weight `>= s_j`. `U` contains the root and is closed under
   taking ancestors, so it is a connected subtree with `q >= j` nodes. It therefore has
   `2q - (q-1) = q+1` outgoing children.
2. Pick `v` in `U` with no child in `U`. Both of `v`'s children are outgoing.
3. The other `q-1` outgoing subtrees contain distinct leaves, so
   `W >= weight(v) + (x_(1)+...+x_(q-1))`.
4. `s_j <= weight(v)` because `v` is in `U`, and `q >= j`. So
   `s_j <= W - (x_(1)+...+x_(j-1))`.
5. The decreasing-order sequential tree removes the smallest leaf at each level, so it attains
   every bound. ∎

**Corollary.** `rho` is nondecreasing and every node bound is `rho((1+γ_{n-1}) W_k)`. Then for
every `T` in `AnyTree`:

    R(T) <= Σ_{j=1}^{n-1} rho((1+γ_{n-1}) · (W - x_(1) - ... - x_(j-1)))  +  [FP=either: Σ_i rho_op(x_i)]

So the candidate worst-case tree (sequential, largest first) is a proved maximum **of the bound
function** over the whole admissible set. That is the proof the specification asks for before a
single path may stand for all of them. It does not claim that this tree produces the largest
actual error; it does not need to.

For equal magnitudes `X` this is about `u·X·n²/2`. That is half of the classical `γ_{n-1}·Σ|x_i|`
for an unknown order.

    AdmissibleBound(order, X, F):
        match order:
          Known(T):        return KnownTreeBound(T, X, F)
          AnyTree(cs):     W_c, E_c = [ (subtree weight, AdmissibleBound(c)) for c in cs ]
                           sort W_c ascending;  W = Σ W_c;  acc = 0;  R = Σ E_c
                           for j in 1..len(cs)-1:
                               R += rho_acc((1+γ) · (W - acc));  acc += W_c[j]
                           return R + (F == either ? Σ_leaves rho_op : 0)
          Depth(cs, D):    return min(AdmissibleBound(AnyTree(cs)),       # superset of Depth
                                      γ_D · Σ X_i + E_children,           # Higham, depth <= D
                                      DPDepth(cs, D) if leaves are uniform and n·D is small)
          Chunks(os, comb):return Σ AdmissibleBound(o) + AdmissibleBound(comb over chunk sums)

`DPDepth` maximizes `Σ rho(W_k)` over trees of depth `<= D` with uniform leaves:
`F(m,d) = rho(m·X) + max_{a+b=m, a,b <= 2^(d-1)} F(a,d-1) + F(b,d-1)`. Every bound in the `min`
is valid for the whole set, so the minimum is too.

**Cost.** `O(n log n)` per output class, from sorting the leaf bounds. It is vectorized over
columns, like Strategy 1: `O(K log K · N)` for a matmul.

### 5.3 One order known

Keep Strategy 1 on the known side and use Strategy 2 only on the unknown side:

    err = P + KnownTreeBound(T_known, X, F_known) + AdmissibleBound(𝒯_unknown, X, F_unknown)

`RelationalDiff` needs both trees, so it does not apply. **Soundness:** the decomposition in
section 3 bounds each program's rounding separately; the known side's bound is valid for its
actual tree, and the unknown side's bound is valid for every tree in `𝒯`, which contains the
actual one.

### 5.4 Cancellation-aware node magnitudes

Sections 4 and 5.2 bound a node's magnitude by `(1+γ)·W_k`, the sum of the leaf *magnitudes*
below it. Leaves of opposite sign cancel, so the value a node actually rounds is much smaller.
With `s_k` the exact sum of the leaves under node `k`:

    |z_k| <= |s_k| + γ·W_k            (z_k: the exact sum of the two computed operands)
    |fl(z_k) - z_k| <= u · pow2floor(|z_k|)     (half an ulp of z_k; pow2floor(z) <= z)

`|s_k|` is bounded from the leaf **intervals** `[lo_i, hi_i]` (for a product, the interval
product of weight and input range):

| Order | Bound on `|s_k|` |
|---|---|
| Known sequential chain | `s_k` is a prefix sum: `max(|Σ_{i<=k} lo_i|, |Σ_{i<=k} hi_i|)` |
| Chunks | the same inside each chunk; the combine uses the chunk sums' enclosures |
| AnyTree | the leaves under `k` are an unknown subset: `min(W_k, G)` with `G = max(Σ_i max(hi_i, 0), Σ_i max(-lo_i, 0))` |

For AnyTree, `w -> u·pow2floor(min(w, G) + γ·w)` is nondecreasing, so by the lemma the sum over
the nodes is largest for the decreasing sequential tree, as before. Each bound is never larger
than the one it replaces. The gain is large for known chains (prefix sums of mixed-sign products
grow like a random walk, not like `k`), and at most 2× for AnyTree from `G` (an adversarial tree
can add all positive leaves first), plus about 1.4× from `pow2floor`.

The probabilistic model uses the same node magnitudes in its variance proxy
`Σ_k (u·pow2floor(M_k))²`. `signed_accumulation=False` restores the magnitude-only bounds.

### 5.5 Identical kernels on identical inputs

A unit whose inputs are bitwise equal in both programs has bitwise-equal outputs when every
operation in it is either exact, a correctly rounded operation run on its own, a pair of
identical known trees (section 4.3), or **the same PyTorch eager kernel on both sides** (the
model itself, or an operator that a Triton/TileLang path leaves in place). The last case is an
assumption (run-to-run determinism of the kernel for equal inputs, attributes, shapes and
device); it is listed in `result.assumptions` and can be switched off with
`assume_deterministic_kernels=False`. It makes "model vs eager" and "Triton vs TileLang" certify
at exactly 0, which is what is observed. It does not apply to ONNX Runtime graphs.

### 5.6 Exact arithmetic as the reference (`reference="exact"`)

`|program 1 - program 2|` needs a rounding bound for both programs. When program 1 is the eager
model, its accumulation orders are unknown (and its cuDNN 3×3 convolutions are not summation
trees, section 8), so its any-order bound dominates and cascades. With `reference="exact"`
the certified quantity is

    |program 2(x) - F(x)|,     F = program 1's graph evaluated in exact real arithmetic,

for every `x` in the domain. Program 1 then only defines `F`: its kernels contribute no
rounding, nothing is cancelled against it (section 5.5 and the relational case are off), and
ranges use program 2's error only. The assumptions about program 1's kernels
(`assume_sum_of_products_conv`, determinism) are no longer needed for it; operators that
program 2 itself runs on eager kernels still use the any-order bounds.

To compare two compiled programs under this reference, certify each against `F` and add the two
bounds.

### 5.7 Unstable ReLUs close to one side

For `a, a'` in `[lo, hi]` with `lo < 0 < hi`, the midpoint slope ½ leaves a remainder of
`|a - a'|/2`. When the range lies mostly on one side, the slope of that side is better:

    slope 1:  |relu(a) - relu(a') - (a - a')| = |min(a,0) - min(a',0)| <= min(|a - a'|, -lo)
    slope 0:  |relu(a) - relu(a')|                                     <= min(|a - a'|, hi)

It is used when the short side is under a quarter of the range (`relu_point_slope`).

## 6. Example: a 4-term reduction

Here `x_1..x_4` lie in `[0, 1]` (so each `X_i = 1`), fp32 RN, plain sums, `u = 2^-24`. Node
bounds use `rho(M) = u·M` (second-order terms dropped for readability).

- `T1 = ((x1 + x2) + x3) + x4`: sequential, e.g. a single FMA chain (default Triton/TileLang).
- `T2 = (x1 + x2) + (x3 + x4)`: two chunks, e.g. `split_k = 2`, then one combine.

| What is known | Computation | Bound |
|---|---|---|
| Both orders, incoming e = 0 | Shared subtree `y = x1+x2` (bitwise equal in both). `T1' = (y+x3)+x4`: `3u + 4u`. `T2' = y+(x3+x4)`: `2u + 4u`. | **13u ≈ 7.7e-7** |
| Both orders, incoming e > 0 | No sharing. `R1 = 2u+3u+4u`, `R2 = 2u+2u+4u`, plus `P = 4e` | 17u + 4e |
| Only `T2` known (`T1` unknown) | `R2 = 8u` retained; `R1` by the lemma: `4u + 3u + 2u = 9u` | 17u (+4e) |
| Only `T1` known | `R1 = 9u`; `R2` by the lemma `= 9u` | 18u (+4e) |
| Neither known | lemma on both sides | 18u (+4e) |
| Neither known, `FP = either` with products | add `Σ_i rho_op(x_i)` on each side | 18u + 2·4u |

Knowing both orders allows exact cancellation of the shared prefix, which no single-program bound
can do. Knowing one order helps only when that order is better than the worst case. A sequential
chain *is* the worst case, so "only T1 known" gains nothing here. A pairwise or chunked order
gains a lot as `n` grows: `O(u·n log n)` against `O(u·n²)`.

## 7. When the certifier returns inconclusive

`certify()` reports `Status.INCONCLUSIVE` with the reason below, never a bound, when:

| Condition | Reason (`outcomes.Inconclusive`) |
|---|---|
| Match incomplete or ambiguous | `UNSUPPORTED_OPERATOR` |
| A unit has no transfer function: opaque op, an elementary function with no declared ULP assumption, a GELU idiom whose constants differ and no term covers the difference | `UNSUPPORTED_OPERATOR` |
| Precision is not established (any `HMMA`/TF32/fp16 path), a multi-term fused accumulation with no declared model, or an FP environment that cannot be pinned (rounding mode, FTZ unknown and the "both behaviors" bound not computed) | `UNSUPPORTED_PATH` |
| Run-to-run nondeterminism (atomics) without a set-valued order model | `UNSUPPORTED_PATH` |
| The kernel or tactic actually run differs from the pinned one at certification time | `UNSUPPORTED_PATH` |
| A range reaches the overflow threshold; a NaN is possible (`log` or `rsqrt` of a range that includes `<= 0`, division by a range that contains 0, `inf - inf`) | `EXCEPTIONAL_UNRESOLVED` |
| The final per-output `e` exceeds `ε` | `BOUND_ABOVE_EPSILON`; this is never `refuted` without a replayed witness |
| Analysis budget exhausted | `TIMEOUT` / `OUT_OF_MEMORY` |

Missing order evidence alone is **not** a reason to stop. It moves that kernel from Strategy 1 to
Strategy 2. The certifier stops only when Strategy 2 cannot be applied (no FP model at all), or
when its bound exceeds `ε`.

## 8. Implementation

| Module | Contents |
|---|---|
| `fpmodel.py` | formats (`u`, `eta`), `SumTree`, order models `Known` / `AnyTree` / `Depth` / `Chunks`, `known_tree_bound`, `sequential_bound`, `anytree_bound` (the lemma), `relational_diff`, `bound`, the CUDA fp32 ULP table |
| `fprev.py` | `reveal_order` (FPRev), exact IEEE emulation of a tree (`emulate`), `collect_evidence` (determinism plus bit-exact emulation on adversarial inputs, level `validated`), `OrderEvidence`, `matmul_sum_fn`, `detect_fma` |
| `absint.py` | `FloatErrorInterpreter` (transfer functions), `OrderOracle`, `InconclusiveError` |
| `backsub.py` | `BacksubAnalyzer` (`method="backsub"`, the default): per-element bounds by differential back-substitution (module docstring); the per-element accumulation bounds of sections 4, 5.2 and 5.4; the bitwise-equality rule of section 5.5; formal and probabilistic guarantees |
| `certifier.py` | `certify()` returns `CertificationResult`: `verified`, `bounded` (no epsilon) or `inconclusive` with a reason; `assumptions` and their digest; per-unit reports |

How a unit is interpreted (`FloatErrorInterpreter.transfer`):

1. **P** on program 1's canonical nodes, in exact arithmetic over the input ranges. At a GELU
   matched through an idiom with approximate constants, the distance between the idiom and the
   reference GELU is added for each side that uses it.
2. **R1, R2** on each program's *raw* lowering: its source graph lowered with no rewrite that
   changes rounding (no reassociation, scalar hoisting or idiom folding). Each operation adds its
   local rounding; earlier rounding propagates through Lipschitz bounds.
3. **Joint pass** when the two raw unit graphs are isomorphic: it bounds `|out1 - out2|`
   operation by operation and can cancel: a correctly rounded operation that both programs run
   on its own, with bitwise-equal inputs, gives bitwise-equal outputs; two known trees use
   `relational_diff`. The smaller of `P + R1 + R2` and the joint bound is kept, and the
   intersection of the two ranges.

Accumulations take their order from `OrderOracle`:

- Triton and TileLang GEMMs (linear, matmul, bmm, addmm, conv via im2col):
  `Known(sequential FMA chain from 0)`. With `split_k`, `Chunks` of such chains combined by an
  unknown-order `torch.sum`. The bias add is a separate rounding.
- Any `(backend, op)` with `OrderEvidence` in `config.order_evidence`, used at level `known`, or
  at level `validated` when `accept_validated_orders` is set. Each one goes into the assumptions.
- Everything else: `AnyTree` (`either` FMA for products). A bias from the same source operation
  becomes one more leaf, since cuBLAS-style epilogues can add it anywhere in the tree.
- Library convolutions (cuDNN, ONNX Runtime) are inconclusive (`UNSUPPORTED_PATH`): Winograd and
  FFT algorithms are not summation trees. `assume_sum_of_products_conv=True` records the
  assumption and treats them as `AnyTree`.

Other transfer functions:
- layout ops, `relu`, `max`, `min`, `where` and pooling-max are exact;
- `add`, `mul` and `div` round once, correctly;
- `exp`, `log`, `tanh`, `erf`, `rsqrt`, `sqrt`, `pow`, `sin` and `cos` use the ULP table;
- `sigmoid`, `silu`, `gelu` (erf and tanh forms), `softmax`, `log_softmax`, `layer_norm`,
  `rms_norm` and inference `batch_norm` use closed-form bounds for their standard evaluation;
  each records that evaluation as an assumption.

### Validation

- `tests/test_absint.py` (all pass):
  - the lemma, by enumerating every binary tree on up to 6 leaves;
  - the section 6 numbers;
  - known, any-tree, FMA, split-K and relational bounds against exact rational emulation;
  - FPRev on sequential, pairwise and strided kernels, and its rejection of value-dependent and
    random orders;
  - end-to-end bounds against PyTorch vs ONNX Runtime on sampled inputs;
  - the inconclusive paths.
- `test/test_fprev_gpu.py` (RTX 6000 Ada):
  - FPRev reveals a sequential chain for the Triton and TileLang kernels (K = 16, 48, 96), and
    the FMA probe confirms FMA. That matches the static order the oracle uses.
  - cuBLAS SGEMM reveals a different, non-sequential tree (depth 6 at K = 16, 26 at K = 96),
    with separate multiply and add. The oracle keeps cuBLAS unknown unless evidence is supplied.
- `test/test_certifier_models.py`: every bounded result is at least the error observed by
  `diff_testing` on 32 samples at the certified batch size. No pair was unsound.

### Current precision

`method="backsub"` (the default), ResNet18, domain = one sample (`utils.local_domain(..., 0)`),
RTX 6000 Ada, 2026-11-11. "Before" is the back-substitution analysis without sections 5.4 and
5.5 and with `backsub_tol = 1e-2`.

64×64 input:

| Pair | Observed | Formal, before | Formal | Probabilistic, before | Probabilistic |
|---|---|---|---|---|---|
| model vs eager | 0 | 2.9e9 | **0** | 5.5e-2 | **0** |
| Triton vs TileLang | 0 | 1.8e7 | **0** | 2.7e-2 | **0** |
| Triton vs Triton `split_k=4` | 7.6e-6 | 1.2e7 | 0.14 | 3.6e-2 | 3.8e-4 |
| model vs Triton | 1.3e-5 | 2.2e9 | 7.6e5 | 4.7e-2 | 1.4e-3 |

224×224 input:

| Pair | Observed | Formal | Probabilistic |
|---|---|---|---|
| model vs eager | 0 | 0 | 0 |
| Triton vs TileLang | 0 | 0 | not run |
| Triton vs Triton `split_k=4` | 5.7e-6 | 3.7 | not run |
| model vs Triton | 5.7e-6 | 5.5e8 | 1.0e-3 |
| model vs Triton `split_k=4` | 3.3e-6 | not run | 1.0e-3 |
| model vs TileLang | 5.7e-6 | not run | 1.0e-3 |
| SqueezeNet, model vs Triton | 3.8e-6 | not run | 3.7e-4 |

A 224×224 pair takes 8 to 18 minutes.

**Against exact arithmetic** (`reference="exact"`, section 5.6; formal; observed = error
against the model run in float64; `test/test_certifier_models.py --radius 0 --configure
'{"reference": "exact"}'`):

| Program | 64×64 bound | 64×64 observed | 224×224 bound | 224×224 observed |
|---|---|---|---|---|
| Triton `split_k=4` | 3.2e-2 | 3.6e-6 | 6.9e-2 | 2.9e-6 |
| Triton / TileLang (one chain) | 8.1e-2 | 7.8e-6 | 5.0e-1 | 3.6e-6 |
| eager (unknown order) | 6.8e5 | 7.9e-6 | 7.9e7 | 4.0e-6 |
| ONNX Runtime optimized graph (unknown order) | not run | | 1.3e8 | not measured |

This is the formal answer for "model vs compiled": a compiled program with known orders gets a
finite bound that does not depend on the eager kernels. It is still about 10⁴ to 10⁵ above the
observed error, which is the cost of assuming every rounding error has the worst sign in every
node, layer and element. `split_k=4` certifies tighter than one chain because four short
chains have smaller partial sums.

One layer, per element, bound / observed error against float64 (`test/test_layer_bounds.py`,
median over the output; no element violates the bound):

| Layer | Triton chain, magnitudes | Triton chain, signed (5.4) | eager, magnitudes | eager, signed |
|---|---|---|---|---|
| conv 7×7, K=147 | 340 | 27 to 37 | 550 to 620 | 270 to 290 |
| conv 3×3, K=576 | 1400 | 53 to 75 | 2400 to 2500 | 1100 to 1200 |
| conv 3×3, K=4608 | 11000 | 170 to 230 | 20000 to 22000 | 7700 to 9000 |
| linear, K=512 | 1100 to 1200 | 50 to 69 | 5400 to 5900 | 2600 to 3100 |
| linear, K=4096 | 9300 to 10000 | 130 to 200 | 60000 to 64000 | 22000 to 23000 |

What this shows:

- **Pairs that agree bitwise now certify at 0** (section 5.5).
- **Known chains gain 9× to 70× per layer** from signed prefix sums, more for larger K; unknown
  orders gain 2× to 3×.
- **The two-program formal bound is still useless whenever the reference order is unknown**
  ("model vs X"); use `reference="exact"` there. Every eager convolution injects the any-tree bound, the per-element error bound reaches
  the size of the activations about two thirds of the way through the network, ReLUs become
  unstable, and the bound cascades. With both orders known the cascade does not start at 64×64
  (0.14) and only starts in the last layers at 224×224 (3.7).
- **The probabilistic bound is 50× to 310× the observed error.**
- `backsub_tol = 1e-2` let intermediate bounds grow to 1% of the activation before tightening
  them, which was enough to start the cascade (2.9e-2 instead of 1.4e-3 for model vs Triton,
  probabilistic). The default is now `1e-4`, which gives the same bound as `0`.
- Windowed back-substitution is much looser: 8.7e15 with `backsub_window=2` at 224×224, against
  2.3e11 for the older unwindowed analysis of the same pair. It is not a way to save time.

Not solved:

- **Transformers** (DistilBERT, one sample) stop with `unit u12: cannot bind raw input`, a
  limitation of the per-program pass, before any bound is produced.
- **L-inf balls**: at radius 1/255 the input-variation term `V` explodes for ResNet18 (2e15).
- **Order evidence for the eager reference**, the one thing that would make "model vs X"
  formal bounds finite. A probe (FPRev masks on the real kernels, batched) found:
  - cuDNN convolutions with 3 input channels (3×3, 7×7) and 1×1 convolutions: a sequential
    chain, reproduced bit for bit by the emulator;
  - cuDNN 3×3 convolutions with 16 or 64 input channels: the masked inputs return `inf` or
    `nan`, so the algorithm is **not a summation tree of the products** (Winograd-like). The
    `assume_sum_of_products_conv` assumption is therefore doubtful for exactly these layers;
  - cuBLAS `linear`: a tree is revealed, but the emulator does not reproduce the kernel on
    most adversarial inputs, so the order cannot be treated as fixed.
  The probe used batched calls, which can select a different algorithm than batch 1.

### Remaining assumptions and limits

- Bound arithmetic is float64 with a relative margin `SAFETY = 1 + 2^-40`. There is no
  independent rational re-check yet (`stubs.CertificateChecker`).
- The DSL kernel order is taken from the kernel source. FPRev corroborates it at three K values
  on one GPU; the PTX/SASS is not checked automatically.
- Reductions inside composite ops (softmax, norms) are bounded for any order, but their
  *algorithm* (max-subtracted exp; two-pass or Welford variance; rsqrt) is assumed and listed in
  `result.assumptions`.
- The certificate covers the example-input shapes (batch size, sequence length) only.
