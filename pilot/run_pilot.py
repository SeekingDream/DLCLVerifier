#!/usr/bin/env python3
"""Exploratory pilot. Evidence class: EMPIRICAL (except P5, which is SMT).

Every "max observed" number is a LOWER bound on the true worst case over the
input domain. Nothing here verifies anything. "Naive bound" values are computed
in float64 without outward rounding, so they are estimates, not certificates.
"""
import argparse, json, math, os, platform, sys, time

os.environ.setdefault("XLA_FLAGS", "")
import numpy as np

U32 = 2.0 ** -24  # unit roundoff, binary32 round-to-nearest


def rec_versions():
    import jax, onnxruntime, torch
    return dict(python=platform.python_version(), machine=platform.machine(),
                processor=platform.processor(), numpy=np.__version__, torch=torch.__version__,
                jax=jax.__version__, onnxruntime=onnxruntime.__version__)


def p1_summation(seed, n=1_000_000, trials=20):
    """Same multiset of floats, different summation trees and systems."""
    import jax, jax.numpy as jnp, torch
    jit_sum = jax.jit(jnp.sum)
    rng = np.random.default_rng(seed)
    worst = {}
    bits_equal_all = 0
    bound = []
    for _ in range(trials):
        x = rng.standard_normal(n).astype(np.float32)
        exact = math.fsum(float(v) for v in x)
        vals = {
            "sequential_loop": float(np.cumsum(x, dtype=np.float32)[-1]),
            "numpy_sum": float(np.sum(x, dtype=np.float32)),
            "jax_jit_sum": float(jit_sum(x)),
            "torch_sum": float(torch.from_numpy(x).sum()),
        }
        if len(set(vals.values())) == 1:
            bits_equal_all += 1
        bound.append((n - 1) * U32 * float(np.abs(x.astype(np.float64)).sum()))
        for k, v in vals.items():
            worst[k] = max(worst.get(k, 0.0), abs(v - exact))
        ks = list(vals)
        for i, a in enumerate(ks):
            for b in ks[i + 1:]:
                key = f"{a}__vs__{b}"
                worst[key] = max(worst.get(key, 0.0), abs(vals[a] - vals[b]))
    return dict(n=n, trials=trials, all_systems_bit_equal_trials=bits_equal_all,
                order_agnostic_bound_first_order=max(bound), max_abs=worst)


def make_mlp(seed, d=64, width=128, depth=4, k=10):
    rng = np.random.default_rng(seed)
    dims = [d] + [width] * depth + [k]
    return [((rng.standard_normal((o, i)) / math.sqrt(i)).astype(np.float32),
             (0.1 * rng.standard_normal(o)).astype(np.float32))
            for i, o in zip(dims[:-1], dims[1:])]


def naive_bound(layers, input_hi=1.0):
    """Domain-wide triangle bound for x in [0,1]^d: 2 * e, e = |M_f - M_R|.

    m: magnitude envelope. e: forward error. gamma_{n+1} for an n-term dot+add.
    """
    d = layers[0][0].shape[1]
    m = np.full(d, input_hi, dtype=np.float64)
    e = np.zeros(d, dtype=np.float64)
    for w, b in layers:
        W = np.abs(w.astype(np.float64))
        B = np.abs(b.astype(np.float64))
        k = W.shape[1] + 2
        gamma = k * U32 / (1 - k * U32)
        m_new = W @ m + B
        e = W @ e + gamma * m_new
        m = m_new
    return 2.0 * float(e.max())


def onnx_mlp(layers):
    import onnx
    from onnx import TensorProto as T, helper as h, numpy_helper as nh
    nodes, inits, cur = [], [], "x"
    for i, (w, b) in enumerate(layers):
        inits += [nh.from_array(w.T, f"w{i}"), nh.from_array(b, f"b{i}")]
        nodes.append(h.make_node("MatMul", [cur, f"w{i}"], [f"m{i}"]))
        nodes.append(h.make_node("Add", [f"m{i}", f"b{i}"], [f"a{i}"]))
        cur = f"a{i}"
        if i + 1 < len(layers):
            nodes.append(h.make_node("Relu", [cur], [f"r{i}"]))
            cur = f"r{i}"
    nodes.append(h.make_node("Identity", [cur], ["y"]))
    g = h.make_graph(nodes, "mlp",
                     [h.make_tensor_value_info("x", T.FLOAT, [None, layers[0][0].shape[1]])],
                     [h.make_tensor_value_info("y", T.FLOAT, [None, layers[-1][0].shape[0]])], inits)
    m = h.make_model(g, opset_imports=[h.make_opsetid("", 17)])
    m.ir_version = 9
    onnx.checker.check_model(m)
    return m.SerializeToString()


def ort_run(model_bytes, x, level, threads=1, cfg=None):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = getattr(ort.GraphOptimizationLevel, level)
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    for k, v in (cfg or {}).items():
        so.add_session_config_entry(k, v)
    s = ort.InferenceSession(model_bytes, so, providers=["CPUExecutionProvider"])
    return s.run(["y"], {"x": x})[0]


def linf(a, b):
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def p2_mlp(seed, inputs=20000):
    import jax, jax.numpy as jnp, torch
    layers = make_mlp(seed)
    x = np.random.default_rng(seed + 1).random((inputs, layers[0][0].shape[1]), dtype=np.float32)

    def ref64(x):
        h = x.astype(np.float64)
        for i, (w, b) in enumerate(layers):
            h = h @ w.T.astype(np.float64) + b.astype(np.float64)
            if i + 1 < len(layers):
                h = np.maximum(h, 0.0)
        return h

    def jf(h):
        for i, (w, b) in enumerate(layers):
            h = h @ jnp.asarray(w.T) + jnp.asarray(b)
            if i + 1 < len(layers):
                h = jax.nn.relu(h)
        return h

    def tf(h):
        for i, (w, b) in enumerate(layers):
            h = h @ torch.from_numpy(w.T.copy()) + torch.from_numpy(b.copy())
            if i + 1 < len(layers):
                h = torch.relu(h)
        return h

    bm = onnx_mlp(layers)
    with torch.no_grad():
        t1 = tf(torch.from_numpy(x)).numpy()
        t2 = torch.jit.trace(tf, torch.from_numpy(x[:4]))(torch.from_numpy(x)).numpy()
    out = {
        "torch_eager": t1, "torchscript_traced": t2,
        "jax_jit": np.asarray(jax.jit(jf)(x)),
        "ort_disable_all": ort_run(bm, x, "ORT_DISABLE_ALL"),
        "ort_enable_all": ort_run(bm, x, "ORT_ENABLE_ALL"),
    }
    with jax.disable_jit():
        out["jax_eager"] = np.asarray(jf(x))
    ref = ref64(x)
    ks = list(out)
    pair, gt = {}, {}
    for k in ks:
        gt[k] = linf(out[k], ref)
    for i, a in enumerate(ks):
        for b in ks[i + 1:]:
            pair[f"{a}__vs__{b}"] = dict(linf=linf(out[a], out[b]),
                                         identical_outputs=bool(np.array_equal(out[a], out[b])),
                                         inputs_with_any_bit_diff=int(np.any(out[a] != out[b], axis=1).sum()))
    nb = naive_bound(layers)
    best = max(v["linf"] for v in pair.values())
    return dict(shape=[64, 128, 128, 128, 128, 10], inputs=inputs, max_error_vs_float64_reference=gt,
                pairs=pair, naive_domain_bound_estimate=nb, max_pair_observed=best,
                bound_over_observed=(nb / best if best else None))


def conv_bn_model(seed, ch=16):
    """Conv + BatchNorm -> eligible for ORT's level-1/2 Conv-BN fusion."""
    import onnx
    from onnx import TensorProto as T, helper as h, numpy_helper as nh
    r = np.random.default_rng(seed)
    w = (r.standard_normal((ch, 3, 3, 3)) * 0.1).astype(np.float32)
    b = (r.standard_normal(ch) * 0.1).astype(np.float32)
    s = (0.5 + r.random(ch)).astype(np.float32)
    mu = (0.1 * r.standard_normal(ch)).astype(np.float32)
    var = (0.5 + r.random(ch)).astype(np.float32)
    bb = (0.05 * r.standard_normal(ch)).astype(np.float32)
    inits = [nh.from_array(v, n) for v, n in [(w, "w"), (b, "b"), (s, "s"), (bb, "bb"), (mu, "mu"), (var, "var")]]
    n = [h.make_node("Conv", ["x", "w", "b"], ["c"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
         h.make_node("BatchNormalization", ["c", "s", "bb", "mu", "var"], ["y"], epsilon=1e-5)]
    g = h.make_graph(n, "convbn", [h.make_tensor_value_info("x", T.FLOAT, [None, 3, 32, 32])],
                     [h.make_tensor_value_info("y", T.FLOAT, [None, ch, 32, 32])], inits)
    m = h.make_model(g, opset_imports=[h.make_opsetid("", 17)])
    m.ir_version = 9
    onnx.checker.check_model(m)
    return m.SerializeToString()


def p3_conv_bn(seed, inputs=256):
    bm = conv_bn_model(seed)
    x = np.random.default_rng(seed + 2).random((inputs, 3, 32, 32), dtype=np.float32)
    a = ort_run(bm, x, "ORT_DISABLE_ALL")
    b = ort_run(bm, x, "ORT_ENABLE_ALL")
    ulp = np.abs(a.astype(np.float64) - b.astype(np.float64)) / np.spacing(np.abs(a)).astype(np.float64)
    import onnx, onnxruntime as ort, tempfile
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    with tempfile.TemporaryDirectory() as td:
        so.optimized_model_filepath = os.path.join(td, "o.onnx")
        ort.InferenceSession(bm, so, providers=["CPUExecutionProvider"])
        optimized_ops = [n.op_type for n in onnx.load(so.optimized_model_filepath).graph.node]
    return dict(inputs=inputs, optimized_ops=optimized_ops, linf=linf(a, b), identical_outputs=bool(np.array_equal(a, b)),
                fraction_elements_differ=float(np.mean(a != b)), max_ulp=float(ulp.max()),
                median_ulp_of_differing=float(np.median(ulp[a != b])) if np.any(a != b) else 0.0)


def p4_subnormal(seed):
    """Absolute vs relative metrics when FTZ/DAZ is really applied (torch sets MXCSR).

    ORT's session.set_denormal_as_zero is also tried. Single-threaded fast paths may
    never hit the configured pool, so a zero result there is not evidence of safety.
    """
    import onnx, torch
    from onnx import TensorProto as T, helper as h
    x = np.full((2048, 8), 1e-40, dtype=np.float32)
    t = torch.from_numpy(x)
    torch.set_num_threads(1)
    base = (t + t).numpy().copy()
    applied = torch.set_flush_denormal(True)
    ftz = (t + t).numpy().copy()
    torch.set_flush_denormal(False)
    d = np.abs(base.astype(np.float64) - ftz.astype(np.float64))
    n = [h.make_node("Add", ["x", "x"], ["y"])]
    g = h.make_graph(n, "add", [h.make_tensor_value_info("x", T.FLOAT, [None, 8])],
                     [h.make_tensor_value_info("y", T.FLOAT, [None, 8])])
    m = h.make_model(g, opset_imports=[h.make_opsetid("", 17)])
    m.ir_version = 9
    bm = m.SerializeToString()
    o0 = ort_run(bm, x, "ORT_ENABLE_ALL", threads=4)
    o1 = ort_run(bm, x, "ORT_ENABLE_ALL", threads=4, cfg={"session.set_denormal_as_zero": "1"})
    return dict(torch_ftz_supported=bool(applied), torch_linf=float(d.max()),
                torch_flushed_fraction=float(np.mean(ftz == 0)),
                torch_max_relative_error=float((d / np.abs(base.astype(np.float64))).max()),
                ort_session_option_changed_output=bool(not np.array_equal(o0, o1)))


def p5_threads(seed):
    """Thread count and repeat stability."""
    layers = make_mlp(seed, d=512, width=1024, depth=3)
    bm = onnx_mlp(layers)
    x = np.random.default_rng(seed + 3).random((256, 512), dtype=np.float32)
    r = {}
    runs = {t: [ort_run(bm, x, "ORT_ENABLE_ALL", threads=t) for _ in range(5)] for t in (1, 2, 4, 8)}
    r["same_thread_repeats_identical"] = {t: all(np.array_equal(v[0], w) for w in v) for t, v in runs.items()}
    r["vs_1_thread_linf"] = {t: linf(runs[1][0], v[0]) for t, v in runs.items()}
    import torch
    z = torch.from_numpy(np.random.default_rng(seed + 4).standard_normal(5_000_000).astype(np.float32))
    v = {}
    for t in (1, 2, 4, 8):
        torch.set_num_threads(t)
        v[t] = [float(z.sum()) for _ in range(5)]
    torch.set_num_threads(1)
    r["torch_sum_by_threads"] = {t: sorted(set(w)) for t, w in v.items()}
    return r


def p6_smt(timeout_s):
    """z3 bit-blasted FP on tiny relational queries. Honest sat/unsat/unknown."""
    import z3
    z3.set_param("timeout", int(timeout_s * 1000))
    rm = z3.RNE()
    F = z3.Float32()
    res = []

    def box(s, vs):
        lo, hi = z3.FPVal(-1.0, F), z3.FPVal(1.0, F)
        for v in vs:
            s.add(z3.fpLEQ(lo, v), z3.fpLEQ(v, hi))

    def check(name, vs, lhs, rhs, eps_exp):
        # Compare in Float64: |l-r| is exact for these exponent ranges in binary64 unless
        # the operands are far apart, and is itself rounded; pilot-only encoding.
        D = z3.Float64()
        l, r = z3.fpFPToFP(rm, lhs, D), z3.fpFPToFP(rm, rhs, D)
        s = z3.Solver()
        box(s, vs)
        eps = z3.FPVal(2.0 ** eps_exp, D)
        diff = z3.fpAbs(z3.fpSub(rm, l, r))
        s.add(z3.fpGT(diff, eps))
        t = time.time()
        a = str(s.check())
        res.append(dict(query=name, eps=f"2^{eps_exp}", z3=a,
                        outcome={"unsat": "VERIFIED(bound holds, modulo encoding)", "sat": "REFUTED", "unknown": "INCONCLUSIVE"}[a],
                        seconds=round(time.time() - t, 2)))

    a, b, c, d = z3.FPs("a b c d", F)
    fma = z3.fpFMA(rm, a, b, c)
    mulAdd = z3.fpAdd(rm, z3.fpMul(rm, a, b), c)
    for e in (-25, -22):
        check("fma(a,b,c) vs (a*b)+c", [a, b, c], fma, mulAdd, e)
    s1 = z3.fpAdd(rm, z3.fpAdd(rm, a, b), z3.fpAdd(rm, c, d))
    s2 = z3.fpAdd(rm, z3.fpAdd(rm, z3.fpAdd(rm, a, b), c), d)
    for e in (-26, -20):
        check("(a+b)+(c+d) vs ((a+b)+c)+d", [a, b, c, d], s1, s2, e)
    return dict(z3=z3.get_version_string(), timeout_s=timeout_s, results=res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20261031)
    ap.add_argument("--smt-timeout", type=float, default=120)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    steps = dict(p1_summation=lambda: p1_summation(a.seed), p2_mlp=lambda: p2_mlp(a.seed),
                 p3_conv_bn=lambda: p3_conv_bn(a.seed), p4_subnormal=lambda: p4_subnormal(a.seed),
                 p5_threads=lambda: p5_threads(a.seed), p6_smt=lambda: p6_smt(a.smt_timeout))
    r = dict(evidence_class="empirical; P6 is SMT on a pilot encoding", seed=a.seed, versions=rec_versions())
    for k, f in steps.items():
        if a.only and k not in a.only.split(","):
            continue
        t = time.time()
        try:
            r[k] = f()
        except Exception as e:  # keep failures visible in the record
            r[k] = dict(error=f"{type(e).__name__}: {e}")
        r[k + "_wall_s"] = round(time.time() - t, 2)
        print(k, json.dumps(r[k], indent=1)[:1800], flush=True)
    with open(a.out, "w") as f:
        json.dump(r, f, indent=2, sort_keys=True, default=str)


if __name__ == "__main__":
    sys.exit(main())
