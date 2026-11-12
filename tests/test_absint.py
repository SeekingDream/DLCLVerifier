"""Floating-point bounds, FPRev, and the abstract interpreter on small CPU programs.

Soundness is checked against exact rational arithmetic (emulated IEEE trees) and against the
real programs (PyTorch vs ONNX Runtime on sampled inputs)."""
import math
from fractions import Fraction

import numpy as np
import pytest

from dlclv import fpmodel as fpm
from dlclv.fprev import (RevealError, collect_evidence, emulate, reveal_order)
from dlclv.outcomes import Inconclusive

F32 = fpm.FORMATS["float32"]
U = F32.u


# --------------------------------------------------------------------------------------------
# Trees and bounds
# --------------------------------------------------------------------------------------------

def all_trees(leaves):
    """Every full binary tree over the set `leaves` (unordered children)."""
    leaves = tuple(leaves)
    if len(leaves) == 1:
        yield leaves[0]
        return
    first, rest = leaves[0], leaves[1:]
    n = len(rest)
    for mask in range(1 << n):
        left = (first,) + tuple(rest[i] for i in range(n) if mask >> i & 1)
        right = tuple(rest[i] for i in range(n) if not mask >> i & 1)
        if not right:
            continue
        for a in all_trees(left):
            for b in all_trees(right):
                yield (a, b)


def canon(t):
    return t if isinstance(t, int) else frozenset(canon(c) for c in t)


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_lemma_any_tree_bound_covers_every_tree(n):
    rng = np.random.default_rng(n)
    X = rng.exponential(size=(n, 3)) * 10.0 ** rng.integers(-3, 3, size=(n, 3))
    for products in (False, True):
        top = fpm.anytree_bound(X, F32, products)
        worst = np.zeros(3)
        for t in all_trees(range(n)):
            tree = fpm.SumTree.from_nested(t, n)
            for fp in (("fma", "separate") if products else ("separate",)):
                b = fpm.known_tree_bound(tree, X, F32, products, fp)
                assert np.all(b <= top * (1 + 1e-12)), (t, fp)
                worst = np.maximum(worst, b)
        if not products:
            # attained: sequential summation in decreasing magnitude, per column
            for c in range(3):
                order = list(np.argsort(-X[:, c]))
                seq = fpm.SumTree.sequential(n, order=order, zero_init=False)
                b = fpm.known_tree_bound(seq, X, F32, False)[c]
                assert b == pytest.approx(top[c], rel=1e-12)


def test_four_term_reduction_example():
    X = np.ones(4)
    t1 = fpm.SumTree.from_nested((((0, 1), 2), 3))
    t2 = fpm.SumTree.from_nested(((0, 1), (2, 3)))
    k = lambda t: float(fpm.known_tree_bound(t, X, F32, False)[0]) / U  # noqa: E731
    assert k(t1) == pytest.approx(9, rel=1e-5) and k(t2) == pytest.approx(8, rel=1e-5)
    rel, shared = fpm.relational_diff(t1, t2, X, F32, False, "separate", "separate")
    assert shared == 1 and float(rel[0]) / U == pytest.approx(13, rel=1e-5)
    anyb = float(fpm.anytree_bound(X, F32, False)[0]) / U
    assert anyb == pytest.approx(9, rel=1e-5)
    assert k(t2) + anyb == pytest.approx(17, rel=1e-5)        # only T2 known
    assert k(t1) + anyb == pytest.approx(18, rel=1e-5)        # only T1 known
    assert 2 * anyb == pytest.approx(18, rel=1e-5)            # neither known


def test_sequential_fast_path_matches_tree():
    rng = np.random.default_rng(0)
    X = rng.exponential(size=(37, 5))
    for fp in ("fma", "separate"):
        for zero in (True, False):
            t = fpm.SumTree.sequential(37, zero_init=zero)
            slow = fpm.known_tree_bound(t, X, F32, True, fp)
            fast = fpm.bound(fpm.Known(t, fp), X, F32, True)
            assert np.allclose(slow, fast, rtol=1e-12)


def exact_sum(xs):
    return sum((Fraction(float(v)) for v in xs), Fraction(0))


def rand_inputs(rng, n):
    kind = rng.integers(4)
    x = rng.standard_normal(n)
    if kind == 1:
        x *= 2.0 ** rng.integers(-20, 20, n)
    elif kind == 2:
        x[: n // 2] = -x[n // 2: 2 * (n // 2)] * (1 + 2.0 ** -18)
    elif kind == 3:
        x = np.abs(x)
    return x.astype(np.float32)


def test_tree_bounds_hold_for_emulated_sums():
    rng = np.random.default_rng(1)
    n = 12
    trees = [fpm.SumTree.sequential(n, zero_init=False), fpm.SumTree.pairwise(n),
             fpm.SumTree.from_nested(((0, 1, ), (2, (3, (4, 5)))), 6)]
    for _ in range(60):
        x = rand_inputs(rng, n)
        X = np.abs(x.astype(np.float64))
        exact = exact_sum(x)
        anyb = float(fpm.anytree_bound(X, F32, False)[0])
        for t in trees[:2]:
            got = Fraction(emulate(t, x))
            err = abs(float(got - exact))
            assert err <= float(fpm.known_tree_bound(t, X, F32, False)[0])
            assert err <= anyb


def test_dot_product_bounds_with_fma():
    rng = np.random.default_rng(2)
    n = 10
    seq = fpm.SumTree.sequential(n, zero_init=True)
    pw = fpm.SumTree.pairwise(n)
    for _ in range(40):
        a = rand_inputs(rng, n)
        b = rand_inputs(rng, n)
        prods = list(zip(a, b))
        exact = sum((Fraction(float(p)) * Fraction(float(q)) for p, q in prods), Fraction(0))
        X = np.abs(a.astype(np.float64) * b.astype(np.float64))
        anyb = float(fpm.anytree_bound(X, F32, True)[0])
        for t in (seq, pw):
            for fp in ("fma", "separate"):
                got = Fraction(emulate(t, None, products=prods, fp=fp))
                err = abs(float(got - exact))
                assert err <= float(fpm.bound(fpm.Known(t, fp), X, F32, True)[0])
                assert err <= anyb


def test_relational_bound_holds_between_two_trees():
    rng = np.random.default_rng(3)
    n = 8
    t1 = fpm.SumTree.from_nested(((((0, 1), 2), 3), (4, (5, (6, 7)))))
    t2 = fpm.SumTree.from_nested((((0, 1), 2), (3, ((4, 5), (6, 7)))))
    for _ in range(50):
        x = rand_inputs(rng, n)
        X = np.abs(x.astype(np.float64))
        d = abs(emulate(t1, x) - emulate(t2, x))
        rel, shared = fpm.relational_diff(t1, t2, X, F32, False, "separate", "separate")
        assert shared >= 1
        assert d <= float(rel[0])
        assert float(rel[0]) <= float(fpm.known_tree_bound(t1, X, F32, False)[0]
                                      + fpm.known_tree_bound(t2, X, F32, False)[0])


def test_identical_known_trees_cancel_exactly():
    t = fpm.SumTree.sequential(16)
    rel, note = fpm.relational(fpm.Known(t, "fma"), fpm.Known(fpm.SumTree.sequential(16), "fma"),
                               np.ones(16), F32, True)
    assert float(rel[0]) == 0.0 and "bitwise" in note


def test_chunks_bound_covers_split_k():
    rng = np.random.default_rng(4)
    K, S = 12, 3
    ranges = tuple((s * 4, s * 4 + 4) for s in range(S))
    order = fpm.Chunks(ranges, tuple(fpm.Known(fpm.SumTree.sequential(4), "fma")
                                     for _ in ranges), fpm.AnyTree("separate"))
    for _ in range(30):
        a, b = rand_inputs(rng, K), rand_inputs(rng, K)
        exact = sum((Fraction(float(p)) * Fraction(float(q)) for p, q in zip(a, b)), Fraction(0))
        partial = [Fraction(emulate(fpm.SumTree.sequential(4), None,
                                    products=list(zip(a[i:j], b[i:j])), fp="fma"))
                   for i, j in ranges]
        for comb in all_trees(range(S)):
            got = Fraction(emulate(fpm.SumTree.from_nested(comb, S),
                                   [float(p) for p in partial]))
            X = np.abs(a.astype(np.float64) * b.astype(np.float64))
            assert abs(float(got - exact)) <= float(fpm.bound(order, X, F32, True)[0])


# --------------------------------------------------------------------------------------------
# FPRev
# --------------------------------------------------------------------------------------------

def seq_sum(x):
    s = np.float32(0)
    for v in x:
        s = np.float32(s + v)
    return float(s)


def pairwise_sum(x):
    x = [np.float32(v) for v in x]
    while len(x) > 1:
        y = [np.float32(x[i] + x[i + 1]) for i in range(0, len(x) - 1, 2)]
        if len(x) % 2:
            y.append(x[-1])
        x = y
    return float(x[0])


def strided_sum(x, lanes=4):
    acc = [np.float32(0)] * lanes
    for i, v in enumerate(x):
        acc[i % lanes] = np.float32(acc[i % lanes] + v)
    return float(np.float32(np.float32(acc[0] + acc[1]) + np.float32(acc[2] + acc[3])))


@pytest.mark.parametrize("fn,expected", [
    (seq_sum, lambda n: fpm.SumTree.sequential(n, zero_init=False)),
    (pairwise_sum, fpm.SumTree.pairwise),
])
def test_fprev_reveals_known_trees(fn, expected):
    n = 9
    got = reveal_order(fn, n)
    assert canon(got.to_nested()) == canon(expected(n).to_nested())


def test_fprev_strided_kernel_and_evidence():
    n = 12
    t = reveal_order(strided_sum, n)
    rng = np.random.default_rng(0)
    for _ in range(20):
        x = rand_inputs(rng, n)
        assert emulate(t, x) == strided_sum(x)
    ev = collect_evidence(strided_sum, n, kernel="strided", trials=25)
    assert ev.level == "validated" and ev.order().tree == ev.tree


def test_fprev_rejects_value_dependent_and_random_orders():
    def sorted_sum(x):
        return seq_sum(sorted(x, key=abs))

    with pytest.raises(RevealError):
        collect_evidence(sorted_sum, 8, trials=20)

    rng = np.random.default_rng(0)

    def random_order(x):
        return seq_sum(np.asarray(x)[rng.permutation(len(x))])

    with pytest.raises(RevealError):
        collect_evidence(random_order, 8, trials=20)


# --------------------------------------------------------------------------------------------
# The interpreter on real programs
# --------------------------------------------------------------------------------------------

torch = pytest.importorskip("torch")
nn = torch.nn

from dlclv.certifier import TensorProgramCertifier  # noqa: E402
from dlclv.frontends import TorchExportFrontend  # noqa: E402


class Box:
    def __init__(self, *shapes, lo=-1.0, hi=1.0):
        self.shapes, self.lower, self.upper = shapes, [lo] * len(shapes), [hi] * len(shapes)
        self.lo, self.hi = lo, hi

    def sample(self, n, generator=None):
        return tuple(torch.rand((n, *s), generator=generator) * (self.hi - self.lo) + self.lo
                     for s in self.shapes)


def ex(*shapes, n=2):
    g = torch.Generator().manual_seed(0)
    return tuple(torch.rand((n, *s), generator=g) * 2 - 1 for s in shapes)


def onnx_file(model, x, tmp_path):
    pytest.importorskip("onnxruntime")
    p = str(tmp_path / "m.onnx")
    torch.onnx.export(model, x, p, input_names=[f"input_{i}" for i in range(len(x))],
                      opset_version=17, dynamo=False)
    return p


def observed_vs_onnx(model, path, box, x, trials=300):
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    g = torch.Generator().manual_seed(1)
    worst = 0.0
    with torch.no_grad():
        for _ in range(trials):
            xs = box.sample(x[0].shape[0], g)
            a = model(*xs).numpy()
            b = sess.run(None, {f"input_{i}": t.numpy() for i, t in enumerate(xs)})[0]
            worst = max(worst, float(np.max(np.abs(a.astype(np.float64) - b))))
    return worst


def mlp(act=None):
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 16), act or nn.ReLU(), nn.Linear(16, 4),
                         nn.Softmax(-1)).eval()


def test_certified_bound_covers_onnx_runtime(tmp_path):
    model, x, box = mlp(), ex((8,)), Box((8,))
    path = onnx_file(model, x, tmp_path)
    res = TensorProgramCertifier(model, path, box, {"example_inputs": x}).certify()
    assert res.status == "bounded", res.summary()
    bound = res.certified_bound
    assert 0 < bound < 1e-3
    assert observed_vs_onnx(model, path, box, x) <= bound
    t = res.output_triples[0]
    assert t.min_val >= 0 and t.max_val <= 1 + 1e-6
    assert res.assumptions and res.assumptions_digest


def test_epsilon_gives_verified_or_bound_above_epsilon(tmp_path):
    model, x, box = mlp(), ex((8,)), Box((8,))
    path = onnx_file(model, x, tmp_path)
    ok = TensorProgramCertifier(model, path, box, {"example_inputs": x, "epsilon": 1e-3})
    assert ok.certify().status == "verified"
    tight = TensorProgramCertifier(model, path, box, {"example_inputs": x, "epsilon": 1e-12})
    r = tight.certify()
    assert r.status == "inconclusive"
    assert r.inconclusive_reason is Inconclusive.BOUND_ABOVE_EPSILON
    assert r.output_triples is not None and r.certified_bound > 1e-12


class GeluExpr(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.fc = nn.Linear(8, 8)

    def forward(self, x):
        y = self.fc(x)
        return 0.5 * y * (1 + torch.erf(y * 0.7071067811865476))


class GeluOp(GeluExpr):
    def forward(self, x):
        return torch.nn.functional.gelu(self.fc(x))


def test_gelu_idiom_gap_is_part_of_the_bound():
    x, box = ex((8,)), Box((8,), lo=-3, hi=3)
    res = TensorProgramCertifier(GeluExpr().eval(), GeluOp().eval(), box,
                                 {"example_inputs": x}).certify()
    assert res.status == "bounded", res.summary()
    g = torch.Generator().manual_seed(1)
    worst = 0.0
    with torch.no_grad():
        for _ in range(200):
            xs = box.sample(2, g)
            worst = max(worst, float((GeluExpr()(*xs) - GeluOp()(*xs)).abs().max()))
    assert worst <= res.certified_bound


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.head = nn.Linear(4, 2)

    def forward(self, x):
        return self.head(torch.relu(self.conv(x)).mean((2, 3)))


def test_library_conv_needs_an_algorithm_assumption(tmp_path):
    model, x, box = ConvNet().eval(), ex((3, 6, 6)), Box((3, 6, 6), lo=0, hi=1)
    path = onnx_file(model, x, tmp_path)
    r = TensorProgramCertifier(model, path, box, {"example_inputs": x}).certify()
    assert r.status == "inconclusive"
    assert r.inconclusive_reason is Inconclusive.UNSUPPORTED_PATH
    assert "Winograd" in r.detail
    r = TensorProgramCertifier(model, path, box, {"example_inputs": x,
                                                  "assume_sum_of_products_conv": True}).certify()
    assert r.status == "bounded", r.summary()
    assert any("Winograd" in a for a in r.assumptions)
    assert observed_vs_onnx(model, path, box, x, trials=100) <= r.certified_bound


def test_exceptional_operation_is_inconclusive():
    class Log(nn.Module):
        def forward(self, x):
            return torch.log(x)

    r = TensorProgramCertifier(Log(), Log(), Box((4,)), {"example_inputs": ex((4,))}).certify()
    assert r.status == "inconclusive"
    assert r.inconclusive_reason is Inconclusive.EXCEPTIONAL_UNRESOLVED


def test_incomplete_match_is_inconclusive_without_a_bound():
    class A(nn.Module):
        def forward(self, x):
            return x / 3.0

    class B(nn.Module):
        def forward(self, x):
            return x * (1 / 3.0)

    r = TensorProgramCertifier(A(), B(), Box((4,)), {"example_inputs": ex((4,))}).certify()
    assert r.status == "inconclusive" and r.output_triples is None
    assert r.inconclusive_reason is Inconclusive.UNSUPPORTED_OPERATOR


def graph_with_impl(model, x, impl):
    return TorchExportFrontend().extract(model, example_inputs=x, impl=impl)


def test_known_dsl_orders_cancel_and_one_known_side_is_kept():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 8)).eval()
    x, box = ex((64,)), Box((64,))
    tri = graph_with_impl(model, x, {"backend": "triton", "split_k": 1, "block_k": 32})
    tl = graph_with_impl(model, x, {"backend": "tilelang"})
    eager = graph_with_impl(model, x, None)
    tri4 = graph_with_impl(model, x, {"backend": "triton", "split_k": 4, "block_k": 32})

    I = {"example_inputs": x, "method": "interval"}
    both = TensorProgramCertifier(tri, tl, box, I).certify()
    assert both.status == "bounded" and both.certified_bound == 0.0, both.summary()
    accs = [a for r in both.unit_reports for a in r.accumulations if "D" in a]
    assert accs and all(a["D"] == 0 for a in accs)

    one = TensorProgramCertifier(tri, eager, box, I).certify()
    none_ = TensorProgramCertifier(eager, eager, box, I).certify()
    split = TensorProgramCertifier(tri, tri4, box, I).certify()
    assert 0 < one.certified_bound <= none_.certified_bound
    assert 0 < split.certified_bound
    orders = {a.get("order_1") for r in one.unit_reports for a in r.accumulations}
    assert any("known sequential" in (o or "") for o in orders)
    orders = {a.get("order_2") for r in split.unit_reports for a in r.accumulations}
    assert any("chunks" in (o or "") for o in orders)


class Sum(nn.Module):
    def forward(self, x):
        return x.sum(-1)


def test_fprev_evidence_is_used_only_when_accepted():
    n = 16
    ev = collect_evidence(pairwise_sum, n, kernel="pairwise test kernel", trials=10)
    x, box = ex((n,)), Box((n,))
    base = dict(example_inputs=x, method="interval")
    plain = TensorProgramCertifier(Sum(), Sum(), box, base).certify()
    cfg = dict(base, order_evidence={("eager", "reduce_sum"): ev})
    ignored = TensorProgramCertifier(Sum(), Sum(), box, cfg).certify()
    assert ignored.certified_bound == plain.certified_bound          # validated is not enough
    accepted = TensorProgramCertifier(Sum(), Sum(), box,
                                      dict(cfg, accept_validated_orders=True)).certify()
    assert accepted.certified_bound < plain.certified_bound
    assert any("pairwise test kernel" in a for a in accepted.assumptions)
    ev.level = "known"
    ev.static_evidence = "test: kernel source is the pairwise loop"
    known = TensorProgramCertifier(Sum(), Sum(), box, cfg).certify()
    assert known.certified_bound == accepted.certified_bound


# --------------------------------------------------------------------------------------------
# Differential back-substitution (method="backsub", the default)
# --------------------------------------------------------------------------------------------

class Ball:
    """L-inf ball of radius r around x0 (a batched tuple), clipped to [lo, hi]."""

    def __init__(self, x0, r, lo=-math.inf, hi=math.inf):
        self.x0, self.r = x0, r
        self.lower = [torch.clamp(t - r, min=lo) for t in x0]
        self.upper = [torch.clamp(t + r, max=hi) for t in x0]

    def sample(self, n, generator=None):
        return tuple(lo + torch.rand(lo.shape, generator=generator) * (hi - lo)
                     for lo, hi in zip(self.lower, self.upper))


def observed(p1, p2, dom, trials=200):
    g = torch.Generator().manual_seed(3)
    worst = 0.0
    with torch.no_grad():
        for _ in range(trials):
            xs = dom.sample(1, g)
            worst = max(worst, float((p1(*xs) - p2(*xs)).abs().max()))
    return worst


class DeepCNN(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.c = nn.ModuleList([nn.Conv2d(3 if i == 0 else 8, 8, 3, padding=1) for i in range(4)])
        self.fc = nn.Linear(8, 5)

    def forward(self, x):
        for i, c in enumerate(self.c):
            x = torch.relu(c(x))
            if i == 1:
                x = torch.nn.functional.max_pool2d(x, 2)
        return self.fc(x.mean((2, 3)))


@pytest.mark.parametrize("radius", [0.0, 1 / 255, 0.05])
def test_backsub_is_sound_and_tighter_than_intervals(tmp_path, radius):
    model = DeepCNN().eval()
    x0 = (torch.rand(1, 3, 10, 10, generator=torch.Generator().manual_seed(0)),)
    dom = Ball(x0, radius, 0.0, 1.0)
    path = onnx_file(model, x0, tmp_path)
    cfg = {"example_inputs": x0, "assume_sum_of_products_conv": True}
    bs = TensorProgramCertifier(model, path, dom, cfg).certify()
    iv = TensorProgramCertifier(model, path, dom, dict(cfg, method="interval")).certify()
    assert bs.status == "bounded", bs.summary()
    assert bs.certified_bound < iv.certified_bound / 5
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    ort_run = lambda x: torch.as_tensor(sess.run(None, {"input_0": x.numpy()})[0])  # noqa: E731
    assert observed(model, ort_run, dom, trials=100 if radius else 1) <= bs.certified_bound
    t = bs.output_triples[0]
    with torch.no_grad():
        y = model(*x0)
    assert t.min_val <= float(y.min()) and float(y.max()) <= t.max_val


def test_backsub_float64_analysis_matches_float32():
    model = mlp()
    x0 = ex((8,), n=1)
    dom = Ball(x0, 0.01)
    cfg = {"example_inputs": x0, "assume_deterministic_kernels": False}
    b32 = TensorProgramCertifier(model, model, dom, cfg).certify()
    b64 = TensorProgramCertifier(model, model, dom, dict(cfg, analysis_dtype="float64")).certify()
    assert b32.certified_bound > 0
    assert b32.certified_bound == pytest.approx(b64.certified_bound, rel=1e-2)


def test_backsub_known_dsl_orders_cancel():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 8)).eval()
    x0 = ex((64,), n=1)
    dom = Ball(x0, 0.01)
    tri = graph_with_impl(model, x0, {"backend": "triton", "split_k": 1, "block_k": 32})
    tl = graph_with_impl(model, x0, {"backend": "tilelang"})
    eager = graph_with_impl(model, x0, None)
    both = TensorProgramCertifier(tri, tl, dom, {"example_inputs": x0}).certify()
    assert both.status == "bounded" and both.certified_bound == 0.0, both.summary()
    one = TensorProgramCertifier(tri, eager, dom, {"example_inputs": x0}).certify()
    none_ = TensorProgramCertifier(eager, eager, dom, {
        "example_inputs": x0, "assume_deterministic_kernels": False}).certify()
    assert 0 < one.certified_bound < none_.certified_bound


def test_backsub_identical_eager_kernels_are_bitwise_equal():
    model = DeepCNN().eval()
    x0 = (torch.rand(1, 3, 10, 10, generator=torch.Generator().manual_seed(0)),)
    dom = Ball(x0, 0.01, 0.0, 1.0)
    cfg = {"example_inputs": x0, "assume_sum_of_products_conv": True}
    same = TensorProgramCertifier(model, model, dom, cfg).certify()
    assert same.status == "bounded" and same.certified_bound == 0.0, same.summary()
    assert any("deterministic" in a for a in same.assumptions)
    off = TensorProgramCertifier(model, model, dom,
                                 dict(cfg, assume_deterministic_kernels=False)).certify()
    assert off.certified_bound > 0
    assert not any("deterministic" in a for a in off.assumptions)


# --------------------------------------------------------------------------------------------
# Cancellation-aware node magnitudes (signed_accumulation, the default)
# --------------------------------------------------------------------------------------------

def signed_core(lo, hi, mode, zero, n):
    from dlclv.backsub import BacksubAnalyzer
    an = BacksubAnalyzer.__new__(BacksubAnalyzer)
    an.prob = False
    g = F32.gamma(n + 1) + (n + 2) * 2.0 ** -50
    t = lambda v: torch.as_tensor(np.asarray(v, dtype=np.float64)).reshape(1, n, 1)  # noqa: E731
    return float(an._signed_core(t(lo), t(hi), (mode, zero, g, 0.0, ()))[mode]) * U


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_signed_node_magnitudes_cover_every_tree(n):
    rng = np.random.default_rng(100 + n)
    seq = fpm.SumTree.sequential(n, zero_init=False)
    trees = [fpm.SumTree.from_nested(t, n) for t in all_trees(range(n))]
    for _ in range(20):
        x = rand_inputs(rng, n)
        X = np.abs(x.astype(np.float64))
        exact = exact_sum(x)
        anyb = signed_core(x, x, "any", True, n)
        seqb = signed_core(x, x, "seq", False, n)
        # never looser than the bounds on leaf magnitudes
        assert anyb <= float(fpm.anytree_bound(X, F32, False)[0]) * (1 + 1e-9)
        assert seqb <= float(fpm.sequential_bound(X, F32, False, zero_init=False)[0]) * (1 + 1e-9)
        assert abs(float(Fraction(emulate(seq, x)) - exact)) <= seqb
        for t in trees:
            assert abs(float(Fraction(emulate(t, x)) - exact)) <= anyb


def test_signed_node_magnitudes_cover_leaf_intervals():
    rng = np.random.default_rng(7)
    n = 5
    trees = [fpm.SumTree.from_nested(t, n) for t in all_trees(range(n))]
    seq = fpm.SumTree.sequential(n, zero_init=False)
    for _ in range(10):
        c = rand_inputs(rng, n)
        r = (np.abs(c) * rng.uniform(0, 0.5, n)).astype(np.float32)
        lo, hi = (c - r).astype(np.float32), (c + r).astype(np.float32)
        anyb = signed_core(lo, hi, "any", True, n)
        seqb = signed_core(lo, hi, "seq", False, n)
        for _ in range(8):
            x = np.clip((lo + rng.uniform(0, 1, n) * (hi - lo)).astype(np.float32), lo, hi)
            exact = exact_sum(x)
            assert abs(float(Fraction(emulate(seq, x)) - exact)) <= seqb
            for t in trees:
                assert abs(float(Fraction(emulate(t, x)) - exact)) <= anyb


def test_signed_accumulation_is_tighter_end_to_end():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 8)).eval()
    x0 = ex((64,), n=1)
    dom = Ball(x0, 0.001)
    tri = graph_with_impl(model, x0, {"backend": "triton", "split_k": 1, "block_k": 32})
    tri4 = graph_with_impl(model, x0, {"backend": "triton", "split_k": 4, "block_k": 32})
    eager = graph_with_impl(model, x0, None)
    for p1, p2 in ((tri, eager), (tri, tri4)):
        for guarantee in ("formal", "probabilistic"):
            cfg = {"example_inputs": x0, "guarantee": guarantee}
            new = TensorProgramCertifier(p1, p2, dom, cfg).certify()
            old = TensorProgramCertifier(p1, p2, dom,
                                         dict(cfg, signed_accumulation=False)).certify()
            assert 0 < new.certified_bound < old.certified_bound, (guarantee, new.summary())


# --------------------------------------------------------------------------------------------
# reference="exact": program 2 against exact arithmetic on program 1's graph
# --------------------------------------------------------------------------------------------

def test_exact_reference_bounds_one_program_against_real_arithmetic(tmp_path):
    model = DeepCNN().eval()
    x0 = (torch.rand(1, 3, 10, 10, generator=torch.Generator().manual_seed(0)),)
    dom = Ball(x0, 0.0, 0.0, 1.0)
    cfg = {"example_inputs": x0, "assume_sum_of_products_conv": True}
    ex_ = TensorProgramCertifier(model, model, dom, dict(cfg, reference="exact")).certify()
    assert ex_.status == "bounded" and ex_.certified_bound > 0, ex_.summary()
    assert any("exact real arithmetic" in a for a in ex_.assumptions)
    import copy
    with torch.no_grad():
        y64 = copy.deepcopy(model).double()(x0[0].double())
        err = float((model(*x0).double() - y64).abs().max())
    assert err <= ex_.certified_bound
    # program 1's rounding is not charged: about half of the two-program bound
    two = TensorProgramCertifier(model, model, dom,
                                 dict(cfg, assume_deterministic_kernels=False)).certify()
    assert ex_.certified_bound < two.certified_bound
    # a known-order program 2 does not inherit the unknown order of program 1
    tri = graph_with_impl(model, x0, {"backend": "triton", "split_k": 1, "block_k": 32})
    known = TensorProgramCertifier(model, tri, dom, dict(cfg, reference="exact")).certify()
    assert 0 < known.certified_bound < ex_.certified_bound
    with pytest.raises(ValueError):
        TensorProgramCertifier(model, model, dom,
                               dict(cfg, reference="exact", method="interval")).certify()
