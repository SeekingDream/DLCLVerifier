"""dlclv.primitives against an exact rational oracle, on inputs chosen to separate
accumulation orders, fused from separate products, and one rounding from two."""
import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional

import fp_oracle as fo  # noqa: E402
from dlclv import fpmodel as fpm  # noqa: E402
from dlclv import primitives as P  # noqa: E402
from dlclv.bitwise import bitwise_equal  # noqa: E402
from dlclv.fprev import emulate_dot  # noqa: E402

f32 = torch.float32


def t32(*v):
    return torch.tensor(v, dtype=f32)


def fma_cases(n=4000, seed=0):
    """(a, b, c) float32 triples: random, wide exponents, cancellation of the product,
    subnormal results, rounding ties, signed zeros, overflow."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(n).astype(np.float32)
    b = rng.standard_normal(n).astype(np.float32)
    c = rng.standard_normal(n).astype(np.float32)
    k = n // 8
    a[k:2 * k] *= (2.0 ** rng.integers(-60, 60, k)).astype(np.float32)
    b[k:2 * k] *= (2.0 ** rng.integers(-60, 60, k)).astype(np.float32)
    c[k:2 * k] *= (2.0 ** rng.integers(-100, 100, k)).astype(np.float32)
    c[2 * k:3 * k] = -(a[2 * k:3 * k] * b[2 * k:3 * k])            # c = -fl(a*b): tiny result
    a[3 * k:4 * k] *= np.float32(2.0 ** -70)                         # subnormal / underflow
    b[3 * k:4 * k] *= np.float32(2.0 ** -70)
    c[3 * k:4 * k] *= (2.0 ** rng.integers(-149, -120, k)).astype(np.float32)
    # c far above the product: fl64(a*b + c) is inexact and often lands on a float32 tie
    c[4 * k:5 * k] = (2.0 ** rng.integers(24, 40, k)).astype(np.float32)
    a[4 * k:5 * k] = (rng.integers(1, 2 ** 12, k) * 2.0 ** rng.integers(-3, 4, k)).astype(np.float32)
    b[4 * k:5 * k] = np.float32(1) + (2.0 ** rng.integers(-23, -12, k)).astype(np.float32) * \
        rng.choice([-1, 1], k).astype(np.float32)
    z = slice(5 * k, 5 * k + 8)
    a[z] = [0.0, -0.0, 0.0, -0.0, 1.0, -1.0, 2 ** -80, -2 ** -80]
    b[z] = [1.0, 1.0, -1.0, 2.0, 1.0, 1.0, 2 ** -80, 2 ** -80]
    c[z] = [-0.0, -0.0, 0.0, 0.0, -1.0, 1.0, -0.0, -0.0]
    o = slice(5 * k + 8, 5 * k + 12)
    a[o] = [2 ** 100, 2 ** 100, -2 ** 100, 2 ** 64]
    b[o] = [2 ** 100, 2 ** 27, 2 ** 28, 2 ** 64]
    c[o] = [1.0, 2 ** 127, 2 ** 127, -2 ** 127]
    return a, b, c


def test_fma32_is_the_correctly_rounded_fused_operation():
    a, b, c = fma_cases()
    got = P.fma32(torch.from_numpy(a), torch.from_numpy(b), torch.from_numpy(c)).numpy()
    want = np.array([fo.fma(x, y, z) for x, y, z in zip(a, b, c)], dtype=np.float32)
    assert got.tobytes() == want.tobytes()
    assert np.isinf(want).any() and (want == 0).any() and np.signbit(want[want == 0]).any()
    assert (np.abs(want[want != 0]) < np.finfo(np.float32).tiny).any()      # subnormal results


def test_fma32_rounds_once_where_float64_rounds_twice():
    """a*b + c = 2^30 + 192 - 3*2^-26 lies just below the midpoint of two float32 values.
    float64 rounds it onto the midpoint; the second rounding then goes to the even neighbour,
    which is the wrong one."""
    a, b, c = t32(24 + 3 * 2.0 ** -13), t32(8 - 2.0 ** -13), t32(2.0 ** 30)
    twice = (a.double() * b.double() + c.double()).float()
    once = P.fma32(a, b, c)
    assert once.item() == 2.0 ** 30 + 128 and twice.item() == 2.0 ** 30 + 256
    assert fo.bits(fo.fma(a.item(), b.item(), c.item())) == fo.bits(once.numpy())
    # and a fused product is not the separately rounded one
    x, y, z = t32(1 + 2.0 ** -23), t32(1 - 2.0 ** -23), t32(-1.0)
    assert P.fma32(x, y, z).item() == -(2.0 ** -46) and (x * y + z).item() == 0.0


def tie_cases(n=20000, seed=1):
    """c = 2^e and a * b within a few float64 ulps of an odd multiple of half a float32 ulp of
    c: a * b + c sits next to a float32 rounding boundary, and float64 cannot hold the sum."""
    rng = np.random.default_rng(seed)
    e = rng.integers(26, 40, n)
    c = (2.0 ** e).astype(np.float32)
    target = (rng.integers(0, 64, n) * 2 + 1) * 2.0 ** (e - 24)
    a = (np.sqrt(target) * (1 + 2.0 ** rng.integers(-12, -8, n))).astype(np.float32)
    b = (target / a.astype(np.float64)).astype(np.float32)
    return a, b, c


def test_fma32_many_double_rounding_cases():
    a, b, c = tie_cases()
    ta, tb, tc = (torch.from_numpy(v) for v in (a, b, c))
    once = P.fma32(ta, tb, tc)
    twice = (ta.double() * tb.double() + tc.double()).float()
    differ = (once != twice).nonzero().flatten().tolist()
    assert len(differ) > 50, "the case generator no longer reaches float32 ties"
    want = np.array([fo.fma(x, y, z) for x, y, z in zip(a, b, c)], dtype=np.float32)
    assert once.numpy().tobytes() == want.tobytes()


def test_fma32_non_finite_and_broadcast():
    inf = float("inf")
    assert P.fma32(t32(inf), t32(2.0), t32(1.0)).item() == inf
    assert P.fma32(t32(1.0), t32(2.0), t32(-inf)).item() == -inf
    assert torch.isnan(P.fma32(t32(inf), t32(1.0), t32(-inf))).all()
    assert torch.isnan(P.fma32(t32(float("nan")), t32(1.0), t32(1.0))).all()
    out = P.fma32(torch.ones(3, 1), torch.ones(1, 4), torch.zeros(3, 4))
    assert out.shape == (3, 4) and out.dtype == f32


def test_two_sum_is_error_free():
    rng = np.random.default_rng(0)
    a = torch.from_numpy(rng.standard_normal(2000) * 2.0 ** rng.integers(-300, 300, 2000))
    b = torch.from_numpy(rng.standard_normal(2000) * 2.0 ** rng.integers(-300, 300, 2000))
    s, e = P.two_sum(a, b)
    from fractions import Fraction as Fr
    for i in range(0, 2000, 7):
        assert Fr(a[i].item()) + Fr(b[i].item()) == Fr(s[i].item()) + Fr(e[i].item())


def oracle_chain(a, b, ks, fused):
    """acc = +0; for k in ks: acc = fma(a_k, b_k, acc)  or  acc + fl(a_k * b_k)."""
    acc = np.float32(0.0)
    for k in ks:
        acc = fo.fma(a[k], b[k], acc) if fused else fo.add(acc, fo.mul(a[k], b[k]))
    return acc


@pytest.mark.parametrize("fp", ["fma", "separate"])
def test_chain_dot_matches_the_oracle_and_depends_on_order_and_fusion(fp):
    rng = np.random.default_rng(3)
    M, K, N = 3, 13, 4
    a = (rng.standard_normal((M, K)) * 2.0 ** rng.integers(-8, 8, (M, K))).astype(np.float32)
    b = (rng.standard_normal((K, N)) * 2.0 ** rng.integers(-8, 8, (K, N))).astype(np.float32)
    a[0, :3], b[:3, 0] = [2 ** 12, 1.0, -2 ** 12], [2 ** 13, 2 ** -3, 2 ** 13]   # cancellation
    ta, tb = torch.from_numpy(a), torch.from_numpy(b)
    for ks in (list(range(K)), list(range(K))[::-1], [5, 0, 12, 3, 7]):
        got = P.chain_dot(ta, tb, ks, fp).numpy()
        want = np.array([[oracle_chain(a[m], b[:, n], ks, fp == "fma") for n in range(N)]
                         for m in range(M)], dtype=np.float32)
        assert got.tobytes() == want.tobytes()
    fwd, rev = P.chain_dot(ta, tb, None, fp), P.chain_dot(ta, tb, list(range(K))[::-1], fp)
    assert not bitwise_equal(fwd, rev)                       # the order is observable
    assert not bitwise_equal(P.chain_dot(ta, tb, None, "fma"), P.chain_dot(ta, tb, None, "separate"))
    with pytest.raises(TypeError):
        P.chain_dot(ta.double(), tb.double())


def test_chain_dot_signed_zero_and_underflow():
    """A product that underflows to -0 keeps acc = -0; a masked fma(+0, +0, acc) turns it +0;
    an exact -0 product on the +0 accumulator gives +0."""
    a, b = t32(2.0 ** -80).reshape(1, 1), t32(-(2.0 ** -80)).reshape(1, 1)
    acc = P.chain_dot(a, b)
    assert acc.item() == 0 and torch.signbit(acc).item()
    zero = torch.zeros_like(acc)
    assert not torch.signbit(P.fma32(zero, zero, acc)).item()
    assert not torch.signbit(P.chain_dot(t32(0.0).reshape(1, 1), t32(-1.0).reshape(1, 1))).item()
    assert fo.bits(fo.fma(2.0 ** -80, -(2.0 ** -80), 0.0)) == fo.bits(acc.numpy())


def random_tree(n, rng):
    items = list(range(n))
    rng.shuffle(items)
    while len(items) > 1:
        i = rng.randrange(len(items) - 1)
        items[i:i + 2] = [(items[i], items[i + 1])]
        if rng.random() < 0.5:
            rng.shuffle(items)
    return fpm.SumTree.from_nested(items[0], n)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_tree_dot_matches_the_rational_emulator(seed):
    rng, nrng = random.Random(seed), np.random.default_rng(seed)
    K, M, N = 23, 2, 3
    tree = random_tree(K, rng) if seed else fpm.SumTree.pairwise(K)
    fused = [False] * K
    for ch in tree.children:
        real = [c for c in ch if c < K]
        if real and rng.random() < 0.6:
            fused[real[0]] = True
    a = (nrng.standard_normal((M, K)) * 2.0 ** nrng.integers(-6, 6, (M, K))).astype(np.float32)
    b = (nrng.standard_normal((K, N)) * 2.0 ** nrng.integers(-6, 6, (K, N))).astype(np.float32)
    bias = nrng.standard_normal(N).astype(np.float32)
    got = P.tree_dot(tree, torch.from_numpy(a), torch.from_numpy(b), fused,
                     bias=torch.from_numpy(bias)).numpy()
    want = np.array([[emulate_dot(tree, a[m], b[:, n], fused, bias[n]) for n in range(N)]
                     for m in range(M)], dtype=np.float32)
    assert got.tobytes() == want.tobytes()
    if seed == 0:                                  # fusing products changes the bits
        plain = P.tree_dot(tree, torch.from_numpy(a), torch.from_numpy(b)).numpy()
        assert plain.tobytes() != P.tree_dot(tree, torch.from_numpy(a), torch.from_numpy(b),
                                             fused).numpy().tobytes()
    with pytest.raises(ValueError, match="two fused"):
        P.tree_dot(fpm.SumTree.from_nested((0, 1), 2), torch.ones(1, 2), torch.ones(2, 1),
                   [True, True])


def test_tree_dot_zero_models():
    tree = fpm.SumTree.from_nested((0, 1), 2)
    a, b = torch.ones(1, 2), t32(-0.0, -0.0).reshape(2, 1)
    assert torch.signbit(P.tree_dot(tree, a, b)).item()                       # -0 + -0
    assert not torch.signbit(P.tree_dot(tree, a, b, zero_init=[True, False])).item()
    assert not torch.signbit(P.tree_dot(tree, a, b, zero_root=True)).item()
    assert torch.signbit(P.tree_dot(tree, a, b, [True, False])).item()        # fma(-0, -0)
    one = fpm.SumTree(1, [])
    assert P.tree_dot(one, t32(3.0).reshape(1, 1), t32(0.1).reshape(1, 1)).item() == \
        fo.mul(3.0, np.float32(0.1))


def test_tree_sum_orders():
    x = t32(2.0 ** 25, 1.0, -(2.0 ** 25), 1.0)
    seq = fpm.SumTree.sequential(4, zero_init=False)
    pair = fpm.SumTree.pairwise(4)
    assert P.tree_sum(seq, x, 0).item() == 1.0           # ((2^25 + 1) - 2^25) + 1
    assert P.tree_sum(pair, x, 0).item() == 0.0          # (2^25 + 1) + (-2^25 + 1)
    assert P.tree_sum(fpm.SumTree.from_nested(((0, 2), (1, 3)), 4), x, 0).item() == 2.0
    neg = torch.full((3, 4), -0.0)
    assert torch.signbit(P.tree_sum(pair, neg, 1)).all()
    assert not torch.signbit(P.tree_sum(pair, neg, 1, zero_init=True)).any()
    rng = np.random.default_rng(0)
    y = torch.from_numpy((rng.standard_normal((5, 9, 2)) * 2.0 ** rng.integers(-20, 20, (5, 9, 2)))
                         .astype(np.float32))
    tree = random_tree(9, random.Random(0))
    got = P.tree_sum(tree, y, 1, keepdim=True)
    assert got.shape == (5, 1, 2)
    from dlclv.fprev import emulate
    want = np.array([[emulate(tree, y[i, :, j].numpy()) for j in range(2)] for i in range(5)],
                    dtype=np.float32)
    assert got.squeeze(1).numpy().tobytes() == want.tobytes()


@pytest.mark.parametrize("cfg", [((3, 3), (1, 1), (1, 1), (1, 1)), ((3, 2), (1, 2), (1, 2), (2, 1)),
                                 ((1, 1), (1, 1), (0, 0), (2, 2)), ((2, 4), (2, 1), (0, 3), (1, 3))])
def test_im2col_is_unfold(cfg):
    kernel, dilation, padding, stride = cfg
    x = torch.randn(2, 3, 9, 8)
    x[0, 0, 0, 0] = -0.0
    want = F.unfold(x, kernel, dilation=dilation, padding=padding, stride=stride)
    assert bitwise_equal(P.im2col(x, kernel, dilation, padding, stride), want)


@pytest.mark.parametrize("op", [lambda a, b: a @ b, lambda a, b: a.sum(), lambda a, b: torch.exp(a),
                                lambda a, b: torch.addcmul(a, a, a), lambda a, b: torch.add(a, a, alpha=2),
                                lambda a, b: torch.dot(a[0], a[0]), lambda a, b: a.to("meta")])
def test_guard_rejects_everything_that_is_not_an_explicit_primitive(op):
    a, b = torch.ones(2, 2), torch.ones(2, 2)
    with pytest.raises(P.NonPrimitiveOperation):
        with P.PrimitiveGuard():
            op(a, b)


def test_guard_counts_the_operations_of_a_lowered_kernel():
    a, b = torch.randn(2, 5), torch.randn(5, 3)
    with P.PrimitiveGuard() as g:
        P.chain_dot(a, b)
        P.chain_dot(a, b, fp="separate")
    assert g.counts["aten.nextafter.default"] == 5 and g.counts["aten.mul.Tensor"] == 10
    assert not any("mm" in k or "sum" in k for k in g.counts)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_primitives_give_the_same_bits_on_cuda_and_cpu():
    a, b, c = (torch.from_numpy(v) for v in fma_cases(20000, seed=2))
    assert bitwise_equal(P.fma32(a.cuda(), b.cuda(), c.cuda()).cpu(), P.fma32(a, b, c))
    x, y = torch.randn(4, 3, 40), torch.randn(4, 40, 5)
    for fp in ("fma", "separate"):
        assert bitwise_equal(P.chain_dot(x.cuda(), y.cuda(), fp=fp).cpu(), P.chain_dot(x, y, fp=fp))
    tree = fpm.SumTree.pairwise(40)
    fused = [i % 2 == 0 for i in range(40)]
    assert bitwise_equal(P.tree_dot(tree, x.cuda(), y.cuda(), fused).cpu(),
                         P.tree_dot(tree, x, y, fused))
