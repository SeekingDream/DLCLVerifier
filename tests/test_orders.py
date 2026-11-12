"""Rebuilt accumulation orders: fast tree reveal, product emulation, and the per-element bound
of a revealed tree (fprev.py, backsub._signed_core mode "tree")."""
import random
from fractions import Fraction

import numpy as np
import pytest

from dlclv import fpmodel as fpm
from dlclv.fprev import (OrderEvidence, RevealError, emulate_dot, linear_key,
                         reveal_order_batched, tree_depth)

torch = pytest.importorskip("torch")
nn = torch.nn

from dlclv.backsub import BacksubAnalyzer  # noqa: E402
from dlclv.certifier import TensorProgramCertifier  # noqa: E402


def canon(t):
    return t if isinstance(t, (int, str)) else frozenset(canon(c) for c in t)


def random_tree(n, rng):
    items = list(range(n))
    rng.shuffle(items)
    while len(items) > 1:
        i = rng.randrange(len(items) - 1)
        items[i:i + 2] = [(items[i], items[i + 1])]
        if rng.random() < 0.5:
            rng.shuffle(items)
    return fpm.SumTree.from_nested(items[0], n)


def lca_oracle(tree):
    """LCA sizes of a known tree, and a counter of the pairs asked."""
    total = tree.n_total
    parent, size = {}, {i: 1 for i in range(total)}
    for k, ch in enumerate(tree.children):
        size[total + k] = sum(size[c] for c in ch)
        for c in ch:
            parent[c] = total + k

    def anc(i):
        out = [i]
        while out[-1] in parent:
            out.append(parent[out[-1]])
        return out

    def sizes(pairs):
        res = []
        for i, j in pairs:
            ai = set(anc(i))
            res.append(size[next(a for a in anc(j) if a in ai)])
        return res
    return sizes


@pytest.mark.parametrize("n", [2, 3, 7, 32, 61])
def test_batched_reveal_rebuilds_any_binary_tree(n):
    rng = random.Random(n)
    for tree in [random_tree(n, rng), fpm.SumTree.pairwise(n),
                 fpm.SumTree.sequential(n, zero_init=False),
                 fpm.SumTree.sequential(n, order=list(range(n))[::-1], zero_init=False)]:
        got = reveal_order_batched(lca_oracle(tree), n)
        assert canon(got.to_nested()) == canon(tree.to_nested())
        assert got.queries <= n * (n - 1) // 2
    chain = reveal_order_batched(lca_oracle(fpm.SumTree.sequential(n, zero_init=False)), n)
    assert chain.queries == n - 1                 # a chain needs one query per leaf


def test_batched_reveal_rejects_inconsistent_sizes():
    with pytest.raises(RevealError):
        reveal_order_batched(lambda pairs: [3] * len(pairs), 6)


def test_emulate_dot_distinguishes_fused_and_rounded_products():
    t = fpm.SumTree.from_nested((0, 1), 2)
    a = np.float32(1 + 2.0 ** -12)
    x, w = [a, a], [a, np.float32(-(1 - 2.0 ** -12))]
    assert emulate_dot(t, x, w, [True, False]) == 2.0 ** -11 + 2.0 ** -23
    assert emulate_dot(t, x, w, [False, False]) == 2.0 ** -11 + 2.0 ** -24
    assert emulate_dot(t, x, w, [False, False], bias=1.0) == 1 + 2.0 ** -11
    with pytest.raises(RevealError):
        emulate_dot(t, x, w, [True, True])


def _analyzer():
    an = object.__new__(BacksubAnalyzer)
    an.prob = False
    return an


def test_tree_mode_equals_chain_mode_on_a_chain():
    K = 37
    g = torch.Generator().manual_seed(0)
    lo = torch.randn(3, K, 2, dtype=torch.float64, generator=g)
    hi = lo + torch.rand(3, K, 2, dtype=torch.float64, generator=g) * 0.1
    chain = fpm.SumTree.sequential(K, zero_init=False)
    generic = fpm.SumTree(K, chain.children)                 # the same tree without seq_order
    an = _analyzer()
    a = an._signed_core(lo, hi, ("seq", False, 1e-6, 0.0, ()))["seq"]
    b = an._signed_core(lo, hi, (("tree", generic), True, 1e-6, 0.0, ()))["tree"]
    assert torch.equal(a, b)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_tree_bound_covers_the_emulated_kernel(seed):
    """|kernel - exact| <= u * (sum over nodes of pow2floor(node magnitude) + leaf roundings)
    for a random tree with some fused products."""
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    K = 40
    tree = random_tree(K, rng)
    fmt = fpm.FORMATS["float32"]
    an = _analyzer()
    g_ = fmt.gamma(K + 1) + (K + 2) * 2.0 ** -50
    for trial in range(20):
        x = nrng.standard_normal(K).astype(np.float32)
        w = (nrng.standard_normal(K) * 2.0 ** nrng.integers(-6, 6, K)).astype(np.float32)
        fused = [False] * K
        for ch in tree.children:                               # at most one fused leaf per node
            real = [c for c in ch if c < K]
            if real and rng.random() < 0.6:
                fused[real[0]] = True
        got = Fraction(emulate_dot(tree, x, w, fused))
        exact = sum(Fraction(float(a)) * Fraction(float(b)) for a, b in zip(x, w))
        p = torch.as_tensor(x.astype(np.float64) * w.astype(np.float64)).reshape(1, K, 1)
        st = an._signed_core(p, p, (("tree", tree), True, g_, 0.0, ()))
        bound = fmt.u * (float(st["tree"]) + float(st["leaf"])) * fpm.SAFETY
        assert abs(got - exact) <= bound
        if trial == 0:                                         # and it is not vacuous
            worst = fmt.u * float(p.abs().sum()) * (K - 1)
            assert bound < worst


class Box:
    def __init__(self, *shapes):
        self.shapes, self.lower, self.upper = shapes, [-1.0] * len(shapes), [1.0] * len(shapes)

    def sample(self, n, generator=None):
        return tuple(torch.rand((n, *s), generator=generator) * 2 - 1 for s in self.shapes)


def test_probed_eager_order_tightens_a_sound_bound():
    """CPU BLAS: when the probe establishes the order of nn.Linear, the bound of the eager
    model against itself (without the determinism rule) shrinks and still covers the error
    against float64."""
    torch.manual_seed(0)
    model = nn.Linear(96, 5).eval()
    x = (torch.rand(1, 96) * 2 - 1,)
    cfg = {"example_inputs": x, "assume_deterministic_kernels": False, "reference": "exact"}
    cert = TensorProgramCertifier(model, model, Box((96,)), cfg)
    loose = cert.certify()
    report = cert.probe_eager_orders()
    if report["failed"] or not report["established"]:
        pytest.skip(f"order of the CPU BLAS kernel not established: {report['failed']}")
    key = linear_key("eager", (1, 96), (96, 5), "aten.linear", True)
    assert key in cert.config.order_evidence and tree_depth(cert.config.order_evidence[key].tree)
    tight = cert.certify()
    assert tight.status == loose.status == "bounded"
    assert 0 < tight.certified_bound < loose.certified_bound
    assert any("order[" in a for a in tight.assumptions)
    m64 = nn.Linear(96, 5).double()
    m64.load_state_dict(model.state_dict())
    g = torch.Generator().manual_seed(1)
    worst = 0.0
    with torch.no_grad():
        for _ in range(200):
            xi = torch.rand(1, 96, generator=g) * 2 - 1
            worst = max(worst, float((model(xi).double() - m64(xi.double())).abs().max()))
    assert worst <= tight.certified_bound
    # strict policy: validated evidence is not used unless it is accepted
    strict = TensorProgramCertifier(model, model, Box((96,)), dict(
        cfg, order_evidence=cert.config.order_evidence)).certify()
    assert strict.certified_bound == loose.certified_bound


def test_order_evidence_is_matched_by_the_exact_kernel_call():
    """Plumbing only (the tree here is not the CPU kernel's): evidence is used for the call it
    names, only when validated evidence is accepted, and it is listed in the assumptions."""
    torch.manual_seed(0)
    model = nn.Linear(96, 5).eval()
    x = (torch.rand(1, 96) * 2 - 1,)
    cfg = {"example_inputs": x, "assume_deterministic_kernels": False, "reference": "exact"}
    loose = TensorProgramCertifier(model, model, Box((96,)), cfg).certify()
    ev = OrderEvidence(fpm.SumTree.pairwise(96), "either", "validated", 96, kernel="test tree")

    def bound(key, accept):
        return TensorProgramCertifier(model, model, Box((96,)), dict(
            cfg, order_evidence={key: ev}, accept_validated_orders=accept)).certify()

    key = linear_key("eager", (1, 96), (96, 5), "aten.linear", True)
    tight = bound(key, True)
    assert 0 < tight.certified_bound < 0.5 * loose.certified_bound
    assert any("order[test tree]" in a for a in tight.assumptions)
    assert bound(key, False).certified_bound == loose.certified_bound
    for other in [linear_key("eager", (2, 96), (96, 5)),               # another batch size
                  linear_key("eager", (1, 96), (96, 5), bias=False),   # another call
                  linear_key("triton", (1, 96), (96, 5))]:
        assert bound(other, True).certified_bound == loose.certified_bound


def test_reference_modes():
    """reference="eager" needs the eager model as program 1 and probes its kernels;
    reference="exact" compares with F; unknown orders are listed, or refused on request."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv2d(1, 2, 3), nn.Flatten(), nn.Linear(2 * 6 * 6, 4)).eval()
    x = (torch.rand(1, 1, 8, 8) * 2 - 1,)
    cfg = {"example_inputs": x, "assume_sum_of_products_conv": True,
           "assume_deterministic_kernels": False}
    box = Box((1, 8, 8))
    eager = TensorProgramCertifier(model, model, box, dict(cfg, reference="eager")).certify()
    assert eager.status == "bounded" and "reference: eager" in eager.notes
    unknown = [n for n in eager.notes if n.startswith("order unknown")]
    assert unknown and "aten.conv2d" in unknown[0]          # no probe for convolutions
    # against F only program 2's rounding counts; with the same probed orders it is smaller
    exact = TensorProgramCertifier(model, model, box, dict(
        cfg, reference="exact", probe_orders=True)).certify()
    assert exact.status == "bounded" and 0 < exact.certified_bound < eager.certified_bound
    assert "reference: exact" in exact.notes
    strict = TensorProgramCertifier(model, model, box, dict(
        cfg, reference="eager", require_known_orders=True)).certify()
    assert strict.status == "inconclusive" and strict.certified_bound is None
    assert "aten.conv2d" in strict.detail
    with pytest.raises(ValueError, match="unknown reference"):
        TensorProgramCertifier(model, model, box, dict(cfg, reference="float64")).certify()
    g = TensorProgramCertifier(model, model, box, cfg).extract_graphs()[0]
    g.meta["impl"] = {"backend": "triton"}
    with pytest.raises(ValueError, match="eager model as program 1"):
        TensorProgramCertifier(g, model, box, dict(cfg, reference="eager")).certify()
