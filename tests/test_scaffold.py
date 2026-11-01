from fractions import Fraction as F
import pytest
from dlclv import config
from dlclv.outcomes import Inconclusive, Result, Status, Witness, check_result, tightness_interval
from dlclv.stubs import Unimplemented, run_experiment

BASE = dict(benchmark="b", path="p", domain="d", epsilon=F(1, 1000), tool="t", guarantee="formal")


def ok(**k):
    return Result(**{**BASE, "status": Status.VERIFIED, "certified_bound": F(1, 2000),
                     "certificate_digest": "c", "assumptions_digest": "a", **k})


def wit(dist=F(1, 100), replay=True):
    return Witness("i", replay, dist, "art")


def test_configs_valid():
    assert config.validate() == []


def test_valid_verified():
    check_result(ok())


@pytest.mark.parametrize("g", ["empirical", "probabilistic", "formal_real_semantics", "formal_first_order", "formal_equality", "formal_bounded"])
def test_only_formal_verifies_epsilon(g):
    with pytest.raises(ValueError):
        check_result(ok(guarantee=g))


def test_equality_tool_can_verify_zero_only():
    check_result(ok(guarantee="formal_equality", epsilon=F(0), certified_bound=F(0)))


def test_verified_needs_certificate():
    with pytest.raises(ValueError):
        check_result(ok(certificate_digest=None))


def test_observed_above_bound_is_alarm():
    with pytest.raises(ValueError, match="soundness alarm"):
        check_result(ok(observed_lower_bound=F(1, 1000)))


def test_loose_bound_is_not_refuted():
    r = Result(**BASE, status=Status.REFUTED, certified_bound=F(1))
    with pytest.raises(ValueError):
        check_result(r)
    check_result(Result(**BASE, status=Status.INCONCLUSIVE, certified_bound=F(1),
                        inconclusive_reason=Inconclusive.BOUND_ABOVE_EPSILON))


def test_refuted_requires_replay():
    with pytest.raises(ValueError):
        check_result(Result(**BASE, status=Status.REFUTED, witness=wit(replay=False)))
    check_result(Result(**BASE, status=Status.REFUTED, witness=wit()))


def test_witness_must_exceed_epsilon():
    with pytest.raises(ValueError):
        check_result(Result(**BASE, status=Status.REFUTED, witness=wit(dist=F(1, 1000))))


def test_inconclusive_needs_reason():
    with pytest.raises(ValueError):
        check_result(Result(**BASE, status=Status.INCONCLUSIVE))


def test_tightness():
    assert tightness_interval(F(10), F(2))[2] == 5
    with pytest.raises(ValueError):
        tightness_interval(F(1), F(2))


def test_unimplemented_is_loud():
    with pytest.raises(Unimplemented):
        run_experiment("E1_coverage")


def test_real_semantics_tools_cannot_verify_fp_at_all():
    from dlclv import config as c
    tools = {b["id"]: b["guarantee"] for b in c.load("baselines/registry.yaml")["baselines"]}
    assert tools["alpha_beta_crown"] == "formal_real_semantics"
    with pytest.raises(ValueError):
        check_result(ok(guarantee="formal_real_semantics", epsilon=F(0), certified_bound=F(0)))


def test_every_model_has_known_dataset(monkeypatch):
    real = config.load
    def broken(rel):
        d = real(rel)
        if rel.endswith("benchmarks.yaml"):
            d["suites"]["T3_models_core"]["items"][0]["dataset"] = ["missing"]
        return d
    monkeypatch.setattr(config, "load", broken)
    assert any("unknown dataset" in p for p in config.validate())
