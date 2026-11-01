"""Outcome rules. These are IMPLEMENTED and tested.

The rules make conflation hard: only sound evidence can produce VERIFIED, only a
replayed witness can produce REFUTED, and a loose bound stays INCONCLUSIVE.
"""
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from typing import Optional


class Status(str, Enum):
    VERIFIED = "verified"
    REFUTED = "refuted"
    INCONCLUSIVE = "inconclusive"
    ERROR = "error"


class Inconclusive(str, Enum):
    BOUND_ABOVE_EPSILON = "bound_above_epsilon"
    TIMEOUT = "timeout"
    OUT_OF_MEMORY = "out_of_memory"
    UNSUPPORTED_OPERATOR = "unsupported_operator"
    UNSUPPORTED_PATH = "unsupported_path"
    EXCEPTIONAL_UNRESOLVED = "exceptional_unresolved"
    SOLVER_UNKNOWN = "solver_unknown"


# "formal_equality" counts only with epsilon == 0 semantics; see check_result.
SOUND_FOR_EPSILON = {"formal"}
SOUND_FOR_EQUALITY = {"formal", "formal_equality"}


@dataclass(frozen=True)
class Witness:
    """Input that reproduced a violation on the real artifact."""
    input_digest: str
    replayed_on_artifact: bool
    exact_distance: Fraction
    artifact_digest: str


@dataclass(frozen=True)
class Result:
    benchmark: str
    path: str
    domain: str
    epsilon: Fraction
    tool: str
    guarantee: str
    status: Status
    certified_bound: Optional[Fraction] = None
    certificate_digest: Optional[str] = None
    assumptions_digest: Optional[str] = None
    witness: Optional[Witness] = None
    inconclusive_reason: Optional[Inconclusive] = None
    observed_lower_bound: Optional[Fraction] = None
    wall_seconds: Optional[float] = None
    peak_rss_bytes: Optional[int] = None
    notes: str = field(default="")


def check_result(r: Result) -> None:
    """Raise ValueError when a record claims more than its evidence supports."""
    if r.epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    if r.status is Status.VERIFIED:
        allowed = SOUND_FOR_EQUALITY if r.epsilon == 0 else SOUND_FOR_EPSILON
        if r.guarantee not in allowed:
            raise ValueError(f"guarantee {r.guarantee!r} cannot yield verified at epsilon={r.epsilon}")
        if r.certificate_digest is None or r.assumptions_digest is None:
            raise ValueError("verified needs a certificate and an assumptions digest")
        if r.certified_bound is None or r.certified_bound > r.epsilon:
            raise ValueError("verified needs certified_bound <= epsilon")
        if r.observed_lower_bound is not None and r.observed_lower_bound > r.certified_bound:
            raise ValueError("soundness alarm: observed distance exceeds certified bound")
        if r.witness is not None:
            raise ValueError("verified record cannot carry a witness")
    elif r.status is Status.REFUTED:
        w = r.witness
        if w is None or not w.replayed_on_artifact:
            raise ValueError("refuted needs a witness replayed on the real artifact")
        if w.exact_distance <= r.epsilon:
            raise ValueError("witness distance does not exceed epsilon")
    elif r.status is Status.INCONCLUSIVE:
        if r.inconclusive_reason is None:
            raise ValueError("inconclusive needs a reason")
        if r.witness is not None:
            raise ValueError("a replayed witness makes this refuted, not inconclusive")
        if r.inconclusive_reason is Inconclusive.BOUND_ABOVE_EPSILON:
            if r.certified_bound is None or r.certified_bound <= r.epsilon:
                raise ValueError("bound_above_epsilon needs certified_bound > epsilon")


def tightness_interval(certified: Fraction, observed: Fraction):
    """True worst case lies in [observed, certified]; ratio is an upper bound on looseness."""
    if observed <= 0 or certified < observed:
        raise ValueError("need 0 < observed <= certified")
    return observed, certified, certified / observed
