"""Interface of the abstract interpretation stage. The transfer functions are in `absint.py`.

The interpreter works on the units found by `matching.py`, in dependency order. Every boundary
tensor of a unit carries one `ErrorTriple`:

    min_val, max_val   enclose every value of that tensor in program 1 and in program 2,
                       for every input in the domain;
    max_error          bounds |v1 - v2| elementwise, where v1 and v2 are the values of that
                       tensor in the two programs for the same input (exact real subtraction).

A unit's transfer function maps the triples of its boundary inputs to triples of its boundary
outputs. It must account for both the propagated input error and the rounding error the two
programs add inside the unit (different addition orders, fused epilogues, ...).

`propagate` seeds graph inputs and constants (error 0, since both programs read the same inputs
and, after matching, constants with identical contents), calls the interpreter on each unit in
order, and checks every returned triple. `DeferredInterpreter` raises `Unimplemented`; it is
kept for callers that want to stop after matching.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from .stubs import Unimplemented


@dataclass(frozen=True)
class ErrorTriple:
    min_val: float
    max_val: float
    max_error: float

    def __post_init__(self):
        for x in (self.min_val, self.max_val, self.max_error):
            if isinstance(x, float) and math.isnan(x):
                raise ValueError("ErrorTriple cannot hold NaN")
        if self.min_val > self.max_val:
            raise ValueError(f"min_val {self.min_val} > max_val {self.max_val}")
        if not self.max_error >= 0:
            raise ValueError(f"max_error {self.max_error} must be >= 0")


@dataclass
class UnitContext:
    """What a transfer function may inspect for one unit."""
    unit: object                  # matching.MatchedUnit
    match: object                 # matching.MatchResult (canonical graphs, constants)
    config: object                # certifier.CertifierConfig


class UnitInterpreter(ABC):
    """Abstract transfer function for matched units."""

    @abstractmethod
    def transfer(self, ctx: UnitContext, inputs: Sequence[ErrorTriple]) -> List[ErrorTriple]:
        """Triples for `ctx.unit.outputs`, given triples for `ctx.unit.inputs` (same order)."""


class DeferredInterpreter(UnitInterpreter):
    def transfer(self, ctx, inputs):
        raise Unimplemented(
            f"abstract interpretation of matched units is not implemented yet (reached unit "
            f"{ctx.unit.id}: {'+'.join(ctx.unit.ops_1)} <-> {'+'.join(ctx.unit.ops_2)})")


def initial_triples(match, input_domain) -> Dict[str, ErrorTriple]:
    """Triples for graph inputs (from the domain box) and constants (from their data).

    Keys are program-1 canonical value ids.
    """
    env = {}
    for b in match.input_pairs:
        idx = match.canonical_1.values[b.value_1].input_index
        lo = float(np.min(np.asarray(input_domain.lower[idx], dtype=np.float64)))
        hi = float(np.max(np.asarray(input_domain.upper[idx], dtype=np.float64)))
        env[b.value_1] = ErrorTriple(lo, hi, 0.0)
    for u in match.units:
        for b in u.inputs:
            v = match.canonical_1.values[b.value_1]
            if v.kind == "const" and b.value_1 not in env:
                d = np.asarray(v.data, dtype=np.float64)
                env[b.value_1] = ErrorTriple(float(d.min()), float(d.max()), 0.0)
    return env


def propagate(match, interpreter: UnitInterpreter, env: Dict[str, ErrorTriple], config=None):
    """Run the interpreter over all units; return triples for the graph outputs."""
    env = dict(env)
    for u in match.units:
        missing = [b.value_1 for b in u.inputs if b.value_1 not in env]
        if missing:
            raise ValueError(f"unit {u.id} reads values with no triple: {missing}")
        outs = interpreter.transfer(UnitContext(u, match, config),
                                    [env[b.value_1] for b in u.inputs])
        if len(outs) != len(u.outputs):
            raise ValueError(f"unit {u.id}: interpreter returned {len(outs)} triples for "
                             f"{len(u.outputs)} outputs")
        for b, t in zip(u.outputs, outs):
            if not isinstance(t, ErrorTriple):
                raise TypeError(f"unit {u.id}: expected ErrorTriple, got {type(t).__name__}")
            env[b.value_1] = t
    result = []
    for b, _ in match.output_pairs:
        if b.value_1 not in env:
            raise ValueError(f"graph output {b.value_1} has no triple")
        result.append(env[b.value_1])
    return result
