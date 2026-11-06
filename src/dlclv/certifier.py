"""TensorProgramCertifier: bound the output difference of an original and an optimized program.

    cert = TensorProgramCertifier(model, compiled, input_domain, configure={"epsilon": 1e-4})
    g1, g2 = cert.extract_graphs()     # stage 1, inspectable
    match = cert.match()               # stage 2, inspectable; print(match.summary())
    result = cert.certify()            # stage 3: abstract interpretation of the matched units
    print(result.summary())

`certify()` returns a `CertificationResult`:

* status "verified": every output bound is <= `epsilon`;
* status "bounded": no epsilon was given; `output_triples` holds the bounds;
* status "inconclusive" with an `outcomes.Inconclusive` reason: the match is incomplete or
  ambiguous, the analysis has no sound model for some unit, a value may overflow or become NaN,
  or the bound exceeds epsilon. An inconclusive result never claims a bound below epsilon.

A structural match alone never produces a bound: every number comes from the interpreter
(`absint.FloatErrorInterpreter` by default), and every assumption it relies on is listed in
`result.assumptions`.

`input_domain` is a box: it needs `lower`, `upper` (one entry per program input, broadcastable to
the input shape) and, unless example inputs are configured, `sample(n, generator)`. The
repository's `utils.InputDomain` has all three. Graphs are extracted at the shapes of the example
inputs, so the certificate covers that batch size and sequence length.
"""
from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Sequence

from .frontends import extract_graph
from .interpretation import ErrorTriple, UnitInterpreter, initial_triples, propagate
from .matching import MatchResult, match_graphs
from .outcomes import Inconclusive
from . import fpmodel as fpm


@dataclass
class CertifierConfig:
    batch_size: int = 1                    # batch of the example inputs used for extraction
    seed: int = 20261031
    example_inputs: Optional[Sequence] = None   # overrides sampling from the domain
    fold_constants: bool = True
    normalize_arith: bool = True
    flatten_associative: bool = True
    recognize_idioms: bool = True
    allow_ambiguous: bool = False          # certify() refuses ambiguous matches by default
    interpreter: Optional[UnitInterpreter] = None   # default: absint.FloatErrorInterpreter
    # ---- abstract interpretation ----
    epsilon: Optional[float] = None        # certify against this bound
    device: Optional[str] = None           # where the programs run; inferred from the examples
    elementary_ulp: Optional[Dict[str, float]] = None   # overrides fpmodel.CUDA_FP32_ULP
    trust_dsl_kernel_order: bool = True    # Triton/TileLang GEMM order from kernels.py source
    assume_sum_of_products_conv: bool = False   # library convs: no Winograd/FFT (assumption)
    order_evidence: Optional[Dict[tuple, Any]] = None  # (backend, op) -> fprev.OrderEvidence
    accept_validated_orders: bool = False  # accept FPRev orders with dynamic evidence only
    method: str = "backsub"                # backsub (per element, relational) | interval
    guarantee: str = "formal"              # formal (worst case) | probabilistic (backsub only)
    confidence: float = 1e-6               # probabilistic: failure probability lambda
    analysis_dtype: str = "float32"        # backsub coefficients: float32 (+inflation) | float64
    backsub_tol: float = 1e-2              # back-substitute where forward bounds exceed this
    backsub_memory: float = 1.6e10         # bytes per back-substitution batch (approximate)
    backsub_batch: int = 0                 # rows per batch (0: from backsub_memory)
    backsub_window: int = 0                # intermediate rows: back-substitute through at
                                           # most this many mixing layers (0: all)

    @classmethod
    def from_any(cls, configure) -> "CertifierConfig":
        if configure is None:
            return cls()
        if isinstance(configure, cls):
            return configure
        if isinstance(configure, dict):
            known = {f.name for f in fields(cls)}
            bad = set(configure) - known
            if bad:
                raise ValueError(f"unknown configure keys: {sorted(bad)}")
            return cls(**configure)
        raise TypeError("configure must be a CertifierConfig, a dict or None")


@dataclass
class CertificationResult:
    status: str                                    # verified | bounded | inconclusive
    match: MatchResult
    interpreter: str
    output_triples: Optional[List[ErrorTriple]] = None
    epsilon: Optional[float] = None
    inconclusive_reason: Optional[Inconclusive] = None
    detail: str = ""
    unit: str = ""                                 # unit where the analysis stopped
    assumptions: List[str] = field(default_factory=list)
    unit_reports: List[Any] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    guarantee: str = "formal"                      # formal | probabilistic

    @property
    def output_error_bounds(self) -> Optional[List[float]]:
        if self.output_triples is None:
            return None
        return [t.max_error for t in self.output_triples]

    @property
    def certified_bound(self) -> Optional[float]:
        b = self.output_error_bounds
        return max(b) if b else None

    @property
    def assumptions_digest(self) -> str:
        return hashlib.sha256("\n".join(sorted(self.assumptions)).encode()).hexdigest()[:16]

    def summary(self, max_units: int = 6) -> str:
        lines = [f"certify: {self.status.upper()} [{self.guarantee}]"
                 + (f" ({self.inconclusive_reason.value}: {self.detail})"
                    if self.inconclusive_reason else "")]
        if self.output_triples:
            for i, t in enumerate(self.output_triples):
                lines.append(f"  output {i}: range [{t.min_val:.6g}, {t.max_val:.6g}], "
                             f"max_error <= {t.max_error:.6g}")
        if self.epsilon is not None:
            lines.append(f"  epsilon = {self.epsilon:g}")
        if self.unit_reports and isinstance(self.unit_reports[0], dict):
            worst = sorted(self.unit_reports, key=lambda r: -max(
                (o["D"] for o in r["outputs"]), default=0))[:max_units]
            for r in worst:
                if r["outputs"]:
                    o = max(r["outputs"], key=lambda o: o["D"])
                    lines.append(f"  {r['unit']} {'+'.join(r['ops_1'])[:40]}: D {o['D']:.3g} "
                                 f"(E1 {o['E1']:.2g}, E2 {o['E2']:.2g}, V {o['V']:.2g}, "
                                 f"|x| {o['center_mag']:.3g})")
        else:
            worst = sorted(self.unit_reports, key=lambda r: -max(
                (o["bound"] for o in r.outputs), default=0))[:max_units]
            for r in worst:
                o = max(r.outputs, key=lambda o: o["bound"]) if r.outputs else None
                if o:
                    lines.append(f"  {r.unit} {'+'.join(r.ops_1)}: bound {o['bound']:.3g} via "
                                 f"{o['via']} (P {o['P']:.2g}, R1 {o['R1']:.2g}, "
                                 f"R2 {o['R2']:.2g})")
        for nt in self.notes:
            lines.append(f"  {nt}")
        lines.append(f"  assumptions ({len(self.assumptions)}, digest {self.assumptions_digest})")
        return "\n".join(lines)


class TensorProgramCertifier:
    def __init__(self, tensor_program_1: Any, tensor_program_2: Any, input_domain: Any,
                 configure: Any = None):
        self.tensor_program_1 = tensor_program_1
        self.tensor_program_2 = tensor_program_2
        self.input_domain = input_domain
        self.config = CertifierConfig.from_any(configure)
        self._examples = None
        self._graphs = None
        self._match = None

    def example_inputs(self):
        if self._examples is None:
            if self.config.example_inputs is not None:
                self._examples = tuple(self.config.example_inputs)
            else:
                import torch
                gen = torch.Generator().manual_seed(self.config.seed)
                self._examples = tuple(self.input_domain.sample(self.config.batch_size, gen))
        return self._examples

    def extract_graphs(self):
        """Stage 1: source-level graphs of both programs at the example shapes."""
        if self._graphs is None:
            x = self.example_inputs()
            g1 = extract_graph(self.tensor_program_1, x, name="program_1")
            g2 = extract_graph(self.tensor_program_2, x, name="program_2")
            self._graphs = (g1, g2)
        return self._graphs

    def match(self) -> MatchResult:
        """Stage 2: matched units, boundary maps, and unmatched or ambiguous regions."""
        if self._match is None:
            g1, g2 = self.extract_graphs()
            c = self.config
            self._match = match_graphs(g1, g2, fold_constants=c.fold_constants,
                                       normalize_arith=c.normalize_arith,
                                       flatten=c.flatten_associative,
                                       recognize_idioms=c.recognize_idioms)
        return self._match

    def _device(self) -> str:
        if self.config.device:
            return self.config.device
        for x in self.example_inputs():
            if getattr(x, "is_cuda", False):
                return "cuda"
        return "cpu"

    def certify(self) -> CertificationResult:
        """Stage 3: abstract interpretation of every matched unit, in dependency order."""
        from .absint import FloatErrorInterpreter, InconclusiveError
        m = self.match()
        cfg = dataclasses.replace(self.config, device=self._device())
        interp = cfg.interpreter or FloatErrorInterpreter()
        name = type(interp).__name__

        def inconclusive(reason, detail, unit="", bounds=None):
            return CertificationResult(
                "inconclusive", m, name, bounds, cfg.epsilon, reason, detail, unit,
                sorted(getattr(interp, "assumptions", ())),
                list(getattr(interp, "reports", ())))

        if not m.complete:
            return inconclusive(Inconclusive.UNSUPPORTED_OPERATOR,
                                f"programs are not fully matched ({len(m.unmatched)} unmatched "
                                "regions); see match.summary()")
        if m.ambiguities and not cfg.allow_ambiguous:
            return inconclusive(Inconclusive.UNSUPPORTED_OPERATOR,
                                f"match has {len(m.ambiguities)} ambiguities; set "
                                "allow_ambiguous=True after reviewing them")
        if cfg.method == "backsub" and cfg.interpreter is None:
            from .backsub import BacksubAnalyzer
            an = BacksubAnalyzer(m, self.input_domain, cfg, self.example_inputs())
            interp, name = an, "BacksubAnalyzer"
            try:
                trip = an.run()
            except InconclusiveError as e:
                return inconclusive(e.reason, e.detail, e.unit)
            except fpm.UnsupportedFormat as e:
                return inconclusive(Inconclusive.UNSUPPORTED_PATH, str(e))
            outs = [ErrorTriple(lo, hi, d) for lo, hi, d in trip]
            res = CertificationResult("bounded", m, name, outs, cfg.epsilon,
                                      assumptions=sorted(an.assumptions),
                                      unit_reports=an.reports, notes=[str(an.stats)],
                                      guarantee=cfg.guarantee)
            return self._finish(res, cfg)
        if cfg.method not in ("backsub", "interval"):
            raise ValueError(f"unknown method {cfg.method!r}")
        env = initial_triples(m, self.input_domain)
        try:
            outs = propagate(m, interp, env, cfg)
        except InconclusiveError as e:
            return inconclusive(e.reason, e.detail, e.unit)
        res = CertificationResult("bounded", m, name, outs, cfg.epsilon,
                                  assumptions=sorted(getattr(interp, "assumptions", ())),
                                  unit_reports=list(getattr(interp, "reports", ())))
        return self._finish(res, cfg)

    @staticmethod
    def _finish(res, cfg):
        if cfg.epsilon is not None:
            worst = max(t.max_error for t in res.output_triples)
            if worst <= cfg.epsilon and cfg.guarantee != "formal":
                res.status = "bounded"          # a probabilistic bound never verifies
                res.detail = f"probabilistic bound {worst:.4g} <= epsilon {cfg.epsilon:g}"
            elif worst <= cfg.epsilon:
                res.status = "verified"
            else:
                res.status = "inconclusive"
                res.inconclusive_reason = Inconclusive.BOUND_ABOVE_EPSILON
                res.detail = f"bound {worst:.4g} > epsilon {cfg.epsilon:g}"
        return res
