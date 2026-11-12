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
    check_extraction: bool = True          # run each program once and compare it with its
                                           # extracted graph (frontends.check_fidelity)
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
    reference: str = "program_1"           # what program 2 is compared with:
                                           # eager: program 1, which must be the PyTorch eager
                                           #        model; the accumulation orders of its
                                           #        kernels are known or probed (probe_orders)
                                           # exact: F, exact real arithmetic on program 1's
                                           #        graph (backsub only)
                                           # program_1: program 1, whatever it is, no probing
    probe_orders: Optional[bool] = None    # rebuild the orders of PyTorch-eager kernels from
                                           # the kernels (probe_eager_orders) before bounding;
                                           # None: yes for reference="eager", no otherwise
    require_known_orders: bool = False     # inconclusive when an accumulation's order is
                                           # neither known nor established by a probe (default:
                                           # bound it over every order and list it in notes)
    method: str = "backsub"                # backsub (per element, relational) | interval
    guarantee: str = "formal"              # formal (worst case) | probabilistic (backsub only)
    confidence: float = 1e-6               # probabilistic: failure probability lambda
    analysis_dtype: str = "float32"        # backsub coefficients: float32 (+inflation) | float64
    backsub_tol: float = 1e-4              # back-substitute where the forward bound exceeds
                                           # this fraction of the value (1e-2 is too loose:
                                           # it makes ReLUs unstable and the bound cascades)
    backsub_memory: float = 1.6e10         # bytes per back-substitution batch (approximate)
    backsub_batch: int = 0                 # rows per batch (0: from backsub_memory)
    backsub_window: int = 0                # intermediate rows: back-substitute through at
                                           # most this many mixing layers (0: all)
    assume_deterministic_kernels: bool = True   # one PyTorch eager kernel on bitwise-equal
                                           # inputs gives bitwise-equal outputs (assumption)
    relu_point_slope: bool = True          # unstable ReLU close to one side: slope 0 or 1
    signed_accumulation: bool = True       # node magnitudes from signed partial sums
                                           # (False: sums of leaf magnitudes)

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
            chk = self.config.check_extraction
            g1 = extract_graph(self.tensor_program_1, x, name="program_1", check=chk)
            g2 = extract_graph(self.tensor_program_2, x, name="program_2", check=chk)
            self._graphs = (g1, g2)
        return self._graphs

    def probe_eager_orders(self, accept: bool = True, **probe_options) -> dict:
        """Rebuild the accumulation order of the PyTorch-eager linear layers of both programs
        (`fprev.probe_eager_linear`: FPRev masks, then bit-exact emulation of the kernel) and
        add the orders that pass to `config.order_evidence`.

        The evidence is dynamic only (level "validated"), so using it is an assumption: with
        `accept=True` it is accepted (`accept_validated_orders`) and every order used is listed
        in `result.assumptions`. Returns {"established": [...], "failed": {key: reason}}.
        """
        from .fprev import eager_linear_evidence
        evidence, failures = eager_linear_evidence(self.extract_graphs(), self._device(),
                                                   **probe_options)
        merged = dict(self.config.order_evidence or {})
        merged.update(evidence)
        self.config = dataclasses.replace(
            self.config, order_evidence=merged,
            accept_validated_orders=self.config.accept_validated_orders or accept)
        return {"established": [ev.kernel + ": " + "; ".join(ev.notes)
                                for ev in evidence.values()],
                "failed": failures}

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
        ref = self.config.reference
        if ref not in ("program_1", "eager", "exact"):
            raise ValueError(f"unknown reference {ref!r}")
        if ref == "eager":
            g1 = m.graph_1
            backends = {g1.meta.get("impl", {}).get("backend", g1.frontend)} | {
                n.meta.get("impl", {}).get("backend", g1.frontend) for n in g1.nodes.values()}
            if g1.frontend != "torch.export" or backends != {"eager"}:
                raise ValueError("reference='eager' needs the PyTorch eager model as program 1 "
                                 f"(got a {g1.frontend} graph run by {sorted(backends)})")
        probe = self.config.probe_orders
        probe_report = None
        if (ref == "eager" if probe is None else probe) and self.config.interpreter is None \
                and self.config.method == "backsub":
            probe_report = self.probe_eager_orders()
        cfg = dataclasses.replace(self.config, device=self._device(),
                                  reference="exact" if ref == "exact" else "program_1")
        interp = cfg.interpreter or FloatErrorInterpreter()
        name = type(interp).__name__

        def inconclusive(reason, detail, unit="", bounds=None):
            return CertificationResult(
                "inconclusive", m, name, bounds, cfg.epsilon, reason, detail, unit,
                sorted(getattr(interp, "assumptions", ())),
                list(getattr(interp, "reports", ())))

        if cfg.interpreter is None:
            from .absint import unmodeled_backends
            missing = unmodeled_backends(m)
            if missing:
                return inconclusive(
                    Inconclusive.UNSUPPORTED_PATH,
                    f"no floating-point model for backend(s) {', '.join(missing)}: the graph "
                    "was extracted and matched, but its kernels (fusion, reduction order, FMA "
                    "contraction) are not modeled, so no bound is claimed")
        if not m.complete:
            return inconclusive(Inconclusive.UNSUPPORTED_OPERATOR,
                                f"programs are not fully matched ({len(m.unmatched)} unmatched "
                                "regions); see match.summary()")
        if m.ambiguities and not cfg.allow_ambiguous:
            return inconclusive(Inconclusive.UNSUPPORTED_OPERATOR,
                                f"match has {len(m.ambiguities)} ambiguities; set "
                                "allow_ambiguous=True after reviewing them")
        if cfg.reference == "exact" and (cfg.method != "backsub" or cfg.interpreter is not None):
            raise ValueError("reference='exact' needs method='backsub'")
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
            unknown = sorted(an.oracle.unknown_orders)
            notes = [str(an.stats), f"reference: {ref}"]
            if unknown:
                notes.append("order unknown, bounded over every order: " + "; ".join(unknown))
            for key, why in (probe_report or {}).get("failed", {}).items():
                notes.append(f"order probe failed for {key}: {why}")
            if unknown and cfg.require_known_orders:
                res = inconclusive(Inconclusive.UNSUPPORTED_PATH,
                                   "accumulation orders neither known nor established by a "
                                   "probe: " + "; ".join(unknown))
                res.notes = notes
                return res
            res = CertificationResult("bounded", m, name, outs, cfg.epsilon,
                                      assumptions=sorted(an.assumptions),
                                      unit_reports=an.reports, notes=notes,
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
