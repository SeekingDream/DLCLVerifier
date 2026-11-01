"""UNIMPLEMENTED interfaces. Every call raises so nothing can pass silently as a result."""
from abc import ABC, abstractmethod


class Unimplemented(NotImplementedError):
    pass


class Frontend(ABC):
    """Extract a (source, target) pair from one compiler stage boundary."""
    @abstractmethod
    def extract(self, model, path_config):
        raise Unimplemented


class RelationalAnalyzer(ABC):
    """Compute a certified bound on d(M(x), C(x)) for all x in the domain."""
    @abstractmethod
    def certify(self, pair, domain, spec):
        raise Unimplemented


class CertificateChecker(ABC):
    """Independently re-check a certificate in exact rational arithmetic."""
    @abstractmethod
    def check(self, certificate):
        raise Unimplemented


class Falsifier(ABC):
    """Search for a witness and replay it on the real artifact."""
    @abstractmethod
    def search(self, pair, domain, epsilon, budget):
        raise Unimplemented


class BaselineAdapter(ABC):
    """Translate a benchmark for a baseline and normalize its output to Result."""
    @abstractmethod
    def run(self, benchmark, domain, epsilon, limits):
        raise Unimplemented


def run_experiment(experiment_id):
    raise Unimplemented(f"experiment runner for {experiment_id} is not implemented")
