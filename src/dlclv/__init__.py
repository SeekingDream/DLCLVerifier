"""DLCLVerifier evaluation scaffold.

Implemented: config validation, outcome rules, graph extraction (`frontends`), canonical lowering
(`canonical`), graph matching (`matching`), the floating-point error model (`fpmodel`), FPRev
order revelation (`fprev`), abstract interpretation of matched units (`absint`), and the
certifier driver (`certifier`).
NOT implemented: an independent certificate checker, baseline adapters, and experiment runners.
"""
__version__ = "0.0.3"


def __getattr__(name):
    # Lazy so that `python -m dlclv` and config tests do not import torch or onnx.
    if name in ("TensorProgramCertifier", "CertifierConfig", "CertificationResult"):
        from . import certifier
        return getattr(certifier, name)
    raise AttributeError(name)
