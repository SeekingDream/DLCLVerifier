"""Check the per-program rounding bound elementwise against real GPU kernels.

For one-layer models (conv, linear) at a single input, every output element must satisfy

    |program(x) - exact(x)| <= E        (exact: the same layer in float64)

where E is the certifier's bound for that program (kind E2 of the back-substitution analysis).
The script also reports how loose the bound is (bound / observed error, per element) with and
without cancellation-aware node magnitudes (`signed_accumulation`).

    CUDA_VISIBLE_DEVICES=7 python test/test_layer_bounds.py
"""
import dataclasses
import os
import sys

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
import utils  # noqa: E402
from dlclv.backsub import KE2, BacksubAnalyzer  # noqa: E402
from dlclv.certifier import TensorProgramCertifier  # noqa: E402

CASES = {
    "conv7x7 C=3 (K=147)": (lambda: nn.Conv2d(3, 64, 7, stride=2, padding=3), (1, 3, 64, 64)),
    "conv3x3 C=64 (K=576)": (lambda: nn.Conv2d(64, 64, 3, padding=1), (1, 64, 28, 28)),
    "conv3x3 C=512 (K=4608)": (lambda: nn.Conv2d(512, 512, 3, padding=1), (1, 512, 7, 7)),
    "linear K=512": (lambda: nn.Linear(512, 1000), (1, 512)),
    "linear K=4096": (lambda: nn.Linear(4096, 1000), (1, 4096)),
}
PATHS = [("triton", {}), ("triton", {"split_k": 4}), ("tilelang", {}), ("eager", {})]


class Point:
    """The domain that holds exactly one input."""

    def __init__(self, x):
        self.lower = [t[0].double().cpu() for t in x]
        self.upper = self.lower


def bound(model, compiled, x, signed):
    cert = TensorProgramCertifier(model, compiled, Point(x), {
        "example_inputs": x, "assume_sum_of_products_conv": True,
        "signed_accumulation": signed, "assume_deterministic_kernels": False})
    an = BacksubAnalyzer(cert.match(), cert.input_domain,
                         dataclasses.replace(cert.config, device="cuda"), x)
    an.run()
    return an.bnd[cert.match().output_pairs[0][0].value_1][KE2]


def main():
    torch.manual_seed(0)
    fails = 0
    for name, (make, shape) in CASES.items():
        model = make().cuda().eval()
        for dist in ("nonnegative", "signed"):
            x = torch.randn(shape, device="cuda")
            x = (x.relu() if dist == "nonnegative" else x,)
            with torch.no_grad():
                exact = model.double()(x[0].double())
                model.float()
            for comp, opts in PATHS:
                c = utils.load_compiler(comp, **opts)(model, x)
                with torch.no_grad():
                    err = (c(*x).double() - exact).abs()
                ratio = {}
                bad = 0
                for signed in (True, False):
                    E = bound(model, c, x, signed).reshape(err.shape)
                    bad += int((err > E).sum())
                    ratio[signed] = float((E / err.clamp(min=1e-30)).median())
                fails += bad > 0
                label = comp + ("" if not opts else " " + ",".join(f"{k}={v}" for k, v in opts.items()))
                print(f"{'FAIL' if bad else 'PASS'} {name:24s} {dist:11s} {label:16s} "
                      f"max_err={float(err.max()):.2e} median bound/err: signed "
                      f"{ratio[True]:8.1f}, magnitudes {ratio[False]:8.1f}", flush=True)
    print("SUMMARY", "PASS" if not fails else f"FAIL ({fails})")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
