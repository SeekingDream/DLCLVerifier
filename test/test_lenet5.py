"""Difference-bound certifier on LeNet-5 (random weights, 1x32x32 input in [0, 1]).

PyTorch eager is the reference (program 1) in every pair. Two variants: the classic network
(tanh, average pooling) and the common ReLU / max-pooling one. Three domains: one sample, the
L-inf ball of radius 1/255 around it, and the whole pixel box.

Three bounds are reported for every pair: against eager with every accumulation order unknown,
against eager with the linear layers' orders rebuilt from the kernels (`reference="eager"`; the
convolutions keep the any-order bound), and against F, the model in exact arithmetic
(`reference="exact"`). The observed column is the largest difference an adversarial search found
(`utils.search_max_diff`, `--samples` program runs), next to the uniform-sampling maximum.

    CUDA_VISIBLE_DEVICES=7 python test/test_lenet5.py
"""
import argparse
import os
import sys
import time

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
import utils  # noqa: E402
from dlclv.certifier import TensorProgramCertifier  # noqa: E402


class LeNet5(nn.Module):
    def __init__(self, classic=True):
        super().__init__()
        act = nn.Tanh if classic else nn.ReLU
        pool = nn.AvgPool2d if classic else nn.MaxPool2d
        self.features = nn.Sequential(nn.Conv2d(1, 6, 5), act(), pool(2),
                                      nn.Conv2d(6, 16, 5), act(), pool(2))
        self.classifier = nn.Sequential(nn.Linear(16 * 5 * 5, 120), act(), nn.Linear(120, 84),
                                        act(), nn.Linear(84, 10))

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


PATHS = [("eager", {}), ("triton", {}), ("triton", {"split_k": 4}), ("tilelang", {}),
         ("onnxruntime_gpu", {}), ("torch_compile", {})]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=1000)
    args = ap.parse_args()
    box = utils.InputDomain("lenet_pixels", "box", ["pixels"], [(1, 32, 32)], [torch.float32],
                            [0.0], [1.0])
    fails = 0
    for classic in (True, False):
        torch.manual_seed(0)
        model = LeNet5(classic).cuda().eval()
        x = tuple(t.cuda() for t in box.sample(1, torch.Generator().manual_seed(0)))
        domains = [("point", utils.local_domain(box, tuple(t[0] for t in x), 0.0)),
                   ("ball 1/255", utils.local_domain(box, tuple(t[0] for t in x), 1 / 255)),
                   ("box [0,1]", box)]
        print(f"\n=== LeNet-5 ({'tanh, avg-pool' if classic else 'ReLU, max-pool'}) ===")
        print(f"{'program 2':<18} {'domain':<11} {'vs eager,':>11} {'vs eager,':>11} "
              f"{'vs F':>11} {'observed':>10} {'(uniform)':>10}")
        print(f"{'':<18} {'':<11} {'any order':>11} {'probed':>11} {'(exact)':>11} "
              f"{'vs eager':>10} {'vs eager':>10}  found by")
        shown = False
        base = {"example_inputs": x, "assume_sum_of_products_conv": True}
        for name, opts in PATHS:
            label = name + "".join(f" {k}={v}" for k, v in opts.items())
            try:
                c = utils.load_compiler(name, **opts)(model, x)
            except Exception as e:
                print(f"{label:<18} compile failed: {type(e).__name__}: {str(e)[:120]}")
                fails += 1
                continue
            for dname, dom in domains:
                t = time.time()
                try:
                    # the same pair under the three references the tool supports
                    res0 = TensorProgramCertifier(model, c, dom, base).certify()
                    res = TensorProgramCertifier(model, c, dom,
                                                 dict(base, reference="eager")).certify()
                    resf = TensorProgramCertifier(model, c, dom, dict(
                        base, reference="exact", probe_orders=True)).certify()
                except Exception as e:
                    print(f"{label:<18} {dname:<11} ERROR {type(e).__name__}: {str(e)[:150]}")
                    fails += 1
                    continue
                if not shown and name != "eager":
                    shown = True
                    for a in res.assumptions:
                        if a.startswith("order["):
                            print(f"  probed {a[:150]}")
                    for nt in res.notes:
                        if nt.startswith("order"):
                            print(f"  {nt[:200]}")
                det = utils.search_max_diff(model, c, dom, args.samples, model=model,
                                            return_details=True)
                obs = det["max_abs_error"]
                b0, b, bf = res0.certified_bound, res.certified_bound, resf.certified_bound
                unsound = any(v is not None and obs > v for v in (b0, b))
                fails += unsound
                fmt = lambda v: "-" if v is None else f"{v:.3e}"  # noqa: E731
                print(f"{label:<18} {dname:<11} {fmt(b0):>11} {fmt(b):>11} {fmt(bf):>11} "
                      f"{obs:>10.3e} {det['uniform_max']:>10.3e}  "
                      f"{'UNSOUND' if unsound else det['phase']} ({time.time() - t:.1f}s)",
                      flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
