"""Difference-bound certifier on one linear layer, y = W x + b (random weights).

PyTorch eager is the reference (program 1) in every pair. Input: a vector of K values in
[-1, 1]; output: 10 values. Three domains: one sample, the L-inf ball of radius 0.01 around it,
and the whole box.

Three bounds are reported for every pair: against eager with its kernel's accumulation order
unknown (any order), against eager with the order rebuilt from the kernel (`reference="eager"`:
FPRev masks, then bit-exact emulation), and against F, the model in exact arithmetic
(`reference="exact"`). The observed column is the largest difference an adversarial search
found (`utils.search_max_diff`, `--samples` program runs), with the uniform-sampling maximum
next to it.

    CUDA_VISIBLE_DEVICES=7 python test/test_linear.py
    CUDA_VISIBLE_DEVICES=7 python test/test_linear.py --k 64 512 4096 --show
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

PATHS = [("eager", {}), ("triton", {}), ("triton", {"split_k": 4}), ("tilelang", {}),
         ("onnxruntime_gpu", {}), ("torch_compile", {})]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, nargs="*", default=[64, 512, 4096])
    ap.add_argument("--out-features", type=int, default=10)
    ap.add_argument("--radius", type=float, default=0.01)
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--show", action="store_true", help="print the match and the assumptions")
    args = ap.parse_args()
    fails = 0
    for K in args.k:
        torch.manual_seed(0)
        model = nn.Linear(K, args.out_features).cuda().eval()
        box = utils.InputDomain(f"linear_{K}", "box", ["x"], [(K,)], [torch.float32],
                                [-1.0], [1.0])
        x = tuple(t.cuda() for t in box.sample(1, torch.Generator().manual_seed(0)))
        point = tuple(t[0] for t in x)
        domains = [("point", utils.local_domain(box, point, 0.0)),
                   (f"ball {args.radius:g}", utils.local_domain(box, point, args.radius)),
                   ("box [-1,1]", box)]
        print(f"\n=== Linear({K}, {args.out_features}) ===")
        print(f"{'program 2':<18} {'domain':<11} {'vs eager,':>11} {'vs eager,':>11} "
              f"{'vs F':>11} {'observed':>10} {'(uniform)':>10}")
        print(f"{'':<18} {'':<11} {'any order':>11} {'probed':>11} {'(exact)':>11} "
              f"{'vs eager':>10} {'vs eager':>10}  found by")
        shown = False
        base = {"example_inputs": x}
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
