"""Check utils.diff_testing.

Per task it checks that
  * a program against itself gives error 0,
  * the same seed gives the same result twice,
  * against each compiled program the error is finite, the witness lies in the domain, and
    replaying the witness through both programs reproduces the reported maximum exactly.

    CUDA_VISIBLE_DEVICES=7 python test/test_diff_testing.py
    CUDA_VISIBLE_DEVICES=7 python test/test_diff_testing.py --tasks 2 --test-num 64
"""
import argparse
import os
import sys
import time
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils  # noqa: E402

COMPILERS = [("torch_compile", {}), ("onnxruntime_gpu", {}), ("tensorrt", {}),
             ("triton", {"split_k": 4}), ("tilelang", {"split_k": 4})]


def load(task_id, batch_size):
    try:
        model, loader, domain = utils.load_data_model(task_id, batch_size, batch_size)
        return model, domain
    except Exception as e:
        if "DatasetNotFound" not in type(e).__name__ and "gated" not in str(e):
            raise
        return utils.load_model(task_id), utils.image_domain()   # ImageNet data is gated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, nargs="*", default=[2, 6, 8])
    ap.add_argument("--test-num", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()
    fails = 0

    def report(ok, tid, name, msg):
        nonlocal fails
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'} [{tid:2d}] {name:<28} {msg}", flush=True)

    for tid in args.tasks:
        task = utils.get_task(tid)
        model, dom = load(tid, args.batch_size)
        example = dom.sample(args.batch_size, torch.Generator().manual_seed(1))
        tag = f"{task.model} {dom.domain_id}"

        e0 = utils.diff_testing(model, model, args.test_num, dom, args.batch_size)
        report(e0 == 0.0, tid, "eager vs eager", f"{tag} max_abs={e0}")

        for name, opts in COMPILERS:
            label = name + ("" if not opts else f" {opts}")
            try:
                t = time.time()
                prog = utils.load_compiler(name, **opts)(model, example)
                d = utils.diff_testing(model, prog, args.test_num, dom, args.batch_size,
                                       return_details=True)
                again = utils.diff_testing(model, prog, args.test_num, dom, args.batch_size)
                w, wi = d["witness"], d["witness_index"]
                with torch.no_grad():
                    replay = utils._output_error(model(*(x.cuda() for x in w)),
                                                 prog(*(x.cuda() for x in w)))[wi].max().item()
                problems = []
                if not (0 <= d["max_abs_error"] < float("inf")):
                    problems.append("non-finite error")
                if again != d["max_abs_error"]:
                    problems.append(f"not reproducible ({again} != {d['max_abs_error']})")
                if not dom.contains(w):
                    problems.append("witness outside domain")
                if replay != d["max_abs_error"]:
                    problems.append(f"witness replay {replay} != {d['max_abs_error']}")
                report(not problems, tid, label,
                       f"{tag} max_abs={d['max_abs_error']:.3e} max_rel={d['max_rel_error']:.2e} "
                       f"mean_max={d['mean_max_abs_error']:.2e} ({time.time() - t:.0f}s) "
                       + "; ".join(problems))
                del prog
            except Exception as e:
                traceback.print_exc()
                report(False, tid, label, f"{type(e).__name__}: {str(e)[:300]}")
            torch.cuda.empty_cache()
        del model
        torch.cuda.empty_cache()
    print("SUMMARY", "fail" if fails else "all pass", f"({fails} failures)")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
