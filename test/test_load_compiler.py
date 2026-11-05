"""Check utils.load_compiler on tasks x compilers.

For each pair it compiles the model on the first batch, runs every batch twice, and checks:
  * outputs have the eager shape and are finite,
  * two runs of the compiled model are bitwise equal (deterministic),
  * precision is unchanged: the error against a float64 run of the same model is at most
    PREC_FACTOR times the eager float32 error, so a silent TF32 or fp16 path fails,
  * the triton and tilelang paths actually replaced GEMM-shaped ops.
It also reports the drift from eager, which is what the verifier has to bound.

If a task's data is gated, inputs are drawn from its input domain instead (marked "domain").

    CUDA_VISIBLE_DEVICES=7 python test/test_load_compiler.py
    CUDA_VISIBLE_DEVICES=7 python test/test_load_compiler.py --tasks 2 6 --compilers triton tilelang
"""
import argparse
import copy
import json
import os
import sys
import time
import traceback

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import utils  # noqa: E402

PREC_FACTOR = 32.0
DEFAULT_TASKS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13]   # skip gated llama and CPU-only int8


def rel(a, b):
    return ((a.double() - b.double()).abs().max() / b.double().abs().max().clamp(min=1e-30)).item()


def load_task(task_id, data_num, batch_size):
    try:
        model, loader, domain = utils.load_data_model(task_id, data_num, batch_size)
        return model, [x for x, _ in loader], "data"
    except Exception as e:
        if not any(k in f"{type(e).__name__} {e}" for k in ("Gated", "401", "gated", "DatasetNotFound")):
            raise
    model = utils.load_model(task_id)
    dom = utils.image_domain()
    g = torch.Generator().manual_seed(0)
    batches = [dom.sample(min(batch_size, data_num - i), g) for i in range(0, data_num, batch_size)]
    return model, batches, "domain"


def run_all(fn, batches, dev):
    with torch.no_grad():
        return torch.cat([fn(*(x.to(dev) for x in b)).float() for b in batches])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, nargs="*", default=DEFAULT_TASKS)
    ap.add_argument("--compilers", nargs="*", default=list(utils.list_compilers().values()))
    ap.add_argument("--data-num", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "runs", "test_load_compiler.json"))
    args = ap.parse_args()
    print(f"torch {torch.__version__}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}, "
          f"{torch.cuda.get_device_name()}", flush=True)

    rows, fails = [], 0
    for tid in args.tasks:
        task = utils.get_task(tid)
        try:
            model, batches, src = load_task(tid, args.data_num, args.batch_size)
        except Exception as e:
            print(f"FAIL [{tid:2d}] {task.model}: load {type(e).__name__}: {e}", flush=True)
            fails += 1
            continue
        dev = next(iter(model.state_dict().values())).device
        ref = run_all(model, batches, dev)
        m64 = copy.deepcopy(model).double()
        b64 = [tuple(x.double() if x.is_floating_point() else x for x in b) for b in batches]
        ref64 = run_all(m64, b64, dev).double()
        del m64
        torch.cuda.empty_cache()
        eager_err = rel(ref, ref64)

        for cname in args.compilers:
            row = {"task": tid, "model": task.model, "dataset": task.dataset, "inputs": src,
                   "compiler": cname}
            t0 = time.time()
            try:
                compiled = utils.load_compiler(cname)(model, batches[0])
                out1 = run_all(compiled, batches, dev)
                out2 = run_all(compiled, batches, dev)
                assert out1.shape == ref.shape, f"shape {tuple(out1.shape)} != {tuple(ref.shape)}"
                assert torch.isfinite(out1).all(), "non-finite output"
                err64 = rel(out1, ref64)
                row.update(
                    compile_s=round(compiled.compile_seconds, 1),
                    drift_vs_eager=rel(out1, ref),
                    max_abs_drift=(out1 - ref).abs().max().item(),
                    bitwise_equal_frac=(out1 == ref).float().mean().item(),
                    top1_agree=(out1.argmax(-1) == ref.argmax(-1)).float().mean().item(),
                    err_vs_fp64=err64, eager_err_vs_fp64=eager_err,
                    deterministic=bool(torch.equal(out1, out2)),
                    swaps=dict(compiled.stats) if compiled.stats is not None else None)
                problems = []
                if not row["deterministic"]:
                    problems.append("nondeterministic")
                if err64 > max(PREC_FACTOR * eager_err, 1e-6):
                    problems.append(f"precision changed ({err64:.1e} vs eager {eager_err:.1e})")
                if cname in ("triton", "tilelang") and not any(
                        not k.startswith("fallback") for k in (compiled.stats or {})):
                    problems.append("no op was swapped")
                row["status"] = "FAIL" if problems else "PASS"
                row["problems"] = problems
                del compiled
            except Exception as e:
                row.update(status="FAIL", problems=[f"{type(e).__name__}: {str(e)[:400]}"])
                traceback.print_exc()
            row["wall_s"] = round(time.time() - t0, 1)
            fails += row["status"] == "FAIL"
            rows.append(row)
            msg = (f"drift={row['drift_vs_eager']:.2e} bitwise={row['bitwise_equal_frac']:.3f} "
                   f"top1={row['top1_agree']:.3f} fp64err={row['err_vs_fp64']:.1e} "
                   f"(eager {eager_err:.1e}) compile={row['compile_s']}s"
                   if "drift_vs_eager" in row else "")
            if row.get("swaps"):
                msg += " swaps=" + ",".join(f"{k}:{v}" for k, v in sorted(row["swaps"].items()))
            print(f"{row['status']} [{tid:2d}] {task.model:<22} {cname:<16} {src:<6} {msg} "
                  f"{'; '.join(row['problems'])}", flush=True)
            torch.cuda.empty_cache()
        del model
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=1)
    print(f"SUMMARY pass={sum(r['status'] == 'PASS' for r in rows)} fail={fails} -> {args.out}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
