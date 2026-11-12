"""Run TensorProgramCertifier end to end on real models and compiled programs.

Pairs per task:
  model vs eager / triton(split_k=4) / tilelang / onnxruntime_gpu
  triton vs tilelang                  both known sequential FMA orders (Strategy 1, relational)
  ONNX export vs ORT-optimized graph  ONNX Runtime's own fusions (FusedMatMul, BiasGelu, ...)
  model vs torch_compile / tensorrt   extraction not implemented: must raise Unimplemented

For every bounded result it samples the domain with utils.diff_testing (at the certified batch
size) and checks soundness: the observed maximum error must not exceed the certified bound.
An inconclusive result passes when it gives a reason; a bound below an observed error FAILS.

    CUDA_VISIBLE_DEVICES=7 python test/test_certifier_models.py
    CUDA_VISIBLE_DEVICES=7 python test/test_certifier_models.py --tasks 0 2 --show
    CUDA_VISIBLE_DEVICES=7 python test/test_certifier_models.py --tasks 2 --radius 0
    CUDA_VISIBLE_DEVICES=7 python test/test_certifier_models.py --tasks 2 --radius 0 \
        --configure '{"guarantee": "probabilistic"}'

    CUDA_VISIBLE_DEVICES=7 python test/test_certifier_models.py --tasks 2 --radius 0 \
        --configure '{"reference": "exact"}'

--radius R certifies on the L-inf ball of radius R around the sample (0: the sample itself)
instead of the task's global domain; --configure adds CertifierConfig fields. With
reference=exact each "model vs X" bound is |X - exact arithmetic| and is checked against the
model run in float64.
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
sys.path.insert(0, os.path.join(ROOT, "src"))
import utils  # noqa: E402
from dlclv.certifier import TensorProgramCertifier  # noqa: E402
from dlclv.stubs import Unimplemented  # noqa: E402

MUST_MATCH = {"eager", "triton", "tilelang", "triton vs tilelang"}
CONFIG = {"assume_sum_of_products_conv": True}


def domain_for(tid, model):
    if utils.get_task(tid).dataset == "imagenet_val":
        return utils.image_domain()
    return utils.load_data(tid, 1, model)[1]


def float64_program(model):
    m64 = copy.deepcopy(model).double()
    return lambda *xs: m64(*[x.double() if x.is_floating_point() else x for x in xs])


def run_pair(label, p1, p2, dom, x, show, samples, extra=None):
    t = time.time()
    row = {"pair": label}
    cert = TensorProgramCertifier(p1, p2, dom, dict(CONFIG, example_inputs=x, **(extra or {})))
    try:
        cert.extract_graphs()
    except Unimplemented as e:
        row.update(status="UNSUPPORTED", detail=str(e)[:160])
        return row
    m = cert.match()
    row.update(m.stats())
    row["regions"] = [f"{'+'.join(r.ops_1) or '-'} <-> {'+'.join(r.ops_2) or '-'}: {r.reason}"
                      for r in m.unmatched]
    row["fusions"] = sorted({f"{'+'.join(u.ops_1)} <-> {'+'.join(u.ops_2)}"
                             for u in m.units if u.kind != "one_to_one"})
    res = cert.certify()
    row["certify"] = res.status
    row["reason"] = res.inconclusive_reason.value if res.inconclusive_reason else None
    row["detail"] = res.detail[:300]
    row["bound"] = res.certified_bound
    row["assumptions"] = res.assumptions
    # per-accumulation reports exist for method="interval" only (backsub reports are dicts)
    accs = [a for r in res.unit_reports for a in getattr(r, "accumulations", ())]
    row["strategies"] = sorted({a.get(f"order_{s}", "") for a in accs for s in (1, 2)} - {""})
    row["relational"] = sum(1 for a in accs if a.get("D") is not None)
    if res.status == "inconclusive" and res.inconclusive_reason is None:
        row.update(status="FAIL", detail="inconclusive without a reason")
    if (extra or {}).get("reference") == "exact" and isinstance(p1, torch.nn.Module):
        p1 = float64_program(p1)       # the bound is |p2 - exact|: observe against float64
    if res.output_triples is not None and callable(p1) and callable(p2) and samples:
        n = 1 if getattr(dom, "kind", "") == "point" else samples
        obs = utils.diff_testing(p1, p2, n, dom, batch_size=x[0].shape[0])
        row["observed"] = obs
        if obs > res.certified_bound:
            row.update(status="FAIL", detail=f"UNSOUND: observed {obs:.3e} > bound "
                                             f"{res.certified_bound:.3e}")
    row.setdefault("status", "PASS")
    row["seconds"] = round(time.time() - t, 1)
    if show:
        print(m.summary(max_items=8))
        print(res.summary())
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, nargs="*", default=[0, 2, 3, 6, 8, 10])
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--radius", type=float, default=None)
    ap.add_argument("--configure", default="{}")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "runs", "certifier_models.json"))
    args = ap.parse_args()
    rows, fails = [], 0
    extra = json.loads(args.configure)
    for tid in args.tasks:
        task = utils.get_task(tid)
        model = utils.load_model(tid)
        dom = domain_for(tid, model)
        x = tuple(t.cuda() for t in dom.sample(1, torch.Generator().manual_seed(0)))
        if args.radius is not None:
            dom = utils.local_domain(dom, tuple(t[0] for t in x), args.radius)
        pairs, compiled = [], {}
        for name, opts in [("eager", {}), ("triton", {"split_k": 4}), ("tilelang", {}),
                           ("onnxruntime_gpu", {}), ("torch_compile", {}), ("tensorrt", {})]:
            try:
                c = utils.load_compiler(name, **opts)(model, x)
            except Exception as e:
                rows.append({"task": tid, "pair": f"model vs {name}", "status": "FAIL",
                             "detail": f"compile: {type(e).__name__}: {e}"[:300]})
                fails += 1
                continue
            compiled[name] = c
            pairs.append((f"model vs {name}", model, c, name))
            if name == "onnxruntime_gpu":
                pairs.append(("onnx vs ort_optimized", c.meta["onnx"],
                              c.meta["ort_optimized_onnx"], "ort_stage"))
        try:
            tri = utils.load_compiler("triton")(model, x)
            if "tilelang" in compiled:
                pairs.append(("triton vs tilelang", tri, compiled["tilelang"],
                              "triton vs tilelang"))
        except Exception as e:
            rows.append({"task": tid, "pair": "triton vs tilelang", "status": "FAIL",
                         "detail": f"compile: {type(e).__name__}: {e}"[:300]})
        for label, p1, p2, name in pairs:
            try:
                row = run_pair(label, p1, p2, dom, x, args.show, args.samples, extra)
            except Exception as e:
                traceback.print_exc()
                row = {"pair": label, "status": "FAIL", "detail": f"{type(e).__name__}: {e}"[:400]}
            row["task"], row["model"] = tid, task.model
            if name in MUST_MATCH and row.get("status") == "PASS" and not row.get("complete"):
                row["status"] = "FAIL"
                row["detail"] = "same graph with different kernels did not match completely"
            if name in ("torch_compile", "tensorrt") and row.get("status") != "UNSUPPORTED":
                row["status"] = "FAIL"
                row["detail"] = "expected Unimplemented extraction"
            elif row.get("status") == "UNSUPPORTED":
                row["status"] = "PASS"
                row["certify"] = "extraction Unimplemented"
            fails += row["status"] == "FAIL"
            rows.append(row)
            b, o = row.get("bound"), row.get("observed")
            print(f"{row['status']} [{tid:2d}] {task.model:<18} {label:<24} "
                  f"complete={row.get('complete')} certify={row.get('certify')} "
                  f"bound={'-' if b is None else f'{b:.3e}'} "
                  f"observed={'-' if o is None else f'{o:.3e}'} "
                  f"reason={row.get('reason')} ({row.get('seconds')}s)", flush=True)
            if row.get("detail"):
                print(f"        {row['detail'][:260]}")
            if row.get("strategies"):
                print(f"        orders: {row['strategies']}  relational units: "
                      f"{row.get('relational')}")
        del model, pairs, compiled
        torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=1, default=str)
    print(f"SUMMARY pass={sum(r['status'] == 'PASS' for r in rows)} fail={fails} -> {args.out}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
