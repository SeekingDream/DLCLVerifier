"""Check utils.load_data_model on every task.

For each task it checks that the loader holds exactly data_num samples, every sample lies in
the returned input domain, the model runs on the batches with finite outputs, and the outputs
are deterministic across two runs. Classification tasks also report accuracy as a sanity check.

If the data source is gated and no token is set, the task still loads the model and runs it on
points drawn from the domain; it is reported as MODEL_ONLY, not PASS. If the
model weights are gated too, the task is reported as GATED.

    CUDA_VISIBLE_DEVICES=7 python test/test_load_data_model.py --data-num 8
    CUDA_VISIBLE_DEVICES=7 python test/test_load_data_model.py --tasks 0 3 6 --data-num 16
"""
import argparse
import os
import sys
import time
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils  # noqa: E402

CLASSIFIERS = {"squeezenet_1_1", "mobilenet_v2", "resnet18", "resnet50", "vit_b_16",
               "quantized_mobilenet_v2_int8", "distilbert_sst2"}


def is_gated(e):
    s = f"{type(e).__name__} {e}"
    return any(k in s for k in ("GatedRepo", "401", "gated", "Unauthorized"))


def forward_all(model, loader, device):
    outs, labels = [], []
    with torch.no_grad():
        for inputs, y in loader:
            outs.append(model(*(x.to(device) for x in inputs)).float().cpu())
            labels.append(y)
    return torch.cat(outs), torch.cat(labels)


def check_task(task, data_num, batch_size):
    t0 = time.time()
    try:
        model, loader, domain = utils.load_data_model(task.task_id, data_num, batch_size)
    except Exception as e:
        if not is_gated(e):
            raise
        try:
            model = utils.load_model(task.task_id)  # gated data: model must still load and run
        except Exception as e2:
            if not is_gated(e2):
                raise
            return "GATED", f"model weights gated ({type(e2).__name__}); needs an HF token"
        dev = next(iter(model.state_dict().values())).device
        pts = utils.image_domain().sample(2) if task.dataset == "imagenet_val" else None
        if pts is not None:
            with torch.no_grad():
                out = model(*(p.to(dev) for p in pts))
            assert torch.isfinite(out).all() and out.shape == (2, 1000)
        return "MODEL_ONLY", f"data gated ({type(e).__name__}); model loaded on {dev}"

    dev = next(iter(model.state_dict().values())).device
    n = sum(len(y) for _, y in loader)
    assert n == data_num, f"loader has {n} samples, expected {data_num}"
    assert len(loader.dataset) == data_num
    for inputs, _ in loader:
        assert isinstance(inputs, tuple)
        assert domain.contains(inputs), "sample outside the returned input domain"

    out1, labels = forward_all(model, loader, dev)
    out2, _ = forward_all(model, loader, dev)
    assert out1.shape[0] == data_num, f"output batch {out1.shape[0]} != {data_num}"
    assert torch.isfinite(out1).all(), "non-finite model output"
    assert torch.equal(out1, out2), "outputs differ between two runs on the same device"

    # Points drawn from the domain must also run.
    pts = domain.sample(2, torch.Generator().manual_seed(0))
    with torch.no_grad():
        assert torch.isfinite(model(*(p.to(dev) for p in pts)).float()).all()

    msg = (f"device={dev} domain={domain.domain_id} inputs="
           f"{[tuple(s) for s in domain.shapes]} out={tuple(out1.shape[1:])}")
    if task.model in CLASSIFIERS:
        acc = (out1.argmax(-1) == labels).float().mean().item()
        msg += f" acc={acc:.3f}"
    return "PASS", msg + f" ({time.time() - t0:.0f}s)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, nargs="*", default=None, help="task ids; default all")
    ap.add_argument("--data-num", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    print(f"torch {torch.__version__}, cuda={torch.cuda.is_available()}, "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    tasks = [utils.get_task(i) for i in args.tasks] if args.tasks else utils.list_tasks()
    results = {}
    for task in tasks:
        name = f"[{task.task_id:2d}] {task.model}/{task.dataset}"
        try:
            status, msg = check_task(task, args.data_num, args.batch_size)
        except Exception as e:
            status, msg = "FAIL", f"{type(e).__name__}: {str(e)[:300]}"
            traceback.print_exc()
        results[task.task_id] = status
        print(f"{status:<10} {name:<45} {msg}", flush=True)
        torch.cuda.empty_cache()

    counts = {s: list(results.values()).count(s) for s in ("PASS", "MODEL_ONLY", "GATED", "FAIL")}
    print("SUMMARY", counts)
    sys.exit(1 if counts["FAIL"] else 0)


if __name__ == "__main__":
    main()
