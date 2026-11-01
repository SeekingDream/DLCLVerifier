"""Config validation. IMPLEMENTED: checks labels and references, not scientific content."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
STATUS = {"implemented", "pilot", "unimplemented"}
GUARANTEE = {"formal", "formal_real_semantics", "formal_first_order", "formal_equality", "formal_bounded", "probabilistic", "empirical"}


def load(rel):
    with open(ROOT / rel) as f:
        return yaml.safe_load(f)


def validate():
    """Return a list of human-readable problems. Empty means valid."""
    bad = []
    paths = load("configs/compilers.yaml")["paths"]
    ids = [p["id"] for p in paths]
    if len(ids) != len(set(ids)):
        bad.append("duplicate compiler path id")
    for p in paths:
        if p.get("status") not in STATUS:
            bad.append(f"path {p['id']}: bad status")
        if p.get("tier") not in {"core", "core_stretch", "extension"}:
            bad.append(f"path {p['id']}: bad tier")
    for k, s in load("configs/benchmarks.yaml")["suites"].items():
        if s.get("status") not in STATUS:
            bad.append(f"suite {k}: bad status")
    seen = set()
    for b in load("baselines/registry.yaml")["baselines"]:
        if b["id"] in seen:
            bad.append(f"duplicate baseline {b['id']}")
        seen.add(b["id"])
        if b.get("guarantee") not in GUARANTEE:
            bad.append(f"baseline {b['id']}: bad guarantee")
        if b.get("integration") not in STATUS:
            bad.append(f"baseline {b['id']}: bad integration status")
    exp = load("configs/experiments.yaml")
    abl = set(exp["ablations"])
    if abl != {f"A{i}" for i in range(1, 8)}:
        bad.append("ablation set must be A1..A7")
    for e in exp["experiments"]:
        if e.get("status") not in STATUS:
            bad.append(f"experiment {e['id']}: bad status")
    return bad


def rows():
    """Yield (kind, id, status) for the status report."""
    for p in load("configs/compilers.yaml")["paths"]:
        yield "path", p["id"], p["status"]
    for k, s in load("configs/benchmarks.yaml")["suites"].items():
        yield "suite", k, s["status"]
    for b in load("baselines/registry.yaml")["baselines"]:
        yield "baseline", b["id"], b["integration"]
    for e in load("configs/experiments.yaml")["experiments"]:
        yield "experiment", e["id"], e["status"]
