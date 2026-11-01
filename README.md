# DLCLVerifier evaluation scaffold

Evaluation plan and scaffold for formally verifying numerical consistency between a model and
its compiled form. **The verifier does not exist yet.** This directory has no verification
results.

## Status

| Component | Status |
|---|---|
| Property draft (`docs/SPECIFICATION.md`) | Proposal for review |
| Evaluation plan (`docs/EVALUATION_PLAN.md`) | Written |
| Configs (`configs/`) | Written and validated for structure |
| Outcome rules (`src/dlclv/outcomes.py`) | **Implemented, tested** |
| Config validation and status report (`dlclv`) | **Implemented, tested** |
| Pilot (`pilot/`, `docs/PILOT_RESULTS.md`) | **Ran locally; exploratory evidence only** |
| Relational analyzer, frontends, certificate checker, falsifier | **UNIMPLEMENTED** (`stubs.py`) |
| All baseline adapters | **UNIMPLEMENTED** (`baselines/INTEGRATION_PLAN.md`) |
| Experiment runners | **UNIMPLEMENTED**; `run_experiment` raises |
| Mutation and minifloat soundness suites | **UNIMPLEMENTED** |

## Use

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pytest
PYTHONPATH=src python -m dlclv validate
PYTHONPATH=src python -m dlclv status   # item-by-item implementation labels
```

To rerun the pilot, install `pilot/requirements.lock.txt`. Those pins need a platform that has
those wheels; the recorded run used Python 3.11 on x86-64 macOS:

```bash
python pilot/run_pilot.py --out /path/to/output.json --smt-timeout 90
```

## Layout

```
docs/       specification, evaluation plan, pilot results, references
configs/    compilation paths, benchmarks, domains, experiment protocol
baselines/  baseline registry and integration plan
src/dlclv/  outcome rules, config checks, unimplemented interfaces
pilot/      exploratory measurement script
results/    recorded pilot output
tests/      tests for the implemented parts
```

## Rules

- `status` and `integration` fields must stay truthful: `implemented`, `pilot` or `unimplemented`.
- Every reported outcome goes through `check_result`.
- Sampling never yields `verified`. A loose bound never yields `refuted`.
