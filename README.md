# Shape-Aware Transformer Program Search

This project generates legal Transformer execution plans, measures them on the
target GPU, and deploys the fastest plan that passes the official comparator.

```text
ProgramSearchSpace -> ConfigSpec -> PlanBuilder -> ExecutionPlan
                   -> GPU benchmark -> TPE selection -> deployment registry
```

The 14 workloads are defined by
[`official/test_shapes.json`](official/test_shapes.json). Shapes 1-13 use
resident execution. Shape 14 uses streamed microbatches because its logical
batch cannot reside in GPU memory at once.

## Project structure

- `solution/`: Transformer model, execution-plan builder, operators, and runtime
  strategies.
- `autotune/`: generated search space, TPE search, candidate evaluation, and
  search-state storage.
- `benchmarking/`: official-protocol measurement, profiling, and hardware probe.
- `deployment/`: measured per-device winners consumed by the runtime.
- `official/`: upstream benchmark semantics and the 14 workload definitions.
- `tests/`: control-plane and GPU-path tests.
- `scripts/`: local development helpers.
- `environments/`: machine-specific dependency files.
- `docs/`: competition rules, engineering notes, and deliverables.

The main entry point is [`cli.py`](cli.py). Generated Optuna state lives in
`search_state/search.sqlite3`; it is local working data and is not committed.
Direct benchmark output defaults to `benchmark_runs/`. Deployed winners are
stored in `deployment/deployed_configs.json` and match the complete measured
software and hardware environment.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For the validated native Windows RTX 4080 environment:

```powershell
.\scripts\activate_windows_rtx4080.ps1
python -m pip install -r environments\windows-rtx4080.txt
```

## Commands

```powershell
# Inspect the GPU environment
.\.venv\Scripts\python.exe cli.py probe --device cuda:0

# Search one or more official shapes
.\.venv\Scripts\python.exe cli.py search `
  --case-id official_01 `
  --device cuda:0 `
  --budget-seconds 900

# Measure one explicit ConfigSpec
.\.venv\Scripts\python.exe cli.py run `
  --case-id official_01 `
  --config path\to\config.json `
  --preset smoke `
  --device cuda:0

# Profile one explicit ConfigSpec
.\.venv\Scripts\python.exe cli.py profile `
  --case-id official_01 `
  --config path\to\config.json `
  --device cuda:0
```

`python -m benchmarking` exposes the same command line interface.
