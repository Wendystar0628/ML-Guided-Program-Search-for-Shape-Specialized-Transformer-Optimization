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

- `solution/`: code that executes a generated Transformer program.
- `autotune/`: code that generates, searches, evaluates, and repeats candidate
  programs.
- `benchmarking/`: official-protocol measurement, profiling, and hardware probe.
- `deployment/`: measured per-device winners consumed by `solution/`.
- `official/`: upstream benchmark semantics and the 14 workload definitions.
- `tests/`: control-plane and GPU-path tests.
- `scripts/`: the two public end-to-end optimization entry points.
- `environments/`: machine-specific dependencies and environment activation.
- `docs/`: competition rules, engineering notes, and deliverables.

The performance mainline is organized by responsibility:

```text
solution/
  config.py                 typed program and launch configuration
  plan_builder.py           ConfigSpec -> validated ExecutionPlan
  plan.py                   immutable execution plan and expected trace
  transformer.py            model that executes the plan
  operators/                PyTorch, SDPA, and compiled operator compositions
  kernels/                  handwritten Triton candidates by subsystem
  runtimes/                 eager/compile/CUDA Graph/streamed schedules

autotune/
  search_space.py           generated structures and parameter domains
  search_engine.py          one-shape staged search
  search_sweep.py           one sweep across a shape group
  optimization_loop.py      repeated sweeps and convergence stopping
  run_log.py                compact append-only search decision history
  evaluation.py             trial and promotion measurements
  optuna_backend.py         Optuna TPE adapter
  study_storage.py          local study identity and SQLite location
```

The main entry point is [`cli.py`](cli.py). Generated Optuna state lives in
`search_state/search.sqlite3`. Human-readable search and optimization milestones
are appended to one JSONL file per invocation under `search_state/runs/`. Both
are local working data and are not committed. The JSONL stays intentionally
small; use the SQLite database only when an individual Trial needs inspection.
Direct benchmark output defaults to `benchmark_runs/`. Deployed winners are
stored in `deployment/deployed_configs.json` and match the complete measured
software and hardware environment.

## Setup

For the validated native Windows RTX 4080 environment:

```powershell
python -m venv .venv
.\environments\activate_windows_rtx4080.ps1
python -m pip install -r environments\windows-rtx4080.txt
```

On another platform, install the CUDA-enabled PyTorch and Triton builds selected
for that platform first, then install `requirements.txt`. The project does not
force one cross-platform GPU runtime combination.

## Commands

```powershell
# Run the complete resident or Shape-14 optimization flow
.\scripts\optimize_shapes_01_13.ps1
.\scripts\optimize_shape_14.ps1

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

[`cli.py`](cli.py) is the development and search CLI.
[`torch_transformer_benchmark.py`](torch_transformer_benchmark.py) is the
separate bridge that runs the immutable official benchmark against the current
solution.
