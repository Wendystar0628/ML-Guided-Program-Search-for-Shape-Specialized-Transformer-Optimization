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
- `scripts/`: six end-to-end optimization entry points by workload group and
  search depth.
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

benchmarking/
  suite.py                  ordered, fresh-process Shape measurement
  device_queue.py           one shared lease per CUDA device
  measure.py                in-process measurement used by each worker
```

The main entry point is [`cli.py`](cli.py). GPU commands share one device lease,
so benchmark, profile, search, and optimization jobs do not compete for the same
GPU. A benchmark runs Shapes in order and measures each Shape in a fresh process,
preventing CUDA Graph, allocator, compilation, and model state from leaking into
the next result.

Local observations use one layout:

```text
observations/
  benchmarks/<run-id>/summary.json
  search/search.sqlite3
  search/logs/<run-id>.jsonl
```

`summary.json` is updated after every completed Shape and contains the ordered
results plus the resident geometric-mean speedup. Search JSONL files retain only
high-value milestones; detailed Optuna Trials remain in SQLite. These generated
files are not committed. Deployed winners live in
`deployment/deployed_configs.json`.

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

The `quick`, `standard`, and `deep` tiers target roughly 10 minutes, no more
than 30 minutes, and about two hours on the reference machine. These are soft
runtime classes: persisted search state and convergence stopping may finish a
run earlier, while an already-started GPU measurement may finish later.

```powershell
# Run the complete resident optimization flow
.\scripts\optimize_shapes_01_13_quick.ps1
.\scripts\optimize_shapes_01_13_standard.ps1
.\scripts\optimize_shapes_01_13_deep.ps1

# Run the complete Shape-14 optimization flow
.\scripts\optimize_shape_14_quick.ps1
.\scripts\optimize_shape_14_standard.ps1
.\scripts\optimize_shape_14_deep.ps1

# Inspect the GPU environment
.\.venv\Scripts\python.exe cli.py probe --device cuda:0

# Benchmark Shapes 01-13 in isolated processes (resident is the default group)
.\.venv\Scripts\python.exe cli.py benchmark `
  --preset formal `
  --device cuda:0

# Benchmark Shape 14 separately
.\.venv\Scripts\python.exe cli.py benchmark `
  --group shape14 `
  --preset smoke `
  --device cuda:0

# Generate the final all-Shape competition performance report
.\.venv\Scripts\python.exe `
  "docs\04_最终交付物\01_最终性能测试\run_final_performance.py"

# Search one or more official shapes
.\.venv\Scripts\python.exe cli.py search `
  --case-id official_01 `
  --device cuda:0 `
  --budget-seconds 900

# Benchmark one explicit ConfigSpec in a fresh process
.\.venv\Scripts\python.exe cli.py benchmark `
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
