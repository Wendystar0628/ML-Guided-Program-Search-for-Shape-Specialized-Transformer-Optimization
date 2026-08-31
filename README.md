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
- `deployment/`: environment identities and measured exact-device winners
  consumed by `solution/`.
- `official/`: upstream benchmark semantics and the 14 workload definitions.
- `tests/`: tests grouped by `autotune/`, `benchmarking/`, `entrypoints/`,
  `official/`, and `solution/` responsibility.
- `scripts/`: two end-to-end optimization entry points, one for resident Shapes
  01-13 and one for streamed Shape 14.
- `environments/`: machine-specific dependencies and environment activation.
- `docs/`: competition rules, engineering notes, and deliverables.
- `observations/`: persistent search memory, compact run logs, and benchmark
  summaries.
- `notes/`: concise chronological observations from optimization cycles.

The performance mainline is organized by responsibility:

```text
solution/
  config.py                 typed program and launch configuration
  plan_builder.py           ConfigSpec -> validated ExecutionPlan
  plan.py                   immutable execution plan and expected trace
  transformer.py            model that executes the plan
  operators/                PyTorch, SDPA, and compiled operator compositions
  kernels/                  handwritten Triton candidates by subsystem
  runtimes/                 resident compile and CUDA Graph wrappers
  shape14/                  streamed planning, execution, and attention kernel

autotune/
  search_space.py           generated structures and parameter domains
  search_engine.py          one-shape staged search
  search_sweep.py           one sweep across a shape group
  cross_shape_warmstart.py  task-similarity warm starts from prior Shapes
  optimization_loop.py      repeated sweeps and convergence stopping
  run_log.py                compact run timeline and deployment decisions
  evaluation.py             trial and promotion measurements
  optuna_backend.py         Optuna TPE adapter
  study_storage.py          local study identity and SQLite location

benchmarking/
  config_resolution.py      explicit/deployed configuration selection
  device_isolation.py       one GPU lease and fresh-process execution
  suite.py                  ordered, fresh-process Shape measurement
  measure.py                public resident/Shape-14 measurement dispatcher
  measurement_core.py       shared result types and measurement primitives
  resident_measure.py       Shapes 01-13 measurement and profiling
  shape14_measure.py        streamed Shape-14 measurement
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
  search/resident/search.sqlite3
  search/resident/logs/<run-id>.jsonl
  search/shape14/search.sqlite3
  search/shape14/logs/<run-id>.jsonl
```

`summary.json` is updated after every completed Shape and contains the ordered
benchmark results plus the resident geometric-mean speedup. Search persistence
is intentionally layered:

- `search.sqlite3` is the detailed evidence store. It retains complete Screen
  Trials and reusable Enhanced measurements, including program configurations,
  constraints, timing, memory, execution signatures, failures, and evidence
  identities.
- Each compact JSONL run log is the readable decision timeline. It records the
  run ID, stage durations and budget overruns, structure coverage, compact
  incumbent and challenger programs for Formal comparison, paired ratios, and
  the explicit deployment outcome without duplicating every Trial.
- `deployment/deployed_configs.json` is only the current exact-environment
  deployment table consumed by `solution/`; it is not deployment history.

These generated observation files are not committed. The deployment table is
committed so the measured current winners remain available to the runtime.

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

The two optimization scripts run the complete search-to-deployment flow in the
foreground. GPU measurements are serial: Shapes do not compete for the device.
The resident script targets Shapes 01-05 and 07-13 by default; pass
`-IncludeShape06` only when a new Shape-06 mechanism justifies its much higher
measurement cost.
For Shapes 01-13, the time and trial limits apply to each Shape in each sweep.
For Shape 14, one script invocation performs one bounded streamed search and at
most one Formal challenger comparison. Time limits are soft because an
already-started measurement is allowed to finish.

Resident search gives every generated structure one Screen witness, then spends
most of the remaining fixed Trial budget on the largest ranked survivor set
that can cross TPE's startup threshold and receive guided proposals. About 10%
is reserved for least-sampled alternatives so one noisy witness does not lock
out a structure. Shape 14 instead
enumerates its small, high-value finite space without replacement; the portable
Reference implementation remains a fallback, not a performance challenger. The
resident workflow additionally uses deployment-based early stopping across
complete sweeps. Re-running a script resumes the studies in
`observations/search/resident/` or `observations/search/shape14/`; it does not
start the search from zero.

The resident defaults are 180 seconds and at most 96 new trials per Shape per
sweep, with at most four sweeps and a three-sweep deployment plateau stop.
Shape 14 defaults to one 900-second soft search budget and at most 12 new Screen
trials. Its sequential Formal comparison adds roughly 5.5-10.5 minutes on the
validated RTX 4080, so a normal invocation is expected to finish in about
20-27 minutes. This is an operating target rather than a hard timeout: an
already-started GPU measurement is allowed to finish. Re-running the script
continues the persistent Shape-14 studies and creates the next deployment
opportunity instead of restarting the search.

```powershell
# Run the complete resident optimization flow
.\scripts\optimize_shapes_01_13.ps1

# Run the complete Shape-14 optimization flow
.\scripts\optimize_shape_14.ps1

# Override a budget when needed (all other parameters keep their defaults)
.\scripts\optimize_shapes_01_13.ps1 -BudgetSecondsPerShape 240 -MaxIterations 6
.\scripts\optimize_shape_14.ps1 -BudgetSeconds 1200 -MaxNewTrials 12

# Rotate the generated structure sample on a later outer optimization cycle
.\scripts\optimize_shapes_01_13.ps1 -Seed 1235 -StructureSeed 1235

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
