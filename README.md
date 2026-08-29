# Shape-Aware Transformer Program Search

The project generates valid Transformer execution configurations, measures them
on the target GPU, and deploys the fastest configuration that passes the
official comparator. The performance path is intentionally direct:

```text
ConfigSpec -> ConfigCompiler -> ExecutionPlan -> GPU measurement -> TPE search
           -> deployments/deployed_configs.json
```

The 14 workloads come from [`official/test_shapes.json`](official/test_shapes.json).
Shapes 1-13 use resident execution; Shape 14 uses streamed microbatches because
the logical batch cannot reside in memory at once.

## Main files

- `solution/`: runtime configuration, compiler, Transformer, and kernels.
- `optimizer/`: generated search space and branch-local constrained TPE.
- `runner/`: direct measurement, profiling, hardware probe, and CLI.
- `official/`: upstream benchmark semantics and workload shapes.
- `deployments/deployed_configs.json`: measured per-device winners.
- `results/intermediate/`: disposable runs, profiles, and Optuna studies.

## Commands

```powershell
# Inspect the GPU environment
.\.venv\Scripts\python.exe -m runner probe --device cuda:0

# Search one or more official shapes
.\.venv\Scripts\python.exe -m runner search `
  --case-id official_01 `
  --device cuda:0 `
  --budget-seconds 900

# Measure one explicit ConfigSpec
.\.venv\Scripts\python.exe -m runner run `
  --case-id official_01 `
  --config path\to\config.json `
  --preset smoke `
  --device cuda:0

# Profile one explicit ConfigSpec
.\.venv\Scripts\python.exe -m runner profile `
  --case-id official_01 `
  --config path\to\config.json `
  --device cuda:0
```

`search` stores one resumable SQLite study database under
`results/intermediate/search/`. This architecture-only revision did not run GPU
search or benchmarks, so it makes no new performance claim.
