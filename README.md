# AI-Assisted Shape-Aware Transformer Kernel Optimizer

This project reduces the end-to-end CUDA latency of the supplied PyTorch
Transformer while preserving its constructor, weights, forward interface,
output shape, and numerical behavior. It targets the official test shapes with
a small set of explainable execution policies, then measures the candidates on
the actual GPU and records an exact route for each validated hardware stack.

The competition path is deliberately direct:

1. load the official Transformer and test shapes;
2. run the reference and optimized model with identical weights and inputs;
3. reject candidates that fail `rtol=0.02` and `atol=0.002` correctness;
4. compare complete Transformer forwards with CUDA Events;
5. deploy only a formally remeasured winner for the exact GPU and shape; and
6. fall back to the portable `auto` policy when no current route matches.

## Official workload

[`official/test_shapes.json`](official/test_shapes.json) is the single
machine-readable source for the published shapes. The default GPU workload,
`official_transformer_v1`, runs `official_01` through `official_13`:

| View | Cases | Published variation | Main optimization pressure |
| --- | --- | --- | --- |
| Batch sweep | 1–6 | `B=1,4,16,64,128,10000` | launch overhead, occupancy, throughput, peak memory |
| Width sweep | 1, 7, 8 | `D=32,128,1024` | small GEMMs, tensor cores, wide projections and FFN |
| Head sweep | 1, 9–11 | `H=1,2,4,16` | head dimension, attention backend, layout and occupancy |
| Sequence sweep | 1, 12, 13 | `S=32,128,1024` | launch overhead, causal attention, softmax and quadratic intermediates |

All published cases are causal and use `FFN Dim = QKV Dim`. The appendix also
contains `official_14` (`B=32, D=1024, H=16, S=100000, L=2`); it remains in the
official shape contract but is excluded from the current default GPU sweep.

The appendix does not define a dtype. Dtype, warm-up, repeat count, timing
rounds, TF32, and random seeds therefore belong to the measurement protocol,
not to duplicated shape definitions.

## Optimization policies

[`solution/policies.py`](solution/policies.py) is the single registry of
runtime policies:

| Policy | Purpose |
| --- | --- |
| `safe` | Conservative official-equivalent path used for diagnosis and fallback comparison |
| `auto` | Hardware-neutral optimized plan and fallback for an unmatched device/shape |
| `causal-sdpa` | Native causal SDPA without materializing a full causal mask |
| `graph` | Full-forward CUDA Graph capture and replay for fixed, launch-bound shapes |
| `batch-tiled` | Batch chunking for shapes whose full intermediates create excessive peak memory |
| `inplace-block` | Reuse selected block intermediates to reduce allocation and memory traffic |

These names describe implemented execution behavior. `inplace-block` is not
presented as a fully fused Transformer block. A custom online/streaming
attention kernel is a possible future long-sequence candidate, not a current
policy.

[`solution/execution_plan.py`](solution/execution_plan.py) resolves one
immutable execution plan from the requested policy, input shape, dtype, device,
and available backend. The forward pass consumes that plan; path reporting is
read-only and records the branches that actually execute. A specialized policy
that falls back cannot be counted as a successful candidate.

## Repository map

```text
official/                    Supplied benchmark, published shapes, snapshot identity
solution/                    Optimized Transformer, policies, execution plan, kernels
runner/                      GPU measurement, profiling, tuning, calibration, routing
verified_hardware/           Exact routes and compact results for measured GPUs
tests/                       Control-plane, architecture, correctness, and real-GPU smoke
results/                     Generated local runs; ignored by Git
docs/                        Official rules and development design
torch_transformer_benchmark.py
                             Thin official-compatible entry
```

The performance implementation remains in `solution/`. Hardware packages do
not copy kernels, Transformer code, or benchmark orchestration. Source-code
comments are written in English for straightforward public review.

## Environment

The local validated path uses Python 3.12 and a CUDA-enabled PyTorch build.
Create an isolated environment and install the repository dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-windows-rtx4080.txt
. .\activate_windows_rtx4080.ps1
python environment_check.py --check-extension
```

The RTX 4080 activation script selects the checked local MSVC and CUDA
toolchains. On another GPU or operating system, install a PyTorch/Triton build
supported by that CUDA stack, install `requirements.txt`, and run:

```powershell
python environment_check.py --device cuda:0
```

Use `--check-extension` only when a local CUDA toolkit and compatible host
compiler are available. Performance conclusions require a CUDA device; CPU
runs are useful only for control-plane diagnostics.

## Quick reproduction

Run the official-compatible single-shape entry:

```powershell
python torch_transformer_benchmark.py --device cuda:0 --dtype float32
```

Run the default official 1–13 smoke or formal sweep:

```powershell
python -m runner benchmark --preset smoke --device cuda:0
python -m runner benchmark --preset formal --device cuda:0
```

Run one published case:

```powershell
python -m runner benchmark --case-id official_02 --preset smoke --device cuda:0
```

Profile one complete Transformer forward:

```powershell
python -m runner profile --case-id official_13 --device cuda:0
```

Measure an explicit, non-deploying candidate set:

```powershell
python -m runner tune --case-id official_02 `
  --candidate eager-auto --candidate graph `
  --preset smoke --device cuda:0
```

`tune` has no hidden default candidate set and never deploys a route. Use it
for focused experiments when the candidate list is already known.

## Cross-hardware calibration

Inspect the target GPU and runtime:

```powershell
python -m runner probe --device cuda:0
```

Generate a white-box routing plan without running full Transformer candidates:

```powershell
python -m runner calibrate --plan-only --device cuda:0
```

Run a bounded smoke screen. The automatic planner measures at most three
eligible candidates per case and does not publish routes:

```powershell
python -m runner calibrate --preset smoke --device cuda:0
```

Run the complete calibration and publish validated exact routes:

```powershell
python -m runner calibrate --preset formal --device cuda:0
```

The formal command performs one probe, a bounded smoke screen, dynamic finalist
selection, formal remeasurement, correctness and observed-execution checks,
and an atomic verified-hardware bundle update. Formal finalists are the
necessary distinct members of `auto`, the best new smoke challenger, and an
existing specialized incumbent; they are not a second fixed Top-N list.

At runtime the dispatcher matches the exact GPU, software stack, dtype, and
published Transformer shape. It never benchmarks inside `forward`, scans old
result files, or copies a winner from an unmeasured GPU.

## Measurement protocol

Each case runs in a fresh worker process. The worker:

1. validates a typed request and performs a peak-memory resource check;
2. constructs an independent baseline and solution;
3. copies identical weights and creates identical inputs;
4. runs the official correctness comparator;
5. alternates baseline and solution timing rounds with CUDA Events; and
6. returns a compact validated result to the parent process.

The result keeps only information useful for optimization and review:

- shape and measurement protocol;
- compact GPU/runtime identity;
- correctness summary;
- baseline and solution median/P90 latency;
- round medians, sample count, and speedup;
- requested, resolved, and observed policy;
- route source and implementation identities.

Raw timing samples, duplicated summaries, full profiler traces, and historical
development results are not stored in the repository.

Generated results use these paths:

```text
results/sweeps/<sweep_id>/summary.json
results/sweeps/<sweep_id>/runs/<run_id>.json
results/tuning/<tuning_id>/summary.json
results/tuning/<tuning_id>/runs/<run_id>.json
results/probes/<run_id>.json
results/profiles/<run_id>.json
results/runs/<run_id>.json
```

## Verified GPU results

Each measured GPU has one small package below
[`verified_hardware/`](verified_hardware/README.md). The package contains its
hardware profile, exact route table, manifest, compact formal summary, and a
thin reproduction script. The manifest binds routes to the official benchmark,
published shapes, current solution, measurement protocol, and runtime stack.

The local reference platform is documented at
[`verified_hardware/nvidia_geforce_rtx_4080/`](verified_hardware/nvidia_geforce_rtx_4080/README.md).
Use its checked entry after a formal package has been generated:

```powershell
python verified_hardware/nvidia_geforce_rtx_4080/run_verified.py --preset smoke
```

The formal FP32 sweep completed all 13 default shapes on the RTX 4080:

- `13/13` successful cases;
- `4.1827x` geometric-mean speedup;
- zero failed output elements across five accuracy trials per case;
- maximum observed absolute error `0.00134444`, below `atol=0.002`;
- per-case speedups from `1.1219x` to `11.0236x`.

The full rounded table is in the
[`RTX 4080 package README`](verified_hardware/nvidia_geforce_rtx_4080/README.md),
and [`results/reference_formal.json`](verified_hardware/nvidia_geforce_rtx_4080/results/reference_formal.json)
is the authoritative machine-readable result. These numbers apply to the
recorded RTX 4080 and software stack; they are not portable speed claims for
every CUDA device.

## Tests

```powershell
python -m pytest -q
.\.venv\Scripts\python.exe -m pytest -q -m architecture tests\architecture
.\.venv\Scripts\python.exe -m pytest -q -m gpu tests\gpu
python -m ruff check runner solution tests verified_hardware `
  torch_transformer_benchmark.py environment_check.py
```

The real-GPU smoke layer confirms that CUDA policies compile and execute, the
official comparator passes, the requested policy is observed, and CUDA Event
timings are finite. It does not replace a formal end-to-end sweep or impose a
fragile fixed performance threshold.

## AI-assisted development

AI assistance is used to inspect the supplied Transformer, derive hardware and
shape hypotheses, generate bounded implementation candidates, review code,
analyze profiler output, and compare measured iterations. Deterministic code
remains responsible for correctness, timing, resource checks, route promotion,
and result persistence. The planned project-specific CLI Agent will reuse the
same runner rather than entering the model hot path.

## Limitations and next steps

- The current default performance scope is `official_01` through
  `official_13`; `official_14` remains part of the published shape contract but
  is not run by the default sweep.
- Routes are deliberately exact. A changed GPU, driver, CUDA/PyTorch/Triton
  stack, official snapshot, or solution source requires new calibration.
- Hardware probing provides an explainable candidate ordering, not a learned
  latency predictor; full-workload measurement remains authoritative.
- A custom online/streaming causal attention kernel is the main future option
  when native SDPA leaves significant long-sequence headroom.
- Further work can explore library epilogues, local compiler boundaries, and
  additional buffer reuse without replacing efficient library GEMMs blindly.

## Public submission

The competition deliverable requires a public GitHub repository and a public
YouTube demo linked from Devpost. The development repository may remain private,
but it must be made public before submission after checking that no API keys,
credentials, virtual environments, caches, local build products, raw profiler
traces, or private user files are tracked.

## Team member contributions

`[Add verified team member names and contributions before submission. If this
is an individual entry, state that explicitly.]`
