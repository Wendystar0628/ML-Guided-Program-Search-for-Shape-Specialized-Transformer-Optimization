# AI-Assisted Shape-Aware Transformer Kernel Optimizer

This repository optimizes the supplied PyTorch Transformer on GPU while
preserving its constructor, weights, forward interface, output shape, and
numerical tolerance. The core idea is simple: describe each workload shape,
rank a small set of eligible execution compositions from hardware signals,
measure the complete Transformer on the target GPU, and deploy the measured
winner for that exact hardware and software stack.

The current competition workload is split by execution reality:

- `official_01` through `official_13` use the resident paired benchmark against
  the supplied baseline.
- `official_14` (`B=32, D=1024, H=16, S=100000, L=2`) uses an independent,
  memory-bounded streamed benchmark and does not enter the resident aggregate.

## Verified result at a glance

The checked RTX 4080 performance record is the only numeric source for this
summary:

- Resident Formal: **13/13 successful**, **8.220830719x** unweighted
  geometric-mean speedup, per-shape speedups from **2.196x** to **28.841x**,
  zero failed output elements, and maximum absolute error **0.00188206**.
- Streamed Shape 14 Formal: **17.136172 s** target median, **17.158589 s**
  target P90, **18.821000 s** host-streamed end-to-end time, **81.1887 useful
  TFLOP/s**, **94.0529% project-estimated MFU**, and **7.307 GiB**
  (**7,845,867,008 bytes**) peak device allocation.
- The streamed artifact selected `mixed-fp16-core-cudnn`, timing microbatch
  size `2`, and microbatch count `16`. These are measured outputs of the joint
  policy/schedule screen, not hard-coded Shape 14 properties.

Authoritative machine-readable records:

- [Unified RTX 4080 performance result](results/final/nvidia_geforce_rtx_4080.json)
- [Exact RTX 4080 routes](verified_hardware/nvidia_geforce_rtx_4080/routes.json)

The unified result keeps the paired Formal rows and geometric mean for Shapes
1-13 together with the separate target-only, provisional Shape 14 row. Every
shape reports latency, useful TFLOP/s, project-estimated MFU, peak allocator
bytes, and a scoped logical operator-traffic estimate. Logical traffic is not a
measured DRAM bandwidth counter.

See [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) for the per-shape table,
measurement interpretation, and optimization insights. The geometric mean and
project-estimated MFU are internal engineering metrics, not claims about an
undisclosed official scoring formula.

## Strategy families

[`policy_registry.py`](policy_registry.py) is the single source of truth for
runtime policies. Policies are compositions of attention backend, linear
compute dtype, residual/LayerNorm implementation, and optional CUDA Graph
replay; they are not unrelated model implementations.

| Family | Registered policies | Role |
| --- | --- | --- |
| Portable | `eager-sdpa` | Hardware-neutral optimized control and unmatched-device fallback |
| Graph | `graph`, `graph-fused-norm` | Remove repeated launch overhead; optionally compile residual + LayerNorm |
| Mixed attention | `mixed-fp16-efficient`, `mixed-fp16-cudnn` | Run selected FP32-model attention in FP16 with an explicit SDPA backend |
| Mixed core | `mixed-fp16-core-efficient`, `mixed-fp16-core-cudnn` | Extend FP16 execution to attention and linear compute for throughput-bound shapes |
| Graph compositions | `graph-mixed-fp16-efficient`, `graph-mixed-fp16-efficient-compiled-norm`, `graph-mixed-fp16-core-efficient-compiled-norm` | Compose full-forward graph replay with mixed attention or mixed core and optional compiled residual-to-normalization boundaries |
| Batch-tiled Graph | `batch-tiled-mixed-fp16-core-efficient-compiled-norm` | Split the exact high-batch resident shape into independent fixed batch tiles and replay one captured full-model graph per tile |
| Compiled forward | `compiled-mixed-fp16-core-efficient` | Compile and cache one full fixed-plan forward for the guarded wide or long resident shapes |
| Custom Triton | `mixed-fp16-core-efficient-triton-norm` | Fuse residual add + LayerNorm for the guarded high-row, width-128 path |

`safe` is an internal, non-routable diagnostic fallback. A specialized policy
is accepted only when its requested backend is observed during execution;
silent fallback cannot be promoted as a successful candidate.

Streamed execution is a workload schedule rather than another policy. It
screens the same eligible policies together with memory-safe microbatch
divisors, then covers Shape 14 as one logical batch without constructing a
dense full-batch reference.

## Architecture flow

1. [`official/test_shapes.json`](official/test_shapes.json) defines the
   published workload once.
2. [`runner/workload_execution.py`](runner/workload_execution.py) separates
   resident and memory-bounded streamed execution.
3. The hardware probe measures launch, transfer, GEMM, and backend capability
   signals; the white-box router ranks a bounded eligible candidate set.
4. [`solution/execution_plan.py`](solution/execution_plan.py) resolves one
   immutable plan from policy, shape, dtype, device, and available backends.
   The plan selects one outer runtime: eager execution, full-forward CUDA
   Graph, batch-tiled CUDA Graph, or fixed-plan compiled forward.
5. Fresh GPU workers run correctness first and then complete-forward CUDA Event
   timing. Observed execution evidence prevents a fallback from masquerading as
   a specialized policy.
6. Formal calibration atomically publishes exact routes and a bound hardware
   bundle. Runtime dispatch performs no benchmarking inside `forward`.

Shape 14 follows the same policy and kernel registry but a separate executor.
It validates a complete `B=1`, long-sequence reference scope, then screens
eligible policy and memory-safe microbatch combinations. The selected schedule
covers the complete logical batch and reports both device target time and
host-streamed end-to-end time; neither the microbatch size nor count is part of
the static workload definition.

Inside the Transformer, residual and normalization boundaries are scheduled as
one pipeline. An attention residual is paired with the same block's `norm2`,
while an FFN residual is paired with the next block's `norm1` or the final
normalization. Compiled and Triton backends can therefore return both the
updated residual stream and its normalized view without changing the supplied
pre-normalization mathematics.

## Contribution boundary

Project-owned work includes:

- shape analysis, workload partitioning, policy composition, immutable
  execution planning, and exact dispatch;
- hardware probing, explainable candidate ranking, real-GPU calibration,
  observed-path checks, and atomic route promotion;
- mixed-core integration, full-forward compilation, batch-tiled Graph replay,
  dynamic Shape 14 scheduling, and guarded compiled/Triton residual-to-norm
  implementations;
- paired/streamed measurement services, correctness checks, compact metrics,
  and verified-hardware artifacts.

The supplied Transformer and comparator define the reference calculation.
PyTorch, ATen/cuBLAS, Efficient SDPA, cuDNN SDPA, CUDA Graph, and
`torch.compile` provide library primitives. Their kernels are not presented as
project-authored innovation; the contribution is the measured, shape-aware
composition around them plus the explicitly identified custom Triton kernel.
No external `flash-attn` package is required by the current verified route.

## Repository map

```text
official/                    Supplied benchmark and published shape source
solution/                    Optimized Transformer, execution plans, kernels
runner/                      Probe, measurement, tuning, calibration, routing
verified_hardware/           Checked GPU profiles, exact routes, manifests
tests/                       Control-plane, architecture, and real-GPU tests
policy_registry.py           Shared policy definitions
route_contracts.py           Route identity and serialization contracts
torch_transformer_benchmark.py
                             Thin official-compatible entry
TECHNICAL_REPORT.md          Results, interpretation, and evaluation narrative
results/final/               One tracked final performance file per hardware ID
results/intermediate/        Generated local experiments; ignored by Git
docs/                        Official material and development design; not runtime
```

Performance code stays in `solution/`; a verified hardware package never
copies kernels or benchmark orchestration. Source comments are in English.

## Environment

The checked RTX 4080 path uses Python 3.12 and the pinned Windows environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-windows-rtx4080.txt
. .\activate_windows_rtx4080.ps1
.\.venv\Scripts\python.exe environment_check.py --check-extension
```

On another supported GPU or operating system, install a CUDA-enabled PyTorch
and Triton stack from `requirements.txt`, then run:

```powershell
python environment_check.py --device cuda:0
```

CPU runs are useful only for control-plane diagnostics; performance conclusions
require a GPU.

## Reproduce the checked RTX 4080 results

Run the exact resident Formal protocol (`official_01` through `official_13`):

```powershell
.\.venv\Scripts\python.exe verified_hardware/nvidia_geforce_rtx_4080/run_verified.py `
  --scope resident --preset formal
```

Run the independent streamed Shape 14 Formal protocol:

```powershell
.\.venv\Scripts\python.exe verified_hardware/nvidia_geforce_rtx_4080/run_verified.py `
  --scope streamed --preset formal
```

Run both scopes sequentially:

```powershell
.\.venv\Scripts\python.exe verified_hardware/nvidia_geforce_rtx_4080/run_verified.py `
  --scope all --preset formal
```

Use `--preset smoke` for a shorter execution-path and correctness check. Formal
resident and streamed execution update their respective sections in the single
hardware result. Shape 14 never enters the resident route table or geometric
mean.

Other useful entries:

```powershell
# Official-compatible single-shape entry
.\.venv\Scripts\python.exe torch_transformer_benchmark.py --device cuda:0 --dtype float32

# Direct resident and streamed development runs
.\.venv\Scripts\python.exe -m runner benchmark --preset smoke --device cuda:0
.\.venv\Scripts\python.exe -m runner benchmark-streamed `
  --case-id official_14 --preset smoke --device cuda:0

# Inspect one complete-forward profile
.\.venv\Scripts\python.exe -m runner profile --case-id official_13 --device cuda:0
```

## Calibrate another hardware stack

Cross-hardware support remains in the project even though only the recorded RTX
4080 stack is currently verified:

```powershell
# Measure routing anchors and inspect the planned candidates
.\.venv\Scripts\python.exe -m runner probe --mode routing --device cuda:0
.\.venv\Scripts\python.exe -m runner calibrate --plan-only --device cuda:0

# Bounded GPU screen; no route publication
.\.venv\Scripts\python.exe -m runner calibrate --preset smoke --device cuda:0

# Formal remeasurement and atomic exact-route publication
.\.venv\Scripts\python.exe -m runner calibrate --preset formal --device cuda:0
```

The theoretical ranking narrows the search; measured full-workload latency and
correctness decide the winner. A changed GPU, driver, CUDA/PyTorch runtime,
official snapshot, or Solution source requires recalibration.

## Validation

```powershell
.\.venv\Scripts\python.exe -m pytest -q -m "not gpu"
.\.venv\Scripts\python.exe -m pytest -q -m architecture tests\architecture
.\.venv\Scripts\python.exe -m pytest -q -m gpu tests\gpu\test_gpu_smoke.py
$env:RUN_SHAPE14_GPU='1'
.\.venv\Scripts\python.exe -m pytest -q -m gpu tests\gpu\test_shape14_gpu.py
.\.venv\Scripts\python.exe -m ruff check runner solution tests verified_hardware `
  route_contracts.py policy_registry.py project_identity.py `
  torch_transformer_benchmark.py environment_check.py
```

## Current limitations

- Shape 14 is a successful but **provisional, target-only** streamed result. It
  has no dense full-batch baseline, no speedup claim, and no exact resident
  route.
- Useful TFLOP/s and project-estimated MFU are engineering interpretations of
  the current operator boundary and measured GEMM anchors. Project MFU is not
  an official score; logical traffic estimates are not measured DRAM traffic.
- Cross-hardware probing and calibration are retained, but only the exact RTX
  4080 hardware/software identity in `verified_hardware/` is currently checked.
- The project-specific autonomous Agent runtime is deliberately deferred. AI
  assistance was used during development, while deterministic code owns
  correctness, timing, routing, and persisted results.
- Library backend availability is platform-dependent. Unsupported specialized
  policies fail eligibility or observed-execution checks and fall back to the
  portable path rather than being reported as successful optimizations.
