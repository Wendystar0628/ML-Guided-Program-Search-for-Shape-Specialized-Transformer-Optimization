# AI-Assisted Shape-Aware Transformer Kernel Optimizer

This project optimizes the end-to-end CUDA latency of a supplied PyTorch
Transformer while preserving its constructor, forward interface, weights,
output shape, and numerical behavior. The implementation is split into two
clear layers:

- the hardware-neutral optimizer and measurement system in `solution/` and
  `runner/`; and
- exact routes and reproducible evidence for GPUs that have actually been
  measured in `verified_hardware/`.

The split is intentional. A route measured on one GPU is useful evidence for
that GPU, not a portable performance claim for every CUDA device.

## Start here

The main implementation entry is
[`solution/transformer.py`](solution/transformer.py). The complete cross-device
workflow is:

1. probe the target GPU and software stack;
2. analyze each workload shape with the white-box hardware cost model;
3. Smoke-screen at most three eligible candidates per workload with the full
   correctness comparator and end-to-end timer;
4. formally remeasure only the dynamically selected controls and finalist;
5. apply the correctness, observed-execution, incumbent, gain, and shared-route
   gates before publishing every accepted route atomically; and
6. bind the published routes to the measured Workload and Solution version in
   a small verified-bundle manifest;
7. use deterministic dispatch at runtime, with `auto` as the safe fallback.

The workload and optimization code stay shared. Only an exact, measured route
table and its evidence belong to a device package.

The currently verified device is the
[`NVIDIA GeForce RTX 4080`](verified_hardware/nvidia_geforce_rtx_4080/README.md).
Its profile, manifest, eight exact routes, formal nine-case result, and
one-command reproduction entry are kept together there.

## Architecture boundaries

The control plane has a small set of explicit sources of truth:

- [`solution/policies.py`](solution/policies.py) defines every concrete
  `PolicySpec`, while [`runner/candidates.py`](runner/candidates.py) defines
  every measurable `CandidateSpec`, including applicability, capabilities,
  deployment eligibility, and the execution evidence required to count it;
- [`solution/execution_plan.py`](solution/execution_plan.py) resolves runtime
  facts and policy intent into one immutable `ExecutionPlan` with per-layer
  plans. Forward consumes it for QKV layout, mask, attention, FFN, residual,
  and graph decisions; the correctness run
  also records the branches that actually execute if a final tensor guard
  falls back. Path reporting is read-only and includes that observed evidence;
- [`runner/result_contracts.py`](runner/result_contracts.py) validates typed
  worker requests, correctness summaries, timings, and benchmark responses at
  the process boundary, while persisted files remain compact JSON;
- [`runner/routing_contracts.py`](runner/routing_contracts.py) owns
  `HardwareIdentity`, exact route-key construction, and route-sharing groups;
  and
- [`runner/calibration.py`](runner/calibration.py) exposes the complete
  cold-start workflow as a reusable `CalibrationService`. The CLI presents the
  service today, and a future project-specific Agent can call the same service
  and consume its structured progress events without parsing terminal text.

The CUDA kernels remain independent implementation modules. These control-plane
contracts select and verify them without adding a plugin framework or moving
benchmark logic into `forward`.

## Environment

The checked Windows development environment uses Python 3.12, PyTorch with
CUDA, and Triton for Windows:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
. .\activate_dev_env.ps1
python environment_check.py
```

`activate_dev_env.ps1` configures the local MSVC, CUDA, Triton, TorchInductor,
and CUDA-extension paths. `environment_check.py` exercises PyTorch CUDA,
Triton JIT, `torch.compile`, and a small CUDA extension. Device-specific
versions are recorded in the corresponding verified-hardware profile rather
than treated as repository-wide requirements.

## Quick start on any supported GPU

Run the full diagnostic probe, including the optional SDPA backend checks:

```powershell
python -m runner probe --device cuda:0
```

Inspect the cross-hardware candidate plan. This runs one routing probe with
small launch, graph, copy, GEMM, and softmax anchors, but no full Transformer
candidate benchmark:

```powershell
python -m runner calibrate --plan-only --device cuda:0
```

Run a bounded, non-deploying Smoke screen on the target device. The default
candidate limit is three per Workload:

```powershell
python -m runner calibrate --preset smoke --device cuda:0
```

Run the complete formal calibration to deploy measured routes. This one command
performs its own Probe, deployable Smoke Top-3 screen, dynamic Formal finalist
remeasurement, and promotion gates. A new device formally compares at most
`eager-auto` and its best Smoke challenger. A device with a specialized current
route also retains that incumbent, so Formal measures at most three distinct
candidates. Runner then locates or creates the exact verified-device package
and publishes all accepted routes with one atomic `routes.json` update. A
companion `manifest.json` binds that table to the exact Workload definition,
Solution implementation, Formal protocol, and compact Summary/Case IDs:

```powershell
python -m runner calibrate --preset formal --device cuda:0
```

Run one case or the complete core workload:

```powershell
python -m runner benchmark --preset smoke --case-id balanced_s128_fp16
python -m runner benchmark --preset smoke
```

Measure an explicit candidate set when iterating on one mechanism. `tune`
requires at least one `--candidate`; automatic candidate planning and ranking
belong to `calibrate`:

```powershell
python -m runner tune --case-id mask_s512_padding_fp16 `
  --candidate eager-auto --candidate padding-fused `
  --candidate padding-packed --preset smoke
```

Use `tune` only for focused, non-deploying experiments over the candidates in
the command. It records measurements but never deploys a route, even with the
formal preset:

```powershell
python -m runner tune --case-id launch_s64_fp16 `
  --candidate eager-auto --candidate launch-cudagraph --preset formal
```

Manual promotion is retained only to replay a compatible historical formal
summary or recover from an interrupted deployment:

```powershell
python -m runner promote `
  --route-table verified_hardware/<device_id>/routes.json `
  --tuning-id <tuning-id>
```

Run a complete formal sweep or profile one representative case:

```powershell
python -m runner benchmark --preset formal
python -m runner profile --case-id attention_s2048_fp16
```

The CLI defaults to `--target solution`, `--solution-policy dispatch`,
`--workload-set transformer_core_v1`, and `--device cuda:0`. For `calibrate`,
`--candidate-limit` defaults to three and limits the Smoke screening pool; the
smaller Formal set is selected automatically rather than controlled by a
second user-facing limit. Use
`python -m runner <command> --help` for policies, compile modes, TF32 controls,
timeouts, result directories, and baseline-only diagnostics.

For a known device, prefer its checked launcher. For example:

```powershell
python verified_hardware/nvidia_geforce_rtx_4080/run_verified.py --preset smoke
```

The launcher validates the device profile and package, runs the shared Runner
through the normal verified-device catalog, confirms that every result came
from the sibling route table, and writes generated measurements below the same
device package.

## Cross-hardware design

[`runner/hardware_router.py`](runner/hardware_router.py) combines two inputs:

- a static workload analysis covering shape, dtype, attention size, GEMM work,
  launch pressure, mask form, and memory pressure; and
- a probed hardware profile covering GPU identity, compute capability, memory,
  cache, bandwidth-related properties, software versions, and short performance
  anchors.

The result is a bounded candidate order with explicit eligibility and reasons.
It is a white-box ordering prior, not a learned latency predictor or a
deployable route. It does not use a decorative confidence score, benchmark
inside `forward`, or copy a winner from an unmeasured GPU.

The timing is explicit:

1. load and validate the selected Workload definitions;
2. run one routing probe for the whole command;
3. combine the shared hardware profile with each Workload shape to produce a
   deployable Smoke plan of at most three candidates, retaining `eager-auto`
   and any exact current incumbent;
4. run the complete Transformer comparator and short paired end-to-end timing
   for that bounded pool;
5. select the best valid new challenger from measured Smoke results;
6. formally remeasure `eager-auto`, the challenger, and any distinct
   specialized incumbent; and
7. use formal measurements, never probe predictions or Smoke rankings alone,
   to atomically update the matching verified-device route table.

The routing probe deliberately skips the broader SDPA backend diagnostic
because that output is not consumed by the current cost model. The standalone
`probe` command keeps diagnostic mode for investigations. `tune` requires an
explicit `--candidate` list and skips the routing probe because the user has
already chosen the candidates, but it still runs the complete correctness and
performance measurement. Automatic candidate ordering is available only
through `calibrate`.

Candidate measurement remains authoritative. `calibrate` and `tune` reuse the
same fresh-worker comparator and full-forward timing protocol as `benchmark`.
The selected `solution_policy` is carried explicitly in the worker request;
Runner does not rely on a parent-process environment variable to identify the
candidate under test.
Only a complete formal `calibrate` run deploys. Smoke calibration only exposes
the quick screening results; plan-only calibration stops after the white-box
plan; and `tune` remains a focused, non-deploying experiment even with a Formal
preset. Workloads that share one runtime route key are screened and formally
decided together, so one incompatible per-case winner cannot overwrite the
other. After all fail-closed implementation, execution, correctness,
conservative-gain, incumbent, and shared-route gates pass, Runner resolves or
creates the verified package and publishes all route changes together. The
manual `promote` command is retained only for compatible history replay or
deployment recovery and applies the same gates.

At inference time, [`solution/dispatch.py`](solution/dispatch.py) loads the
verified device-route catalog once, accepts only bundles whose manifest still
matches the route table, Workload definition, and current Solution
implementation, then matches the exact hardware/software/shape key. A missing,
invalid, or stale bundle is skipped closed and the unmatched input uses `auto`.
There is no online benchmarking and no scan of historical result files in
`forward`. A verified route therefore needs only cheap identity matching during
ordinary execution; performance anchors belong only to cold-start calibration
on an unverified or changed device.

## Core workload

[`runner/workloads/transformer_core_v1.json`](runner/workloads/transformer_core_v1.json)
is the single machine-readable definition of nine core cases. Shapes and
scoring roles remain fixed across devices:

| Performance group | Cases | Main pressure |
|---|---|---|
| Launch / Graph | `launch_s64_fp16` | Small-shape launch and framework overhead |
| Balanced / Precision | `balanced_s128_fp32`, `balanced_s128_fp16` | Shared shape across FP32 and FP16 paths |
| Long Attention | `attention_s2048_fp16`, `attention_s2048_causal_fp16` | Long-context attention and causal behavior |
| Padding / Mask | `mask_s512_full_fp16`, `mask_s512_padding_fp16`, `mask_s512_causal_padding_fp16` | Full, padded, and combined causal-padding masks |
| Wide GEMM / FFN | `wide_s256_bf16` | Wide projections, FFN throughput, and BF16 execution |

The set is deliberately compact rather than a Cartesian product. Its five
groups receive equal weight, so the three mask cases do not dominate the
project-level metric.

## Optimization mainline

The shared Solution keeps a conservative path while exposing bounded,
correctness-gated candidates. Its current mechanisms include packed QKV
projection, zero-copy head layouts, shared mask preprocessing, native SDPA in a
validated region, Triton scale/mask/layout kernels, long-sequence streaming
attention, exact in-place GELU reuse, padding-aware execution, fixed-shape CUDA
Graph replay, and controlled `torch.compile` screening.

The dispatcher only selects a specialized policy when the complete Transformer
comparator and end-to-end measurement support that exact route. Unsupported or
numerically incompatible inputs remain on `auto`. Approximate activation,
all-layer streaming attention, experimental fused-PV, and whole-model compile
paths stay outside deployed dispatch where the comparator or measured benefit
does not justify them.

## Measurement and results

Every benchmark case runs in a fresh worker process. The worker:

1. validates one typed request, including the explicit Solution policy when a
   candidate is being measured;
2. loads the requested target and an independent baseline;
3. copies identical weights and derives packed tensors before device transfer;
4. runs the complete correctness comparator;
5. measures alternating baseline and target rounds with the full-forward CUDA
   timer; and
6. returns one validated, compact schema-v2 result to the parent.

The default development paths are:

```text
results/runs/<run_id>.json
results/tuning/<tuning_id>.json
```

A benchmark result retains the workload and protocol, compact hardware/runtime
environment, correctness aggregate, baseline and target median/P90 latency,
round medians, sample count, speedup, selected execution path, route source and
hash, and implementation hashes. Raw latency samples and duplicate summary
fields are intentionally omitted.

Generated root-level results are ignored by Git. A verified device package may
track one compact reference summary and ignores its generated runs; see
[`verified_hardware/README.md`](verified_hardware/README.md) for the small
package contract.

A complete sweep reports each case, every group geometric mean, the equal-weight
group-balanced geometric mean, and the worst case. The aggregate is marked
complete only when every expected case succeeds, passes correctness, and
produces a finite positive result. Failed or missing cases remain visible but
cannot produce a partial project score.

## Official compatibility entry

The supplied benchmark is preserved byte-for-byte at
[`official/torch_transformer_benchmark.py`](official/torch_transformer_benchmark.py),
with its checksum in [`official/snapshot.json`](official/snapshot.json).

The root [`torch_transformer_benchmark.py`](torch_transformer_benchmark.py) is
a thin compatibility entry that loads `UserOptimizedTransformer` and the
optional `copy_model_weights` hook, then delegates argument parsing,
correctness, and timing to the supplied benchmark:

```powershell
python torch_transformer_benchmark.py --device cuda:0 --dtype float32
```

Use this entry for direct single-configuration compatibility checks. Use
`python -m runner` for repeatable sweeps, probes, profiles, and route tuning.

## Tests

```powershell
python -m pytest -q
.\.venv\Scripts\python.exe -m pytest -q -m architecture tests\architecture
.\.venv\Scripts\python.exe -m pytest -q -m gpu tests\gpu
python -m ruff check runner solution tests verified_hardware `
  torch_transformer_benchmark.py environment_check.py
```

The default suite covers the immutable benchmark snapshot, workload contract,
policy/candidate registry consistency, execution planning and mask fallbacks,
typed worker boundaries, timing order, isolated result persistence,
aggregation, hardware-profile extraction, hardware-cost-model routing, exact
dispatch, stale-bundle fallback, and the compatibility entry. Source-code
comments are written in English so the implementation remains easy to review.
The focused `architecture` suite protects registry and module boundaries. The
focused `gpu` suite runs real CUDA kernel/policy smoke checks with the official
Comparator and CUDA Events; it skips cleanly when CUDA is unavailable. These
smoke checks establish execution and correctness, not a fixed performance
threshold or a substitute for Formal calibration.

## Repository map

```text
official/                    Immutable supplied benchmark snapshot
solution/                    Shared optimized Transformer and kernel candidates
solution/policies.py         Single registry of concrete runtime policies
solution/execution_plan.py   Pure runtime eligibility and execution-plan resolver
runner/                      Probe, analysis, benchmark, profile, tune, and promotion
runner/candidates.py         Candidate applicability, capability, and evidence registry
runner/calibration.py        Reusable cold-start service for CLI and future Agent callers
runner/result_contracts.py   Typed worker and compact benchmark-result boundaries
runner/routing_contracts.py  Shared hardware identity and exact route-key adapters
runner/verified_hardware.py  Shared verified-device validation and run orchestration
runner/workloads/            Hardware-neutral transformer_core_v1 workload
verified_hardware/           Measured profiles, exact routes, manifests, and evidence
tests/architecture/          Fast structural and registry contract guards
tests/gpu/                   Real-CUDA policy and kernel smoke checks
tests/                       Remaining correctness and runner regressions
results/                     Generated cross-device development results, ignored
environment/                 Local Windows runtime compatibility hook
```

Performance work stays centered on the shared `solution/` mainline. A complete
formal calibration on a new GPU uses the common probe, planner, comparator, and
benchmark, then automatically creates its small verified package and publishes
only the exact routes that passed every gate.
