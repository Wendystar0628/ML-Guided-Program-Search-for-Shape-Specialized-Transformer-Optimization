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
3. screen a small eligible candidate set with the full correctness comparator
   and end-to-end timer;
4. promote a stable winner into an exact device route; and
5. use deterministic dispatch at runtime, with `auto` as the safe fallback.

The workload and optimization code stay shared. Only an exact, measured route
table and its evidence belong to a device package.

The currently verified device is the
[`NVIDIA GeForce RTX 4080`](verified_hardware/nvidia_geforce_rtx_4080/README.md).
Its profile, eight exact routes, formal nine-case result, and one-command
reproduction entry are kept together there.

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

Probe the default GPU in a fresh worker process:

```powershell
python -m runner probe --device cuda:0
```

Inspect the cross-hardware candidate plan without running candidates:

```powershell
python -m runner calibrate --plan-only --device cuda:0
```

Run bounded candidate calibration on the target device:

```powershell
python -m runner calibrate --preset smoke --device cuda:0
```

Run one case or the complete core workload:

```powershell
python -m runner benchmark --preset smoke --case-id balanced_s128_fp16
python -m runner benchmark --preset smoke
```

Screen selected candidates when iterating on one mechanism:

```powershell
python -m runner tune --case-id mask_s512_padding_fp16 `
  --candidate eager-auto --candidate padding-fused `
  --candidate padding-packed --preset smoke
```

Repeat finalists with the formal protocol, then promote the measured winner to
the intended device package explicitly:

```powershell
python -m runner tune --case-id launch_s64_fp16 `
  --candidate eager-auto --candidate launch-cudagraph --preset formal
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
`--workload-set transformer_core_v1`, and `--device cuda:0`. Use
`python -m runner <command> --help` for policies, compile modes, TF32 controls,
timeouts, result directories, and baseline-only diagnostics.

For a known device, prefer its checked launcher. For example:

```powershell
python verified_hardware/nvidia_geforce_rtx_4080/run_verified.py --preset smoke
```

The launcher validates the device profile, activates that package's route
table, runs the shared Runner, and writes generated measurements below the same
device package.

## Cross-hardware design

[`runner/hardware_router.py`](runner/hardware_router.py) combines two inputs:

- a static workload analysis covering shape, dtype, attention size, GEMM work,
  launch pressure, mask form, and memory pressure; and
- a probed hardware profile covering GPU identity, compute capability, memory,
  cache, bandwidth-related properties, software versions, and short performance
  anchors.

The result is a bounded candidate order with explicit eligibility and reasons.
It is a white-box ordering prior, not a learned latency predictor. It does not
use a decorative confidence score, benchmark inside `forward`, or copy a
winner from an unmeasured GPU.

Candidate measurement remains authoritative. `calibrate` and `tune` reuse the
same fresh-worker comparator and full-forward timing protocol as `benchmark`;
neither silently deploys a winner. `promote` is separate and fails closed
unless a complete formal result still matches the implementation, the selected
policy really executed, and the specialized route has a sufficient conservative
gain over `auto` and any incumbent route.

At inference time, [`solution/dispatch.py`](solution/dispatch.py) loads the
verified device-route catalog once, matches the exact hardware/software/shape
key, and otherwise selects `auto`. There is no online benchmarking and no scan
of historical result files in `forward`.

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

1. loads the requested target and an independent baseline;
2. copies identical weights and derives packed tensors before device transfer;
3. runs the complete correctness comparator;
4. measures alternating baseline and target rounds with the full-forward CUDA
   timer; and
5. returns one compact schema-v2 result to the parent.

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
python -m ruff check runner solution tests verified_hardware `
  torch_transformer_benchmark.py environment_check.py
```

The tests cover the immutable benchmark snapshot, workload contract, weight
packing, masks, timing order, isolated result persistence, aggregation,
hardware-profile extraction, hardware-cost-model routing, exact dispatch and
fallback, device-package loading, and the compatibility entry. Source-code
comments are written in English so the implementation remains easy to review.

## Repository map

```text
official/                    Immutable supplied benchmark snapshot
solution/                    Shared optimized Transformer and kernel candidates
runner/                      Probe, analysis, benchmark, profile, tune, and promotion
runner/verified_hardware.py  Shared verified-device validation and run orchestration
runner/workloads/            Hardware-neutral transformer_core_v1 workload
verified_hardware/           Measured device profiles, exact routes, and evidence
tests/                       Correctness and runner regressions
results/                     Generated cross-device development results, ignored
environment/                 Local Windows runtime compatibility hook
```

Performance work stays centered on the shared `solution/` mainline. A new GPU
first uses the common probe, planner, comparator, and benchmark, then receives a
small verified package only when its exact routes have been measured.
