# AI-Assisted Shape-Aware Transformer Kernel Optimizer

This project reduces the end-to-end CUDA latency of a supplied PyTorch Transformer while preserving its constructor, forward interface, weights, output shape, and numerical behavior. The primary implementation entry is [`solution/transformer.py`](solution/transformer.py); measurements use the complete Transformer forward pass rather than isolated kernel timings.

The current Solution establishes a stronger performance mainline with four concrete optimizations:

- Q, K, and V weights are packed once by `copy_model_weights`, then evaluated through one fused QKV projection per layer.
- A causal mask is created once per model and shared by all layers instead of being rebuilt during every attention call.
- Token-mask inversion and broadcast views are prepared once per model forward and reused across all layers; masking and score scaling reuse fresh intermediate storage in place.
- The validated short, non-causal CUDA FP32 region uses PyTorch scaled dot-product attention; other regions retain the reference operation and accumulation order as a numerical fallback.

These mechanisms are intentionally bounded by correctness and measured behavior. The repository does not claim that one backend or optimization is best for every shape.

## Target environment

The checked development environment is native Windows with an NVIDIA RTX 4080 (`sm_89`), Python 3.12, PyTorch 2.12.1 + CUDA 13.2, and Triton for Windows 3.7.1.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
. .\activate_dev_env.ps1
python environment_check.py
```

`activate_dev_env.ps1` activates the project environment and configures the local MSVC, CUDA, Triton, TorchInductor, and CUDA extension paths. `environment_check.py` exercises PyTorch CUDA, Triton JIT, `torch.compile`, and a small CUDA extension.

## Quick start

Probe the default GPU in a fresh worker process:

```powershell
python -m runner probe --device cuda:0
```

Run one short development measurement:

```powershell
python -m runner benchmark --preset smoke --case-id balanced_s128_fp16
```

Run the complete nine-case smoke sweep:

```powershell
python -m runner benchmark --preset smoke
```

Run the complete formal sweep with the official accuracy and timing counts:

```powershell
python -m runner benchmark --preset formal
```

Profile one representative case with the same model loader and workload definition:

```powershell
python -m runner profile --case-id attention_s2048_fp16
```

The CLI defaults to `--target solution`, `--workload-set rtx4080_core_v1`, and `--device cuda:0`. Supplying `--case-id` runs one case; omitting it from `benchmark` runs the workload set in its declared order. Use `python -m runner <command> --help` for compile modes, TF32 controls, timeouts, alternative devices, and baseline-only diagnostics.

## Core workload

[`runner/workloads/rtx4080_core_v1.json`](runner/workloads/rtx4080_core_v1.json) is the single machine-readable source for nine RTX 4080 development cases:

| Performance group | Cases | Main pressure |
|---|---|---|
| Launch / Graph | `launch_s64_fp16` | Small-shape launch and framework overhead |
| Balanced / Precision | `balanced_s128_fp32`, `balanced_s128_fp16` | Shared shape across FP32 and FP16 paths |
| Long Attention | `attention_s2048_fp16`, `attention_s2048_causal_fp16` | Long-context attention and causal behavior |
| Padding / Mask | `mask_s512_full_fp16`, `mask_s512_padding_fp16`, `mask_s512_causal_padding_fp16` | Full, padded, and combined causal-padding masks |
| Wide GEMM / FFN | `wide_s256_bf16` | Wide projections, FFN throughput, and BF16 execution |

The set is deliberately compact rather than a Cartesian product. Its five groups have equal weight so that the three mask cases do not dominate the project-level metric.

## Measurement behavior

Before each benchmark worker starts, the parent validates the immutable official snapshot. Every benchmark case then runs in a fresh worker process. The worker:

1. loads the requested target and constructs an independent baseline;
2. copies identical weights and derives packed tensors before device transfer, compilation, correctness checks, warm-up, and timing;
3. checks the Solution with the official comparator;
4. measures alternating baseline and target rounds with the official full-forward timer; and
5. returns a compact structured result to the parent process.

A complete workload sweep prints every case outcome and speedup, the geometric mean within each performance group, the equal-weight group-balanced geometric mean, and the worst-case speedup. Aggregation is reported as `complete` only when every expected case succeeds, passes correctness, and produces a finite positive latency and speedup. Missing, failed, timed-out, out-of-memory, or invalid cases make the sweep `incomplete`; successful cases remain visible, but the runner does not construct a partial project score.

Timeouts and Ctrl+C terminate the worker process tree. Failures are persisted with an explicit stage and type instead of being converted into performance numbers.

## Results

Probe, benchmark, and profile commands each write one strict JSON document per worker run:

```text
results/runs/<run_id>.json
```

Benchmark results include the workload signature and hash, measurement protocol, environment, correctness trials, raw latency samples, medians, speedup, resolved execution path, official snapshot hash, and Solution source hash. Profile results retain a compact top-operation summary for bottleneck selection. Full-sweep aggregation is printed from the ordered per-case results and is not stored as a second experiment database.

`results/` contains generated local measurements and is ignored by Git. Commit implementation and test changes, not machine-specific run history or profiler output.

## Official compatibility entry

The supplied benchmark is preserved byte-for-byte at [`official/torch_transformer_benchmark.py`](official/torch_transformer_benchmark.py), with its checksum recorded in [`official/snapshot.json`](official/snapshot.json).

The root [`torch_transformer_benchmark.py`](torch_transformer_benchmark.py) is a thin compatibility entry that loads the current `UserOptimizedTransformer` and its optional `copy_model_weights` hook, then delegates argument parsing, correctness, and timing to the supplied benchmark:

```powershell
python torch_transformer_benchmark.py --device cuda:0 --dtype float32
```

Use this entry for direct single-configuration compatibility checks. Use `python -m runner` for repeatable development measurements, workload sweeps, device probes, and profiles.

## Tests

```powershell
python -m pytest -q
python -m ruff check runner solution tests torch_transformer_benchmark.py environment_check.py
```

The tests cover the official snapshot and workload contract, weight packing, all mask modes, timing order, fresh-worker result persistence, invalid sweep data, aggregation, and the root compatibility entry.

## Repository map

```text
official/                    Immutable supplied benchmark snapshot
solution/transformer.py     Unique optimized Transformer entry
runner/                      Probe, correctness, benchmark, profile, and sweep logic
runner/workloads/            Machine-readable core workload
tests/                       Focused correctness and runner regressions
results/runs/                Generated local JSON results, ignored by Git
environment/                 Local Windows runtime compatibility hook
```

Performance work stays centered on `solution/`: change the implementation, run the affected case, expand to its performance group when the mechanism works, and use the complete core sweep before retaining a cross-cutting optimization.
