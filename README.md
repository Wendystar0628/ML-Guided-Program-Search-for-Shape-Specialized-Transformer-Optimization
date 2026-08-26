# AI-Assisted Shape-Aware Transformer Kernel Optimizer

This repository has one main goal: reduce the end-to-end CUDA latency of the supplied PyTorch Transformer while preserving its public interface and numerical behavior.

The primary optimization entry is [`solution/transformer.py`](solution/transformer.py). Performance work should stay focused there: change the implementation, run the benchmark, check correctness, and keep only changes that improve end-to-end latency.

## Current status

The current `UserOptimizedTransformer` is baseline-equivalent. It implements the reference computation and parameter structure so that the optimization loop starts from a known-correct solution.

No optimized kernel has been accepted yet, and this repository currently makes **no speedup claim**. Final measurements against the supplied benchmark will be produced only after the performance implementation is stable.

## Setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
. .\activate_dev_env.ps1
```

## Minimal workflow

Run the default CUDA smoke benchmark (`cuda:0`, `provisional_reference_v1`, and `default_fp32_noncausal_full`):

```powershell
python -m runner benchmark
```

Run the longer benchmark after a change is ready for a more stable measurement:

```powershell
python -m runner benchmark --preset formal --device cuda:0
```

Each benchmark initializes the baseline and Solution with identical weights, checks their outputs before timing, and measures the complete Transformer forward pass.

## Results

Each run writes one compact JSON file to:

```text
results/runs/<run_id>.json
```

The structured result keeps only the information needed for performance development:

- Correctness result and numerical error.
- Raw baseline and Solution latency samples.
- Median latencies and observed speedup.
- Workload and measurement configuration.
- Device and runtime environment.
- SHA-256 of the measured Solution source.

Generated runs are local artifacts and are not committed by default.

## Workload scope

[`runner/workloads/provisional_reference_v1.json`](runner/workloads/provisional_reference_v1.json) contains four development cases covering causal and non-causal attention with full and padded sequences at the published default dimensions.

This workload is provisional. It provides stable coverage while the implementation is being optimized, but it does not claim to be a complete official test suite.

## Official snapshot

The supplied benchmark is preserved at [`official/torch_transformer_benchmark.py`](official/torch_transformer_benchmark.py). Its bytes are locked by the SHA-256 recorded in [`official/snapshot.json`](official/snapshot.json). No additional reproduction framework is part of the performance-development loop.

The root `torch_transformer_benchmark.py` is only a thin direct-official entry for final validation. Day-to-day optimization measurements use the structured Runner above.

## Optimization mainline

Current work is to profile the forward pass, identify the dominant GPU costs, and optimize the highest-impact operations for the representative shapes and mask modes. Correctness and compact timing data support each iteration; final official reproduction is deferred until performance work is complete and the Solution has stabilized.
