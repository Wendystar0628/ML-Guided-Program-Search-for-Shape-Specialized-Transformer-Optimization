# NVIDIA GeForce RTX 4080 verified package

This directory contains the RTX 4080-specific conclusion. It is intentionally
separate from the cross-hardware optimizer: the shared code can run on other
devices, while the routes and speedups below only describe the exact measured
RTX 4080 stack.

## Verified environment

| Item | Verified value |
|---|---|
| GPU | NVIDIA GeForce RTX 4080 |
| Architecture key | Ada, compute capability `8.9` (`sm_89`) |
| SM count | 76 |
| L2 cache | 64 MiB |
| Device memory | 16 GiB |
| Memory bus | 256 bit |
| Operating system | Windows |
| Python | 3.12 |
| PyTorch | `2.12.1+cu132` |
| CUDA runtime | `13.2` |
| Triton | `3.7.1` |
| NVIDIA driver in the reference run | `610.88` |

[`profile.json`](profile.json) is the machine-readable identity used by the
device launcher. The driver is retained as provenance; exact route matching is
based on the stable fields in [`routes.json`](routes.json).

## Directory contents

```text
nvidia_geforce_rtx_4080/
  README.md
  profile.json
  routes.json
  run_verified.py
  results/
    reference_formal.json
    runs/
    summaries/
```

- `routes.json` is the only deployed RTX 4080 route table.
- `run_verified.py` calls the shared project Runner; it does not copy kernels or
  Transformer code.
- `results/reference_formal.json` is the compact tracked nine-case evidence.
- `results/runs/` and `results/summaries/` hold generated local output and are
  ignored by Git.

## Run the verified route

From the repository root:

```powershell
python verified_hardware/nvidia_geforce_rtx_4080/run_verified.py --preset smoke
```

For the complete formal protocol:

```powershell
python verified_hardware/nvidia_geforce_rtx_4080/run_verified.py --preset formal
```

The launcher performs four bounded actions:

1. reads the sibling profile and route table;
2. checks that the selected CUDA device and relevant software stack match this
   package;
3. invokes `python -m runner benchmark` with the shared
   `transformer_core_v1` workload and `dispatch` policy; and
4. writes per-case results and a compact sweep summary below this directory.

A mismatch stops before performance claims are produced. To investigate a new
GPU or a changed stack, use the cross-hardware probe/calibration workflow from
the [root README](../../README.md) instead of weakening this guard.

## Exact route decisions

The table contains eight exact runtime keys and covers all nine core cases. The
two non-causal S512 mask cases have the same runtime-visible shape key and
deliberately share one route. Balanced FP32 has an explicit measured
`auto` route, making that decision part of the RTX 4080 package rather than an
accidental catalog fallback.

| Core case | Selected policy | Main measured reason |
|---|---|---|
| `launch_s64_fp16` | `cuda-graph` | Full-model replay removes dominant host launch and framework gaps |
| `balanced_s128_fp32` | `auto` (explicit route) | No specialized candidate justified replacing the shared safe path |
| `balanced_s128_fp16` | `balanced-cuda-graph` | Fixed-shape graph replay reduces repeated launch overhead |
| `attention_s2048_fp16` | `long-tail-online` | Final-block streaming attention avoids one full score/probability materialization |
| `attention_s2048_causal_fp16` | `long-tail-online` | Same bounded streaming mechanism with causal handling |
| `mask_s512_full_fp16` | `s512-native-softmax` | Fused scale/mask plus native half-output softmax removes avoidable conversions |
| `mask_s512_padding_fp16` | `s512-native-softmax` | Shares the non-causal S512 exact shape key and consumes the original padding mask |
| `mask_s512_causal_padding_fp16` | `s512-native-softmax` | Same S512 mechanism with the exact causal route key |
| `wide_s256_bf16` | `wide-triton-inplace` | Single-pass QKV layout plus exact in-place GELU reuses the FFN buffer |

Dispatch does not inspect mask contents or benchmark candidates during
`forward`. Any hardware, software, dtype, shape, or causal condition outside
these eight exact keys falls back to the shared default `auto` policy.

## Formal reference result

The current `transformer_core_v1` formal dispatch sweep used five correctness
trials, 20 warm-up iterations, three alternating timing rounds, and 100
samples per round. All 45 correctness trials passed the full Transformer
comparator, and all nine case decisions came from calibrated RTX 4080 routes.

| Case | Baseline median | Baseline p90 | Target median | Target p90 | Speedup | Route |
|---|---:|---:|---:|---:|---:|---|
| `launch_s64_fp16` | 1.7760 ms | 3.7437 ms | 0.1341 ms | 0.1343 ms | `13.2399x` | `cuda-graph` |
| `balanced_s128_fp32` | 2.2349 ms | 3.6565 ms | 1.4551 ms | 1.5126 ms | `1.5359x` | `auto` |
| `balanced_s128_fp16` | 2.3742 ms | 2.7653 ms | 0.8786 ms | 0.8806 ms | `2.7022x` | `balanced-cuda-graph` |
| `attention_s2048_fp16` | 8.9364 ms | 9.1373 ms | 5.0903 ms | 5.2342 ms | `1.7556x` | `long-tail-online` |
| `attention_s2048_causal_fp16` | 10.7069 ms | 10.9150 ms | 5.0964 ms | 5.2555 ms | `2.1009x` | `long-tail-online` |
| `mask_s512_full_fp16` | 4.9403 ms | 5.0562 ms | 2.7505 ms | 2.7638 ms | `1.7962x` | `s512-native-softmax` |
| `mask_s512_padding_fp16` | 4.9326 ms | 5.0654 ms | 2.7494 ms | 2.7658 ms | `1.7940x` | `s512-native-softmax` |
| `mask_s512_causal_padding_fp16` | 5.2111 ms | 5.3335 ms | 2.7494 ms | 2.7709 ms | `1.8953x` | `s512-native-softmax` |
| `wide_s256_bf16` | 10.2226 ms | 10.4347 ms | 9.6881 ms | 9.9023 ms | `1.0552x` | `wide-triton-inplace` |

| Performance group | Geometric-mean speedup |
|---|---:|
| Launch / Graph | `13.2399x` |
| Balanced / Precision | `2.0372x` |
| Long Attention | `1.9205x` |
| Padding / Mask | `1.8279x` |
| Wide GEMM / FFN | `1.0552x` |
| Equal-weight group-balanced score | `2.5114x` |

The compact machine-readable evidence is
[`results/reference_formal.json`](results/reference_formal.json). Same-run
baseline and target timings are the performance evidence.

## What this result does and does not establish

The result establishes correctness and performance for this workload, this
RTX 4080, and the recorded software stack. It also provides useful mechanism
hypotheses for similar GPUs.

It does not establish that:

- these route choices are best on another GPU, another architecture, or another
  PyTorch/CUDA/Triton stack;
- `13.2399x` CUDA Graph acceleration generalizes beyond the small launch-bound
  fixed-shape case;
- the Wide BF16 path has large remaining headroom—the measured result is only
  `1.0552x` because tuned library GEMMs already dominate that case; or
- a route remains valid after shape, dtype, causal behavior, or numerical
  tolerance changes.

Those changes return to the shared probe, white-box plan, comparator, and
paired candidate measurement loop before promotion into an exact route.
