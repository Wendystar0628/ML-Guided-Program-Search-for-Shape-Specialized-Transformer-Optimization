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
  manifest.json
  run_verified.py
  results/
    reference_formal.json
    runs/
    summaries/
```

- `routes.json` is the only deployed RTX 4080 route table.
- `manifest.json` binds that table to the measured Workload, current Solution
  implementation, Formal protocol, and compact Summary/Case IDs.
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

The launcher performs five bounded actions:

1. reads the sibling profile, route table, and manifest;
2. rejects the package if its route, Workload, or Solution hash is stale;
3. checks that the selected CUDA device and relevant software stack match this
   package;
4. invokes `python -m runner benchmark` with the Workload named by the
   manifest and the `dispatch` policy; and
5. writes per-case results and a compact sweep summary below this directory.

A hardware mismatch or stale manifest stops before performance claims are
produced. To investigate a new GPU, changed software stack, modified Workload,
or new Solution implementation, use the cross-hardware probe/calibration
workflow from the [root README](../../README.md) instead of weakening this
guard.

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
| `launch_s64_fp16` | 1.9076 ms | 2.3371 ms | 0.1341 ms | 0.1352 ms | `14.2203x` | `cuda-graph` |
| `balanced_s128_fp32` | 2.5298 ms | 3.0466 ms | 1.4510 ms | 1.9425 ms | `1.7435x` | `auto` |
| `balanced_s128_fp16` | 2.6450 ms | 3.1585 ms | 0.8786 ms | 1.3380 ms | `3.0105x` | `balanced-cuda-graph` |
| `attention_s2048_fp16` | 10.0977 ms | 11.0578 ms | 5.7586 ms | 6.7630 ms | `1.7535x` | `long-tail-online` |
| `attention_s2048_causal_fp16` | 12.1641 ms | 13.2183 ms | 5.7539 ms | 6.7669 ms | `2.1141x` | `long-tail-online` |
| `mask_s512_full_fp16` | 5.5731 ms | 6.6193 ms | 3.1319 ms | 3.4860 ms | `1.7795x` | `s512-native-softmax` |
| `mask_s512_padding_fp16` | 5.6018 ms | 6.6280 ms | 3.1596 ms | 3.6082 ms | `1.7730x` | `s512-native-softmax` |
| `mask_s512_causal_padding_fp16` | 5.8716 ms | 6.8026 ms | 3.1452 ms | 3.4422 ms | `1.8668x` | `s512-native-softmax` |
| `wide_s256_bf16` | 11.8344 ms | 12.6751 ms | 10.7648 ms | 11.7958 ms | `1.0994x` | `wide-triton-inplace` |

| Performance group | Geometric-mean speedup |
|---|---:|
| Launch / Graph | `14.2203x` |
| Balanced / Precision | `2.2910x` |
| Long Attention | `1.9254x` |
| Padding / Mask | `1.8059x` |
| Wide GEMM / FFN | `1.0994x` |
| Equal-weight group-balanced score | `2.6246x` |

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
- `14.2203x` CUDA Graph acceleration generalizes beyond the small launch-bound
  fixed-shape case;
- the Wide BF16 path has large remaining headroom—the measured result is only
  `1.0994x` because tuned library GEMMs already dominate that case; or
- a route remains valid after shape, dtype, causal behavior, or numerical
  tolerance changes.

Those changes return to the shared probe, white-box plan, comparator, and
paired candidate measurement loop before promotion into an exact route.
