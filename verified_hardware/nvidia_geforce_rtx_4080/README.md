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
device launcher. The GPU name and Compute Capability identify this stable
device directory; platform, PyTorch, CUDA Runtime, Triton, driver, dtype, and
Workload shape remain mandatory exact-match fields in [`routes.json`](routes.json).

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
    sweeps/
```

- `routes.json` is the only deployed RTX 4080 route table.
- `manifest.json` binds that table to the official snapshot, measured Workload,
  current Solution implementation, Formal protocol, and compact Summary/Route
  identities.
- `run_verified.py` calls the shared project Runner; it does not copy kernels or
  Transformer code.
- `results/reference_formal.json` is the tracked unified nine-case
  `SweepSummary`.
- `results/sweeps/` holds generated isolated sweeps and is ignored by Git.

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
2. rejects the package if its route, official snapshot, Workload, or Solution
   hash is stale;
3. checks that the selected CUDA device and relevant software stack match this
   package;
4. calls the shared `BenchmarkSweepService` directly with the Workload named by
   the manifest and the `dispatch` policy; and
5. writes one isolated sweep directory containing per-case results and the
   unified summary. A Formal run atomically refreshes `reference_formal.json`
   with that same summary document; Case records use location-independent
   `run_id` values.

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

The tracked reference keeps the decision-facing fields compact: same-run
baseline median, optimized median, speedup, and the actually observed route.
Detailed per-run timing files remain generated locally under `results/sweeps/`
when a deeper latency-distribution investigation is needed.

| Case | Baseline median | Target median | Speedup | Observed route |
|---|---:|---:|---:|---|
| `launch_s64_fp16` | 1.7039 ms | 0.1341 ms | `12.7019x` | `cuda-graph` |
| `balanced_s128_fp32` | 5.7135 ms | 3.1620 ms | `1.8069x` | `auto` |
| `balanced_s128_fp16` | 6.0218 ms | 0.8776 ms | `6.8619x` | `balanced-cuda-graph` |
| `attention_s2048_fp16` | 9.5867 ms | 5.3822 ms | `1.7812x` | `long-tail-online` |
| `attention_s2048_causal_fp16` | 11.4540 ms | 5.3565 ms | `2.1383x` | `long-tail-online` |
| `mask_s512_full_fp16` | 5.0591 ms | 2.8017 ms | `1.8057x` | `s512-native-softmax` |
| `mask_s512_padding_fp16` | 5.1610 ms | 2.7996 ms | `1.8435x` | `s512-native-softmax` |
| `mask_s512_causal_padding_fp16` | 5.4932 ms | 2.8037 ms | `1.9593x` | `s512-native-softmax` |
| `wide_s256_bf16` | 10.7643 ms | 10.1873 ms | `1.0566x` | `wide-triton-inplace` |

| Performance group | Geometric-mean speedup |
|---|---:|
| Launch / Graph | `12.7019x` |
| Balanced / Precision | `3.5212x` |
| Long Attention | `1.9516x` |
| Padding / Mask | `1.8684x` |
| Wide GEMM / FFN | `1.0566x` |
| Equal-weight group-balanced score | `2.8007x` |

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
- `12.7019x` CUDA Graph acceleration generalizes beyond the small launch-bound
  fixed-shape case;
- the Wide BF16 path has large remaining headroom—the measured result is only
  `1.0566x` because tuned library GEMMs already dominate that case; or
- a route remains valid after shape, dtype, causal behavior, or numerical
  tolerance changes.

Those changes return to the shared probe, white-box plan, comparator, and
paired candidate measurement loop before promotion into an exact route.
