# NVIDIA GeForce RTX 4080 verified package

This directory contains the RTX 4080-specific hardware profile, exact resident
routes, reproducibility manifest, and launcher for `official_transformer_v1`.
Shared Transformer, kernel, routing, and measurement code remains at the
repository root.

## Runtime identity

| Field | Recorded value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4080, Ada, compute capability 8.9, 76 SMs |
| Device memory | 17,170,956,288 bytes (15.99 GiB) |
| Platform | Windows 11, WDDM, driver 610.88 |
| Python / PyTorch | Python 3.12.5 / PyTorch 2.12.1+cu132 |
| CUDA / cuDNN / Triton | CUDA 13.2 / cuDNN 92000 / Triton 3.7.1 |
| Runtime policy | `matmul_precision=high`, TF32 enabled |
| Measured GEMM anchors | FP16 93.43, BF16 99.56, FP32/TF32 49.54 TFLOP/s |
| Measured copy anchor | 291.97 GB/s, 192 MiB device-to-device payload |
| Exact resident scope | `official_01` through `official_13` |
| Provisional streamed scope | `official_14` |

`profile.json` is the authoritative runtime and hardware record. The single
machine-readable performance artifact is
[the RTX 4080 final result](../../results/final/nvidia_geforce_rtx_4080.json).

## Package contents

```text
nvidia_geforce_rtx_4080/
├── README.md
├── manifest.json
├── profile.json
├── routes.json
└── run_verified.py
```

- `manifest.json` binds the official snapshot, workload set, Solution source,
  route table, Formal protocol, and resident/provisional split.
- `profile.json` stores the measured RTX 4080 capabilities and anchors.
- `routes.json` stores one exact deployed policy for every resident Shape.
- `run_verified.py` launches the shared verified runner with this package.

## Reproduce

Run the exact resident routes:

```powershell
.\.venv\Scripts\python.exe verified_hardware/nvidia_geforce_rtx_4080/run_verified.py --scope resident --preset smoke
.\.venv\Scripts\python.exe verified_hardware/nvidia_geforce_rtx_4080/run_verified.py --scope resident --preset formal
```

Run the independent streamed Shape 14 path:

```powershell
.\.venv\Scripts\python.exe verified_hardware/nvidia_geforce_rtx_4080/run_verified.py --scope streamed --preset smoke
.\.venv\Scripts\python.exe verified_hardware/nvidia_geforce_rtx_4080/run_verified.py --scope streamed --preset formal
```

Run both scopes sequentially:

```powershell
.\.venv\Scripts\python.exe verified_hardware/nvidia_geforce_rtx_4080/run_verified.py --scope all --preset formal
```

Formal runs update their corresponding section in the single final artifact;
Smoke artifacts remain under the ignored intermediate-results area. Rebuild the
exact routes after a Solution or runtime change with:

```powershell
.\.venv\Scripts\python.exe -m runner calibrate --preset formal --device cuda:0
```

## Resident Formal result

The current result used FP32 inputs, five Comparator trials per case, 20
warm-ups, 100 timed repeats, three alternating baseline/Solution rounds, seed
1234, high matmul precision, and TF32. All 13 cases completed, all 65
Comparator trials passed, and zero elements failed. The maximum observed
absolute error was `0.0018016696`. Geometric-mean speedup was **8.505x**.

Latencies cover the complete Transformer forward and are measured with CUDA
events. Achieved TFLOP/s counts useful matrix-multiplication work. MFU is the
project estimate against dtype-matched measured GEMM anchors, not an official
competition score. Peak GiB is PyTorch peak CUDA allocation for the target.

| Shape | Exact policy | Baseline median (ms) | Target median / P90 (ms) | Speedup | Achieved TFLOP/s | Est. MFU | Peak GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `official_01` | `graph-mixed-fp16-core-efficient-compiled-norm` | 1.637 | 0.376 / 0.378 | 4.36x | 20.02 | 21.4% | 0.080 |
| `official_02` | `graph-fused-norm` | 1.807 | 0.133 / 0.134 | 13.57x | 0.88 | 1.8% | 0.018 |
| `official_03` | `graph-fused-norm` | 1.840 | 0.139 / 0.139 | 13.21x | 3.38 | 6.8% | 0.021 |
| `official_04` | `graph-fused-norm` | 1.942 | 0.217 / 0.217 | 8.94x | 8.67 | 17.5% | 0.033 |
| `official_05` | `graph-mixed-fp16-core-efficient-compiled-norm` | 2.384 | 0.680 / 0.714 | 3.50x | 22.12 | 23.7% | 0.142 |
| `official_06` | `batch-tiled-mixed-fp16-core-efficient-triton-mixed-norm` | 454.034 | 42.754 / 43.298 | 10.62x | 27.50 | 29.4% | 1.257 |
| `official_07` | `compiled-mixed-fp16-core-efficient` | 1.779 | 0.139 / 0.164 | 12.77x | 4.83 | 5.2% | 0.012 |
| `official_08` | `compiled-mixed-fp16-core-efficient` | 13.435 | 6.054 / 6.588 | 2.22x | 69.54 | 74.4% | 0.227 |
| `official_09` | `graph-mixed-fp16-core-efficient-compiled-norm` | 1.619 | 0.356 / 0.358 | 4.54x | 21.12 | 22.6% | 0.080 |
| `official_10` | `graph-mixed-fp16-core-efficient-compiled-norm` | 1.792 | 0.346 / 0.349 | 5.18x | 21.74 | 23.3% | 0.080 |
| `official_11` | `compiled-mixed-fp16-core-efficient` | 7.130 | 0.385 / 0.388 | 18.52x | 19.54 | 20.9% | 0.017 |
| `official_12` | `graph-fused-norm` | 1.864 | 0.193 / 0.194 | 9.68x | 8.73 | 17.6% | 0.033 |
| `official_13` | `compiled-mixed-fp16-core-shape13-triton-attention` | 110.119 | 2.807 / 3.090 | 39.23x | 42.87 | 45.9% | 0.181 |

The final JSON is authoritative if displayed rounding differs. Its logical
traffic values describe estimated traffic at the current operator boundary;
they are not profiler-measured DRAM traffic.

## Deployed strategy groups

- Shapes 2, 3, 4, and 12 use CUDA Graph replay with fused normalization.
- Shapes 1, 5, 9, and 10 combine mixed-FP16 Efficient SDPA, compiled
  normalization, and CUDA Graph replay.
- Shapes 7, 8, and 11 use whole-forward compiled mixed-FP16 execution.
- Shape 6 uses batch tiling plus a Triton fused mixed-precision residual and
  normalization path.
- Shape 13 uses its own compiled path with a Shape-specific Triton causal
  attention kernel.

These are exact measured winners for this recorded RTX 4080 software stack;
they are not assumed to transfer unchanged to another device or runtime.

## Provisional Shape 14 result

Shape 14 is isolated from resident exact routing. The streamed Formal path
covered logical `B=32` with 16 serial microbatches of size 2.

| Metric | Result |
| --- | ---: |
| Selected policy | `mixed-fp16-core-cudnn` |
| Target median / P90 | 16.237 s / 16.240 s |
| Host end-to-end elapsed time | 18.183 s |
| Useful matmul throughput | 85.69 TFLOP/s |
| Project-estimated MFU | 91.7% |
| Peak CUDA allocation | 7.307 GiB |
| Correctness | 0 / 102,400,000 failed elements |
| Maximum absolute error | 0.000833869 |

Correctness is provisional and uses one full `B=1` long-sequence validation
microbatch. The artifact therefore remains `target_only`: it reports no dense
full-batch baseline, speedup, exact resident route, or contribution to the
resident geometric mean.

## Bound identities

- Workload set SHA-256:
  `621c0f205180f303970ed9e7ce2ee1548cd6c1ac5d46fff1e69dc938039736e9`
- Official snapshot SHA-256:
  `d4f45c9336880b31ab1ae8a8f354aa05862772553162851257490bb936878762`
- Solution implementation SHA-256:
  `57e014b8cbb626905e4a619e2fd468b7c7113b5d2b88217eac876c0fe256d4f4`
- Route-table SHA-256:
  `440d6fa1f6ae86f41ccb5a83ec5029a1f9e84ab1344e28616d98bf2f7de419f9`
- Hardware-profile identity SHA-256:
  `04f06297e42a81b0f9d47eab3aca1bad769e0c2b1ec5b696583e504480f63474`
- Verified-manifest identity SHA-256:
  `d5b33c65c3734f5183c937566df0a888acc080d921166f7a2d5c82a8cd83dfdb`

The verified launcher rejects a stale manifest, unmatched runtime, missing
route, fallback execution, or an observed execution path that does not satisfy
the registered policy evidence. A new hardware or software stack should run
its own Probe and bounded calibration before publishing exact routes.
