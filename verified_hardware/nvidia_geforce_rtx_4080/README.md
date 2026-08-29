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
| Measured GEMM anchors | FP16 86.34, BF16 94.18, FP32/TF32 47.03 TFLOP/s |
| Measured copy anchor | 269.86 GB/s, 192 MiB device-to-device payload |
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
absolute error was `0.0018820614`. The project's unweighted resident
geometric-mean speedup was **16.998x**.

Latencies cover the complete Transformer forward and are measured with CUDA
events. Achieved TFLOP/s counts useful matrix-multiplication work. MFU is the
project estimate against dtype-matched measured GEMM anchors, not an official
competition score. Peak GiB is PyTorch peak CUDA allocation for the target.

| Shape | Exact policy | Baseline median (ms) | Target median / P90 (ms) | Speedup | Achieved TFLOP/s | Est. MFU | Peak GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `official_01` | `graph-fp16-shadow-efficient-compiled-norm` | 4.502 | 0.336 / 0.344 | 13.41x | 22.40 | 25.9% | 0.081 |
| `official_02` | `graph-fused-norm` | 4.810 | 0.133 / 0.151 | 36.13x | 0.88 | 1.9% | 0.018 |
| `official_03` | `graph-fused-norm` | 4.686 | 0.139 / 0.140 | 33.65x | 3.38 | 7.2% | 0.021 |
| `official_04` | `graph-fused-norm` | 4.608 | 0.221 / 0.222 | 20.83x | 8.50 | 18.1% | 0.033 |
| `official_05` | `graph-fp16-shadow-efficient-triton-mixed-norm-reuse-input` | 4.357 | 0.469 / 0.473 | 9.29x | 32.09 | 37.2% | 0.049 |
| `official_06` | `batch-tiled-shape06-triton-mixed-norm-fp16-shadow` | 480.943 | 38.952 / 39.376 | 12.35x | 30.18 | 35.0% | 1.258 |
| `official_07` | `graph-mixed-fp16-core-efficient-compiled-norm` | 4.545 | 0.196 / 0.197 | 23.24x | 3.44 | 4.0% | 0.032 |
| `official_08` | `compiled-shape08-fp16-shadow-weights` | 14.105 | 6.157 / 6.368 | 2.29x | 68.37 | 79.2% | 0.274 |
| `official_09` | `graph-fp16-shadow-efficient-compiled-norm` | 4.185 | 0.326 / 0.337 | 12.85x | 23.11 | 26.8% | 0.081 |
| `official_10` | `graph-fp16-shadow-efficient-compiled-norm` | 4.552 | 0.315 / 0.329 | 14.43x | 23.86 | 27.6% | 0.081 |
| `official_11` | `compiled-shape11-dh8-triton-fp16-shadow` | 7.535 | 0.275 / 0.393 | 27.36x | 27.32 | 31.6% | 0.018 |
| `official_12` | `graph-fused-norm` | 4.655 | 0.199 / 0.200 | 23.43x | 8.46 | 18.0% | 0.033 |
| `official_13` | `compiled-shape13-triton-attention-fp16-shadow` | 116.456 | 2.903 / 3.274 | 40.12x | 41.46 | 48.0% | 0.198 |

The final JSON is authoritative if displayed rounding differs. Its logical
traffic values describe estimated traffic at the current operator boundary;
they are not profiler-measured DRAM traffic.

## Deployed strategy groups

- Shapes 2, 3, 4, and 12 use CUDA Graph replay with fused normalization.
- Shapes 1, 9, and 10 combine FP16 shadow weights, Efficient SDPA, compiled
  normalization, and CUDA Graph replay.
- Shape 5 adds FP16 shadow weights, the custom Triton mixed
  residual/LayerNorm boundary, version-aware unchanged-input staging, and CUDA
  Graph replay.
- Shape 7 uses a Graph-composed mixed-FP16 fixed plan.
- Shape 8 keeps its official FP32 parameters authoritative and uses derived,
  non-persistent FP16 shadow weights in its compiled inference path.
- Shape 6 uses batch tiling, FP16 shadow weights, and Triton kernels for both
  its initial FP32-to-FP16 LayerNorm and mixed residual/LayerNorm boundaries.
- Shape 11 uses a compiled FP16-shadow path with a `head_dim=8` online-attention
  Triton kernel that directly writes the flattened BSD layout.
- Shape 13 uses a compiled FP16-shadow path with its own Triton causal-attention
  kernel.

These are exact measured winners for this recorded RTX 4080 software stack;
they are not assumed to transfer unchanged to another device or runtime.

## Provisional Shape 14 result

Shape 14 is isolated from resident exact routing. The streamed Formal path
covered logical `B=32` with 16 serial microbatches of size 2.

| Metric | Result |
| --- | ---: |
| Selected policy | `mixed-fp16-core-cudnn` |
| Target median / P90 | 17.085 s / 17.086 s |
| Host end-to-end elapsed time | 20.554 s |
| Useful matmul throughput | 81.43 TFLOP/s |
| Project-estimated MFU | 94.3% |
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
  `67c99918effd4c8d37882c7844b368b653bc7e9a98f6e8e64c7206ba4ef2d022`
- Route-table SHA-256:
  `23b07b547f3f19b687b0aed80a971e0905a1dd2441560a18502d52bfbcfbcfc9`
- Hardware-profile identity SHA-256:
  `f51575585b27e9fac56f2a0e900eca078a241cb62398ea3a3383651b939c6d02`
- Verified-manifest identity SHA-256:
  `228f478bdfa6ac33cdd4ba847be0c119f46d2fa3c77ffb7e846aad5cec99c9c0`

The verified launcher rejects a stale manifest, unmatched runtime, missing
route, fallback execution, or an observed execution path that does not satisfy
the registered policy evidence. A new hardware or software stack should run
its own Probe and bounded calibration before publishing exact routes.
