# NVIDIA GeForce RTX 4080 verified package

This directory is the RTX 4080-specific deployment and measurement package for
`official_transformer_v1`. Shared workload, Transformer, kernel, routing, and
measurement code remains at the repository root.

## Scope and runtime

| Field | Recorded value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4080, Ada, compute capability 8.9, 76 SMs |
| Device memory | 17,170,956,288 bytes (15.99 GiB) |
| Platform | Windows 11, WDDM, driver 610.88 |
| Python / PyTorch | Python 3.12.5 / PyTorch 2.12.1+cu132 |
| CUDA / cuDNN / Triton | CUDA 13.2 / cuDNN 92000 / Triton 3.7.1 |
| Runtime policy | `matmul_precision=high`, TF32 enabled |
| Measured roofs | FP16 93.73, BF16 93.91, FP32/TF32 47.22 TFLOP/s |
| Measured copy anchor | 270.88 GB/s, 192 MiB device-to-device payload |
| Exact resident routes | `official_01` through `official_13` |
| Provisional streamed scope | `official_14` |

Efficient SDPA, cuDNN SDPA, CUDA Graph, BF16, and Triton were available in the
recorded Probe. `profile.json` is the authoritative hardware and runtime record.

## Package contents

```text
nvidia_geforce_rtx_4080/
├── README.md
├── manifest.json
├── profile.json
├── routes.json
├── run_verified.py
└── results/
    ├── reference_formal.json
    └── reference_streamed.json
```

- `manifest.json` binds the official snapshot, workload set, Solution source,
  route table, Formal protocol, and resident/provisional workload partition.
- `profile.json` stores the measured RTX 4080 profile and performance anchors.
- `routes.json` stores one exact route for each resident workload.
- `results/reference_formal.json` stores the paired Formal result for Shapes
  1-13.
- `results/reference_streamed.json` stores the separate provisional target-only
  Shape 14 result.
- `run_verified.py` is a thin launcher over the shared verified runner.

## Reproduce

Run only the exact resident routes:

```powershell
.\.venv\Scripts\python.exe verified_hardware/nvidia_geforce_rtx_4080/run_verified.py --scope resident --preset smoke
.\.venv\Scripts\python.exe verified_hardware/nvidia_geforce_rtx_4080/run_verified.py --scope resident --preset formal
```

Run only the independent Shape 14 streamed path:

```powershell
.\.venv\Scripts\python.exe verified_hardware/nvidia_geforce_rtx_4080/run_verified.py --scope streamed --preset smoke
.\.venv\Scripts\python.exe verified_hardware/nvidia_geforce_rtx_4080/run_verified.py --scope streamed --preset formal
```

Run both scopes sequentially:

```powershell
.\.venv\Scripts\python.exe verified_hardware/nvidia_geforce_rtx_4080/run_verified.py --scope all --preset formal
```

Formal resident execution replaces `reference_formal.json`; Formal streamed
execution replaces `reference_streamed.json`. Smoke runs create ordinary run
artifacts without replacing either reference. Recalibrate exact routes after a
code or runtime change with:

```powershell
.\.venv\Scripts\python.exe -m runner calibrate --preset formal --device cuda:0
```

## Resident Formal result

This result uses FP32 inputs, five Comparator trials per case, 20 warm-ups, 100
timed repeats, three alternating baseline/Solution rounds, seed 1234, high
matmul precision, and TF32 enabled. All 13 cases completed; all 65 Comparator
trials passed with zero failed elements. Maximum absolute error was
`0.0017772168` under `rtol=0.02` and `atol=0.002`. Geometric-mean speedup was
**10.980x**.

Times are complete Transformer-forward milliseconds. Peak is Solution CUDA
allocated memory. MFU is the project's estimate against dtype-matched measured
GEMM roofs, not an official competition score.

| Case | Exact route | Baseline median | Solution median / P90 | Speedup | Useful TFLOP/s | Est. MFU | Peak GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `official_01` | `graph-mixed-fp16-efficient-compiled-norm` | 4.544 | 0.492 / 0.493 | 9.25x | 15.31 | 30.1% | 0.084 |
| `official_02` | `graph-fused-norm` | 4.642 | 0.140 / 0.143 | 33.09x | 0.84 | 1.8% | 0.018 |
| `official_03` | `graph-fused-norm` | 4.527 | 0.148 / 0.148 | 30.49x | 3.17 | 6.7% | 0.022 |
| `official_04` | `graph-fused-norm` | 4.482 | 0.238 / 0.238 | 18.87x | 7.92 | 16.8% | 0.034 |
| `official_05` | `graph-mixed-fp16-efficient-compiled-norm` | 4.214 | 0.929 / 1.146 | 4.54x | 16.20 | 31.9% | 0.150 |
| `official_06` | `mixed-fp16-core-efficient-triton-norm` | 450.089 | 104.440 / 106.300 | 4.31x | 11.26 | 12.0% | 4.900 |
| `official_07` | `graph-mixed-fp16-efficient-compiled-norm` | 4.403 | 0.295 / 0.296 | 14.93x | 2.28 | 3.9% | 0.033 |
| `official_08` | `mixed-fp16-core-efficient` | 13.210 | 7.565 / 7.775 | 1.75x | 55.65 | 59.4% | 0.352 |
| `official_09` | `graph-mixed-fp16-efficient-compiled-norm` | 3.892 | 0.479 / 0.480 | 8.12x | 15.70 | 30.9% | 0.084 |
| `official_10` | `graph-mixed-fp16-efficient-compiled-norm` | 4.331 | 0.469 / 0.470 | 9.23x | 16.04 | 31.6% | 0.084 |
| `official_11` | `graph-mixed-fp16-efficient-compiled-norm` | 7.103 | 0.609 / 0.611 | 11.66x | 12.35 | 24.3% | 0.084 |
| `official_12` | `graph-fused-norm` | 4.499 | 0.214 / 0.214 | 21.02x | 7.85 | 16.6% | 0.034 |
| `official_13` | `mixed-fp16-core-efficient` | 108.424 | 5.302 / 5.482 | 20.45x | 22.69 | 24.2% | 0.259 |

`results/reference_formal.json` is authoritative if displayed rounding differs.
Its logical-traffic fields estimate traffic at the current operator boundary;
they are not measured DRAM traffic or profiler counters.

## Provisional Shape 14 result

Shape 14 is intentionally isolated from resident exact routing. The streamed
Formal path dynamically screened supported policy and microbatch combinations,
then covered logical `B=32` as 16 serial microbatches of size 2.

| Metric | Result |
| --- | ---: |
| Selected policy | `mixed-fp16-core-cudnn` |
| Target median / P90 | 15.988 s / 15.988 s |
| End-to-end elapsed time | 19.333 s |
| Useful matmul throughput | 87.02 TFLOP/s |
| Project-estimated MFU | 92.8% |
| Peak CUDA allocation | 6.927 GiB |
| Correctness | 0 / 102,400,000 failed elements |
| Maximum absolute error | 0.000833869 |

Correctness is provisional and uses one full `B=1` long-sequence validation
microbatch. This artifact is therefore `target_only`: it reports no fabricated
dense full-batch baseline, speedup, exact route, or contribution to the
resident geometric mean.

## Identity and reproducibility boundaries

The exact route key includes GPU name, compute capability, operating-system
family, PyTorch version, CUDA runtime, driver, matmul precision, TF32 policy,
input dtype, and complete workload shape. The Manifest additionally binds:

- workload set SHA-256 `621c0f205180f303970ed9e7ce2ee1548cd6c1ac5d46fff1e69dc938039736e9`;
- official snapshot SHA-256 `d4f45c9336880b31ab1ae8a8f354aa05862772553162851257490bb936878762`;
- Solution implementation SHA-256 `07882d07b4d1f682ebaa1fc2d8bc1a7a2ff65ab2d9a3740481e4c9499ae34d9a`;
- route-table SHA-256 `abe29c3fba8efcf2aaf74c5d3177f74e57d2b9fb462fec2a290e14356f9a9b3b`.

The verified launcher rejects stale provenance, an unmatched runtime, missing
routes, fallback execution, or a policy whose observed execution path does not
match its registered evidence. Another GPU or software stack must run its own
Probe and bounded calibration; this package does not claim that RTX 4080 winners
transfer unchanged.
