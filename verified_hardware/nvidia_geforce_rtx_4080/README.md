# NVIDIA GeForce RTX 4080 verified package

This directory contains the RTX 4080-specific routes and compact measurements
for the published `official_transformer_v1` workload. Shared Transformer,
kernel, workload, and runner code remains at the repository root.

## Scope

| Field | Value |
| --- | --- |
| GPU family | NVIDIA GeForce RTX 4080 |
| CUDA architecture | Ada Lovelace, compute capability 8.9 |
| Workload | `official_01` through `official_13` |
| Published but excluded from the default sweep | `official_14` |
| Correctness | `rtol=0.02`, `atol=0.002` |
| Performance unit | Complete Transformer forward latency |

The exact GPU name, driver, CUDA runtime, PyTorch, Triton, operating system,
official snapshot, solution source, and protocol are machine-checked in the
package files rather than approximated in this page.

## Files

```text
nvidia_geforce_rtx_4080/
├── README.md
├── profile.json
├── routes.json
├── manifest.json
├── run_verified.py
└── results/
    └── reference_formal.json
```

- `profile.json` records the compact RTX 4080 hardware/runtime profile.
- `routes.json` contains only exact routes measured for the published shapes.
- `manifest.json` binds those routes to the official benchmark, shapes,
  solution, formal protocol, and compact result.
- `results/reference_formal.json` is the authoritative per-shape result table.
- `run_verified.py` is a thin entry into the shared verified runner.

No previous workload routes or historical development results belong in this
package. A changed solution or runtime invalidates the manifest and falls back
to `auto` until a new formal calibration is published.

## Reproduce

Run a quick checked sweep:

```powershell
python verified_hardware/nvidia_geforce_rtx_4080/run_verified.py --preset smoke
```

Run the complete formal protocol:

```powershell
python verified_hardware/nvidia_geforce_rtx_4080/run_verified.py --preset formal
```

The launcher confirms the current device and package identities, runs the
shared benchmark service over `official_01` through `official_13`, checks that
the sibling route table supplied every matched route, and writes generated
runs below this directory.

To recalibrate after a code or runtime change, use the shared entry:

```powershell
python -m runner calibrate --preset formal --device cuda:0
```

## Results

The formal FP32 sweep used five accuracy trials, 20 warm-up iterations, 100
timed repeats, three alternating timing rounds, high matmul precision, and TF32
enabled. All 13 default shapes completed successfully. Across the complete
sweep:

- geometric-mean speedup: **4.1827x**;
- failed output elements: **0**;
- maximum absolute error: **0.00134444**, below `atol=0.002`;
- `official_14`: excluded from the default sweep.

The table below is a rounded presentation of
[`results/reference_formal.json`](results/reference_formal.json). The JSON file
is the authoritative result if displayed rounding differs.

| Shape | Baseline median | Baseline P90 | Solution median | Solution P90 | Speedup | Policy |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `official_01` | 1.6362 ms | 4.5255 ms | 0.6764 ms | 0.7352 ms | 2.419x | `graph` |
| `official_02` | 1.7675 ms | 4.4503 ms | 0.1608 ms | 0.1608 ms | 10.994x | `graph` |
| `official_03` | 1.8625 ms | 4.4392 ms | 0.1690 ms | 0.1700 ms | 11.024x | `graph` |
| `official_04` | 1.6947 ms | 5.0818 ms | 0.2734 ms | 0.2756 ms | 6.198x | `graph` |
| `official_05` | 4.1389 ms | 4.5259 ms | 1.2360 ms | 1.2421 ms | 3.349x | `graph` |
| `official_06` | 446.5157 ms | 447.4913 ms | 131.3004 ms | 131.8300 ms | 3.401x | `auto` |
| `official_07` | 1.6671 ms | 2.0549 ms | 0.5069 ms | 0.7384 ms | 3.289x | `graph` |
| `official_08` | 13.2070 ms | 13.4277 ms | 11.7720 ms | 11.9963 ms | 1.122x | `auto` |
| `official_09` | 1.4581 ms | 3.8819 ms | 0.6021 ms | 0.6574 ms | 2.422x | `graph` |
| `official_10` | 1.6748 ms | 4.4538 ms | 0.6134 ms | 0.6687 ms | 2.730x | `graph` |
| `official_11` | 7.1368 ms | 7.3575 ms | 1.6888 ms | 2.0576 ms | 4.226x | `auto` |
| `official_12` | 1.7233 ms | 4.4845 ms | 0.2478 ms | 0.2488 ms | 6.954x | `graph` |
| `official_13` | 108.5486 ms | 114.5298 ms | 13.4292 ms | 13.9710 ms | 8.083x | `inplace-block` |

CUDA Graph is the measured winner for the launch-sensitive fixed shapes;
`auto` remains strongest for the extreme-batch, wide-GEMM, and 16-head cases;
`inplace-block` wins the `S=1024` long-sequence case. The comparatively small
`1.122x` improvement on `official_08` identifies the wide `D=1024` GEMM/FFN
path as the clearest remaining optimization target.

## Interpretation

The RTX 4080 package demonstrates that the shared shape analysis, candidates,
formal measurement, and exact dispatcher can produce a concrete route table on
one consumer GPU. It does not imply that another Ada card, a different CUDA
stack, or a professional accelerator has the same winners. Those systems begin
with `auto`, run their own probe and bounded calibration, and receive their own
package.
