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

The exact GPU name, driver, CUDA runtime, PyTorch, operating system, runtime
matmul policy, official snapshot, solution source, and protocol are
machine-checked in the package files rather than approximated in this page.
Triton remains a diagnostic profile field but is not part of the route identity
because the current Solution does not execute Triton kernels.

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
  solution, Formal protocol, run variant, and covered/excluded shape partition.
- `results/reference_formal.json` is the authoritative per-shape result table.
- `run_verified.py` is a thin entry into the shared verified runner.

No previous workload routes or historical development results belong in this
package. A changed solution or runtime invalidates the manifest and falls back
to `auto` until a new formal calibration is published.

## Reproduce

Run a quick checked sweep:

```powershell
.\.venv\Scripts\python.exe verified_hardware/nvidia_geforce_rtx_4080/run_verified.py --preset smoke
```

Run the complete formal protocol:

```powershell
.\.venv\Scripts\python.exe verified_hardware/nvidia_geforce_rtx_4080/run_verified.py --preset formal
```

The launcher confirms the current device and package identities, runs the
shared benchmark service over `official_01` through `official_13`, checks that
the sibling route table supplied every matched route, and writes generated
runs below this directory.

To recalibrate after a code or runtime change, use the shared entry:

```powershell
.\.venv\Scripts\python.exe -m runner calibrate --preset formal --device cuda:0
```

## Results

The formal FP32 sweep used five accuracy trials, 20 warm-up iterations, 100
timed repeats, three alternating timing rounds, high matmul precision, and TF32
enabled. All 13 default shapes completed successfully. Across the complete
sweep:

- geometric-mean speedup: **4.0589x**;
- failed output elements: **0**;
- maximum absolute error: **0.00134444**, below `atol=0.002`;
- `official_14`: excluded from the default sweep.

The table below is a rounded presentation of
[`results/reference_formal.json`](results/reference_formal.json). The JSON file
is the authoritative result if displayed rounding differs.

| Shape | Baseline median | Baseline P90 | Solution median | Solution P90 | Speedup | Policy |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `official_01` | 1.5328 ms | 1.8752 ms | 0.6738 ms | 0.6769 ms | 2.275x | `graph` |
| `official_02` | 1.7341 ms | 2.0271 ms | 0.1608 ms | 0.1608 ms | 10.787x | `graph` |
| `official_03` | 1.7512 ms | 2.0470 ms | 0.1690 ms | 0.1700 ms | 10.364x | `graph` |
| `official_04` | 1.6450 ms | 1.9231 ms | 0.2734 ms | 0.2744 ms | 6.017x | `graph` |
| `official_05` | 2.5298 ms | 4.4271 ms | 1.2334 ms | 1.2390 ms | 2.051x | `graph` |
| `official_06` | 446.7671 ms | 447.4194 ms | 131.2671 ms | 131.7090 ms | 3.403x | `auto` |
| `official_07` | 1.6263 ms | 1.8267 ms | 0.5478 ms | 0.5499 ms | 2.968x | `graph` |
| `official_08` | 13.1917 ms | 13.4147 ms | 11.7862 ms | 11.9974 ms | 1.119x | `auto` |
| `official_09` | 1.4653 ms | 1.6487 ms | 0.6042 ms | 0.6584 ms | 2.425x | `graph` |
| `official_10` | 1.6199 ms | 4.3451 ms | 0.6113 ms | 0.6144 ms | 2.650x | `graph` |
| `official_11` | 7.1168 ms | 7.3145 ms | 1.2339 ms | 1.2380 ms | 5.768x | `graph` |
| `official_12` | 1.7808 ms | 4.4553 ms | 0.2478 ms | 0.2488 ms | 7.186x | `graph` |
| `official_13` | 119.6908 ms | 121.5832 ms | 13.9418 ms | 14.5985 ms | 8.585x | `auto` |

CUDA Graph is the measured winner for the launch-sensitive fixed shapes;
`auto` remains the stable choice for the extreme-batch, wide-GEMM, and
long-sequence cases. `inplace-block` remains an experimental candidate, but its
latest Formal gains on `official_06` and `official_13` were below the 2% route
promotion margin. The comparatively small `1.119x` improvement on
`official_08` identifies the wide `D=1024` GEMM/FFN
path as the clearest remaining optimization target.

## Interpretation

The RTX 4080 package demonstrates that the shared shape analysis, candidates,
formal measurement, and exact dispatcher can produce a concrete route table on
one consumer GPU. It does not imply that another Ada card, a different CUDA
stack, or a professional accelerator has the same winners. Those systems begin
with `auto`, run their own probe and bounded calibration, and receive their own
package.
