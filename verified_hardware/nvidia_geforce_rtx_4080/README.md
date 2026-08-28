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
Triton remains a diagnostic profile field rather than a route-identity field.
The compiled residual/LayerNorm policy is published only after its actual
compiled backend is observed during Formal measurement; if that backend later
cannot execute, the policy fails explicitly instead of being reported as a
fused route while silently running native operators.

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

The formal FP32 sweep used five accuracy trials, a fixed unmeasured 0.5-second
CUDA conditioning step per fresh worker, 20 model warm-up iterations, 100 timed
repeats, three alternating timing rounds, high matmul precision, and TF32
enabled. All 13 default shapes completed successfully. Across the complete
sweep:

- geometric-mean speedup: **5.2548x**;
- failed output elements: **0**;
- maximum absolute error: **0.00155115**, below `atol=0.002`;
- `official_14`: excluded from the default sweep.

The table below is a rounded presentation of
[`results/reference_formal.json`](results/reference_formal.json). The JSON file
is the authoritative result if displayed rounding differs.

| Shape | Baseline median | Baseline P90 | Solution median | Solution P90 | Speedup | Policy |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `official_01` | 1.5697 ms | 2.0112 ms | 0.5407 ms | 0.7678 ms | 2.903x | `graph-mixed-fp16-efficient` |
| `official_02` | 1.7820 ms | 2.1952 ms | 0.1403 ms | 0.1413 ms | 12.703x | `graph-fused-norm` |
| `official_03` | 1.7905 ms | 2.1340 ms | 0.1475 ms | 0.1485 ms | 12.142x | `graph-fused-norm` |
| `official_04` | 1.7875 ms | 2.1365 ms | 0.2365 ms | 0.2376 ms | 7.557x | `graph-fused-norm` |
| `official_05` | 2.3777 ms | 2.8938 ms | 0.9984 ms | 1.4043 ms | 2.382x | `graph-mixed-fp16-efficient` |
| `official_06` | 483.2210 ms | 486.2453 ms | 141.2731 ms | 142.8365 ms | 3.420x | `auto` |
| `official_07` | 1.7025 ms | 2.0548 ms | 0.3533 ms | 0.3564 ms | 4.819x | `graph-mixed-fp16-efficient` |
| `official_08` | 14.1148 ms | 14.5408 ms | 12.6177 ms | 12.9252 ms | 1.119x | `auto` |
| `official_09` | 1.5321 ms | 1.8905 ms | 0.5294 ms | 0.6108 ms | 2.894x | `graph-mixed-fp16-efficient` |
| `official_10` | 1.7209 ms | 2.1107 ms | 0.5192 ms | 0.7134 ms | 3.315x | `graph-mixed-fp16-efficient` |
| `official_11` | 7.5361 ms | 8.0348 ms | 0.6584 ms | 1.0502 ms | 11.446x | `graph-mixed-fp16-efficient` |
| `official_12` | 1.8411 ms | 2.1542 ms | 0.2140 ms | 0.2150 ms | 8.603x | `graph-fused-norm` |
| `official_13` | 116.7857 ms | 117.8454 ms | 7.0385 ms | 7.5213 ms | 16.592x | `mixed-fp16-efficient` |

CUDA Graph remains the launch-overhead solution for fixed small shapes. The
mixed FP16 Efficient Attention policies now cover the measured short-sequence
families and `official_13`; compiled residual/LayerNorm is combined with Graph
for `official_02/03/04/12`. `official_06` retains `auto` because the tested
mixed-attention alternative was slower, while `official_08` remains close to
its library-GEMM throughput limit and is not a good target for a new
Transformer algorithm.

## Interpretation

The RTX 4080 package demonstrates that the shared shape analysis, candidates,
formal measurement, and exact dispatcher can produce a concrete route table on
one consumer GPU. It does not imply that another Ada card, a different CUDA
stack, or a professional accelerator has the same winners. Those systems begin
with `auto`, run their own probe and bounded calibration, and receive their own
package.
