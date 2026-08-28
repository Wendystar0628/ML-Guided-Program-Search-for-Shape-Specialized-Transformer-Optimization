# Verified hardware packages

This directory separates the hardware-neutral optimizer from routes that have
been measured on a specific GPU and software stack. A package is a compact,
executable conclusion for one device; it is not a copy of the Transformer,
kernels, workload, or runner.

```text
verified_hardware/<device_id>/
├── README.md             Human-readable device and result summary
├── profile.json          Compact measured hardware/runtime profile
├── routes.json           Exact published routes
├── manifest.json         Identity binding for the complete package
├── results/
│   ├── reference_formal.json   Last resident Formal result, when current
│   └── reference_streamed.json Provisional streamed Formal result, when current
└── run_verified.py        Thin entry into the shared runner
```

## Package contract

`routes.json` is the only deployed route table for that device. A route binds
an exact published Transformer shape and measurement dtype to an exact GPU and
runtime identity. Unmatched inputs use `eager-sdpa`.

`manifest.json` binds the route table to:

- `official/torch_transformer_benchmark.py`;
- `official/test_shapes.json` and `official_transformer_v1`;
- the current `solution/` implementation;
- the formal measurement protocol;
- the route table and three explicit workload scopes.

The scopes have intentionally different meanings:

- `covered_case_ids`: formally measured workloads with exact deployed routes;
- `provisional_case_ids`: runnable workloads whose current reference or
  measurement method is not yet eligible for a verified route;
- `excluded_case_ids`: workloads with no runnable project path.

The current workload has no excluded shape. `official_14` is streamed and
provisional; it is not disguised as either an exclusion or a verified route.

The loader skips a stale or incomplete package instead of applying old
evidence to changed code. Package updates are written under a package lock and
published as one transaction.

## Calibration and execution

A new or changed GPU is calibrated through the shared service:

```powershell
python -m runner calibrate --preset formal --device cuda:0
```

The service probes the device, analyzes the workload, runs a bounded smoke
screen, formally remeasures route-eligible candidates, and creates or updates
the matching package after conservative-gain checks pass. Streamed provisional
workloads remain part of the benchmark but outside exact-route promotion.

Run a known package's covered exact routes through its thin launcher:

```powershell
python verified_hardware/<device_id>/run_verified.py --preset smoke
python verified_hardware/<device_id>/run_verified.py --preset formal
```

The default `resident` scope verifies `covered_case_ids` and confirms the
sibling route table supplied every exact route. The same launcher can run the
independent provisional scope or both scopes sequentially:

```powershell
python verified_hardware/<device_id>/run_verified.py --scope streamed --preset smoke
python verified_hardware/<device_id>/run_verified.py --scope streamed --preset formal
python verified_hardware/<device_id>/run_verified.py --scope all --preset formal
```

Only a successful streamed Formal run atomically replaces
`results/reference_streamed.json`. It remains target-only and provisional; the
launcher neither adds an exact route nor mixes it into the resident geometric
mean.

## Evidence boundary

When current, `results/reference_formal.json` reports one paired row per
covered exact route. `results/reference_streamed.json` is a compact target-only
record bound to the Bundle Manifest, verified Probe profile, and current
Solution. It contains the selected streamed schedule, latency, memory,
throughput, and project-estimated MFU, but no baseline, speedup, or geometric
mean. A result whose implementation hash predates the current Solution is
historical only and is never loaded as current route evidence.

Generated smoke runs, raw timing samples, profiler traces, failed candidates,
and historical development results are not tracked in a device package. A
result applies only to the recorded device and software stack.

## Available packages

- [NVIDIA GeForce RTX 4080](nvidia_geforce_rtx_4080/README.md)
