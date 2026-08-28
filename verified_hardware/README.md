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
│   └── reference_formal.json Compact official_01–official_13 formal result
└── run_verified.py        Thin entry into the shared runner
```

## Package contract

`routes.json` is the only deployed route table for that device. A route binds
an exact published Transformer shape and measurement dtype to an exact GPU and
runtime identity. Unmatched inputs use `auto`.

`manifest.json` binds the route table to:

- `official/torch_transformer_benchmark.py`;
- `official/test_shapes.json` and `official_transformer_v1`;
- the current `solution/` implementation;
- the formal measurement protocol;
- the route table and compact formal summary.

The loader skips a stale or incomplete package instead of applying old
evidence to changed code. Package updates are written under a package lock and
published as one transaction.

## Calibration and execution

A new or changed GPU is calibrated through the shared service:

```powershell
python -m runner calibrate --preset formal --device cuda:0
```

The service probes the device, analyzes `official_01` through `official_13`,
runs a bounded smoke screen, formally remeasures the dynamic finalists, and
creates or updates the matching package after correctness, observed-execution,
and conservative-gain checks pass.

Run a known package through its thin launcher:

```powershell
python verified_hardware/<device_id>/run_verified.py --preset smoke
python verified_hardware/<device_id>/run_verified.py --preset formal
```

The launcher verifies the device and manifest, invokes the normal benchmark
service, confirms the sibling route table was used, and writes generated runs
under the same device package. It does not duplicate comparison, measurement,
or route-selection logic.

## Evidence boundary

The compact `results/reference_formal.json` summary reports one row per
`official_01`–`official_13`
with baseline/solution median and P90 latency, speedup, correctness, and the
observed policy. `official_14` remains in the published shape contract but is
not part of the current default formal summary.

Generated smoke runs, raw timing samples, profiler traces, failed candidates,
and historical development results are not tracked in a device package. A
result applies only to the recorded device and software stack.

## Available packages

- [NVIDIA GeForce RTX 4080](nvidia_geforce_rtx_4080/README.md)
