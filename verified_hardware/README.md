# Verified hardware packages

This directory separates hardware-neutral optimization code from conclusions
that have been measured on one exact GPU and software stack.

It is deliberately a small package catalog, not a performance database. The
shared implementation remains in `solution/`, the shared workload and Runner
remain in `runner/`, and generated exploratory measurements remain untracked.

## Package contract

Each verified GPU uses one stable directory name and contains:

```text
<device_id>/
  README.md                Human-readable conclusions, routes, and limitations
  profile.json             Checked hardware and software identity
  routes.json              Single machine-readable source of exact routes
  run_verified.py          Thin entry into the shared Runner
  results/
    reference_formal.json  One compact, curated formal result
    runs/                  Generated per-case measurements, ignored by Git
    summaries/             Generated sweep summaries, ignored by Git
```

The package must not copy the Transformer implementation, kernels, workload,
comparator, or benchmark logic. `run_verified.py` validates the local device,
selects the sibling `routes.json`, invokes the shared Runner, and keeps generated
results inside the package.

`routes.json` is the only machine-readable source for that device's deployed
route decisions. The shared dispatcher may discover verified packages, while
an explicit route-table path can select one package for reproduction. Duplicate
exact keys across packages are rejected rather than resolved by directory
order.

`reference_formal.json` is a concise evidence snapshot, not raw history. It
should identify the workload and software stack, list correctness and paired
latency/speedup for every case, and report the fixed workload aggregates.
Timestamped runs and summaries are useful locally but remain ignored.

## Adding another GPU

Run the normal formal calibration on the target device:

```powershell
python -m runner calibrate --preset formal --device cuda:0
```

Runner probes the device once, measures the bounded candidates with the complete
Transformer Workloads, applies every correctness and promotion gate, then
automatically locates or creates the stable package for the measured
hardware/software identity. Internally, the command uses one Probe, a
deployable Smoke Top-3 screen by default, and a dynamically reduced Formal set.
On a new device the Formal set contains at most `eager-auto` and the best valid
Smoke challenger. If an exact specialized route already exists, that incumbent
is retained as a third possible Formal candidate. These controls are
deduplicated before measurement.

Formal publication uses only the strict remeasurements, not the Smoke ranking.
Workloads that share one runtime route key are evaluated jointly. All accepted
routes are published to `routes.json` with one atomic update; any failed case or
gate leaves the prior table unchanged. A challenger below the promotion margin
keeps the measured incumbent, while a new exact key without a qualified
specialized winner records the formally measured `auto` decision.

Smoke calibration, plan-only calibration, and `tune` are non-deploying
workflows. The manual `promote` command exists only for replaying a compatible
historical formal summary or recovering from an interrupted deployment; it is
not part of normal calibration.

After automatic publication, use the package launcher to reproduce the checked
routes and retain one compact formal reference summary when needed.

The package name is an index, not the complete match condition. Exact matching
still uses the hardware, software, dtype, and Transformer shape fields inside
`routes.json`. Changing a relevant software version or GPU architecture requires
measurement again; another package's winner is only a candidate-ordering hint.

## Verified devices

- [NVIDIA GeForce RTX 4080](nvidia_geforce_rtx_4080/README.md)
