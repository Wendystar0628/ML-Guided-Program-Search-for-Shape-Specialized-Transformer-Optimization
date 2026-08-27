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
  manifest.json            Workload, Solution, route, and Formal provenance binding
  run_verified.py          Thin entry into the shared Runner
  results/
    reference_formal.json  One compact, curated formal result
    runs/                  Generated per-case measurements, ignored by Git
    summaries/             Generated sweep summaries, ignored by Git
```

The package must not copy the Transformer implementation, kernels, workload,
comparator, or benchmark logic. `run_verified.py` validates the local device,
loads the sibling package through the normal verified catalog, invokes the
shared Runner, confirms its route attribution, and keeps generated results
inside the package.

`routes.json` is the only machine-readable source for that device's deployed
route decisions. `manifest.json` is the small trust boundary around those
decisions: it binds the route-table hash to the exact Workload hash, current
Solution implementation hash, Formal protocol, and compact Summary/Case IDs.
The shared dispatcher discovers only current packages. A missing, invalid, or
stale manifest causes that package to be skipped closed, so its routes cannot
silently control changed code; unmatched inputs use the shared `auto` fallback.
The checked package launcher rejects the same condition before reporting a
performance result. Duplicate exact keys across accepted packages are rejected
rather than resolved by directory order.

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
routes are published to `routes.json` with one atomic update, then the manifest
is refreshed from those same Formal summaries. Any incomplete package fails
closed. A challenger below the promotion margin keeps the measured incumbent,
while a new exact key without a qualified specialized winner records the
formally measured `auto` decision.

Smoke calibration, plan-only calibration, and `tune` are non-deploying
workflows. `tune` measures only an explicitly supplied `--candidate` list;
automatic candidate planning and ranking belong to `calibrate`. The manual
`promote` command exists only for replaying a compatible historical formal
summary or recovering from an interrupted deployment; it is not part of normal
calibration.

After automatic publication, use the package launcher to reproduce the checked
routes and retain one compact formal reference summary when needed.

The package name is an index, not the complete match condition. Exact matching
still uses the hardware, software, dtype, and Transformer shape fields inside
`routes.json`. Changing a relevant software version or GPU architecture
requires measurement again. Changing the Workload definition, Solution source,
or route table also makes the previous manifest stale until a complete Formal
calibration refreshes it. Another package's winner is only a candidate-ordering
hint.

## Verified devices

- [NVIDIA GeForce RTX 4080](nvidia_geforce_rtx_4080/README.md)
