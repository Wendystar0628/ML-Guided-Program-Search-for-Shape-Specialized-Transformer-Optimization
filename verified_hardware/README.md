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

1. run the shared probe and calibration workflow on the target device;
2. keep `auto` for shapes without a verified specialized winner;
3. create a package only for the exact hardware/software stack that was tested;
4. promote stable formal winners into that package's `routes.json`;
5. run the package launcher and retain one compact formal reference summary.

The package name is an index, not the complete match condition. Exact matching
still uses the hardware, software, dtype, and Transformer shape fields inside
`routes.json`. Changing a relevant software version or GPU architecture requires
measurement again; another package's winner is only a candidate-ordering hint.

## Verified devices

- [NVIDIA GeForce RTX 4080](nvidia_geforce_rtx_4080/README.md)
