# Results

This directory has one public result surface and one local working area.

```text
results/
├── final/          Concise, tracked performance results by verified GPU
└── intermediate/   Regenerable runs, sweeps, tuning, probes, and profiles
```

`final/<hardware_id>.json` is the authoritative performance artifact. It keeps
one row per Shape, separates paired resident results from provisional streamed
results, and defines every derived metric in the same file.

`intermediate/` is ignored by Git. It may contain raw samples and failed or
exploratory candidates needed during measurement, but none of those files is a
published score.
