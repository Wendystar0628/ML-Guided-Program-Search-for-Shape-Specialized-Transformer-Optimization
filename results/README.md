# Performance Reference

`final/nvidia_geforce_rtx_4080.json` is the best recorded RTX 4080 result and
the comparison baseline for future searches. Its measurements come from Git
commit `c12893d6198bfe5d5aaa6b740565bca86ee25cee`; the case records now use the
current `BenchmarkResult` field layout and current generated `ConfigSpec`
identifiers.

Single-run results keep only identity, correctness, latency, memory, and whether
the requested execution path ran. Full configurations remain in
`deployments/deployed_configs.json` instead of being copied into every result.

Shapes 1–13 are paired Formal measurements. Shape 14 is explicitly marked as a
provisional streamed measurement in the JSON. Fields that the historical run
did not collect are `null`, rather than reconstructed from old Policy labels.
