# Figure Sources and QA

## Generation

```powershell
.\.venv\Scripts\python.exe docs\technical_report\figures\generate_figures.py
```

The script reads the latest completed final-performance JSON, `official/test_shapes.json`, and the checked-in deployment registry; it writes compact CSV source tables and exports every figure as editable SVG, PDF, and a 600 dpi PNG preview. The two search CSVs are compact extracts from the named observation cycles so the report does not depend on ignored runtime logs.

## Figure contracts

| Figure | Core conclusion | Archetype | Panels |
| --- | --- | --- | --- |
| `architecture_overview` | Typed program synthesis, isolated measurement, sequential promotion, and exact-device deployment form one closed loop; Shape 14 uses a separate streamed regime. | Schematic-led composite | Program construction; staged evidence; deployment/runtime |
| `performance_summary` | All 14 Shapes pass correctness; resident Shapes reach 14.49× geometric-mean speedup, with strong Shape-dependent variation. | Quantitative grid | Speedup; absolute latency; peak memory and Shape-14 feasibility |
| `workload_landscape` | The official parameter range and non-monotonic measured response motivate per-Shape search. | Asymmetric quantitative composite | Regime map; batch, width, head-count, and sequence slices |
| `useful_throughput` | Relative speedup and absolute useful-work throughput answer different questions; workload composition explains part of the difference. | Quantitative two-panel figure | Estimated TFLOP/s; speedup versus throughput with attention share |
| `deployed_program_matrix` | The 14 Shapes deploy 11 exact configurations and 10 displayed structural signatures rather than one universal policy. | Text-primary categorical matrix | Schedule; attention; dataflow/projections; FFN/norm; precision |
| `shape14_streaming` | Online microbatch execution makes Shape 14 feasible where even one dense `B=2` score tensor exceeds device capacity by orders of magnitude. | Schematic plus log-scale capacity comparison | Streamed execution; dense lower bounds; measured inner-forward peak |
| `search_evidence` | Four resident cycles reduce 3,933 Screen entries to 381 Enhanced entries, 50 Formal comparisons, and six updates; Screen dominates measured stage time. | Funnel plus stacked-duration chart | Aggregate evidence flow; one complete cycle's stage costs |

## Rendered QA notes

- Final width is 182.9 mm (double-column scale); all configured text is at least 5.6 pt and remains editable in SVG/PDF.
- Blue denotes resident/search behavior, teal denotes measured winners or execution, orange denotes promotion, reference emphasis, and the streamed Shape-14 regime, and gray is neutral.
- Quantitative comparison uses position or length on a common scale; logarithmic magnitude comparisons use points rather than bars.
- Color is redundant with text, marker shape, direct labels, or boundaries; no conclusion depends on hue alone.
- Grid lines are limited to major quantitative guides, and explanatory figures use direct labels instead of legend lookup where practical.
- Log axes are guarded against non-positive source values.
- P90 markers are labeled as percentiles, not uncertainty intervals.
- Shape 14 is absent from speedup panels because no dense baseline is measured.
- The 6.56 GiB Shape-14 value is the maximum allocated memory of one `B=2` inner forward, not a materialized full-`B=32` call and not CUDA reserved memory.
- Project-estimated TFLOP/s uses dominant matrix-operation FLOPs; it is not official MFU or a hardware-counter roofline measurement.
- Search counts are stage entries, not globally unique candidates; the figure does not combine incompatible study histories into a convergence claim.
- Sensitivity panels compare independently deployed programs and are not presented as causal Kernel ablations.
- PNG is retained only as a high-resolution preview. TIFF is intentionally omitted because the primary deliverables are editable vector figures and no raster-only journal submission target has been specified.
