# Figure Sources and QA

## Generation

```powershell
.\.venv\Scripts\python.exe docs\technical_report\figures\generate_figures.py
```

The script reads the latest completed final-performance JSON and
`official/test_shapes.json`, writes compact CSV source tables, and exports every
figure as editable SVG, PDF, and a 600 dpi PNG preview.

## Figure contracts

| Figure | Core conclusion | Archetype | Panels |
| --- | --- | --- | --- |
| `architecture_overview` | Typed program synthesis, isolated measurement, sequential promotion, and exact-device deployment form one closed loop; Shape 14 uses a separate streamed regime. | Schematic-led composite | Program construction; staged evidence; deployment/runtime |
| `performance_summary` | All 14 Shapes pass correctness; resident Shapes reach 14.49× geometric-mean speedup, with strong Shape-dependent variation. | Quantitative grid | Speedup; absolute latency; peak memory and Shape-14 feasibility |
| `workload_landscape` | The official parameter range and non-monotonic measured response motivate per-Shape search. | Asymmetric quantitative composite | Regime map; batch, width, head-count, and sequence slices |

## Rendered QA notes

- Final width is 182.9 mm (double-column scale); all configured text is at least
  5.6 pt and remains editable in SVG/PDF.
- Blue denotes resident/search behavior, teal denotes measured winners or execution,
  orange denotes promotion and the streamed Shape-14 regime, and gray is neutral.
- Log axes are guarded against non-positive source values.
- P90 markers are labeled as percentiles, not uncertainty intervals.
- Shape 14 is absent from speedup panels because no dense baseline is measured.
- Sensitivity panels compare independently deployed programs and are not presented as
  causal Kernel ablations.
- PNG is retained only as a high-resolution preview. TIFF is intentionally omitted
  because the primary deliverables are editable vector figures and no raster-only
  journal submission target has been specified.
