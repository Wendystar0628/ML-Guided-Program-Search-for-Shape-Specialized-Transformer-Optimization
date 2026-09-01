# Figure Sources and QA

## Generation

The checked-in CSV files are the compact evidence layer used by the report. Refresh the performance, workload, deployment, and Shape-14 memory tables only when the declared final-performance snapshot changes:

```powershell
.\.venv\Scripts\python.exe docs\technical_report\figures\prepare_figure_data.py
```

Refresh the component-family table only from a completed evidence run. This command serializes all resident Shapes 01–13 behind the project GPU lease and writes both the immutable raw JSON and the compact plotting table:

```powershell
.\.venv\Scripts\python.exe scripts\run_component_ablation.py --figure-csv docs\technical_report\figures\source_data\component_ablation.csv
```

Install the minimal R dependencies once, then render the complete figure bundle:

```powershell
& 'C:\Program Files\R\R-4.6.1\bin\x64\Rscript.exe' docs\technical_report\figures\install_dependencies.R
& 'C:\Program Files\R\R-4.6.1\bin\x64\Rscript.exe' docs\technical_report\figures\render_all.R
```

R/ggplot2 owns the quantitative figures, the system architecture, composition, and vector export. Graphviz is retained only for the Shape-14 streamed-execution schematic and is invoked by the R driver. Python only refreshes source tables; it does not draw or lay out figures.

## Figure contracts

| Figure | Core conclusion | Visual form |
| --- | --- | --- |
| `architecture_overview` | Typed program construction, multi-fidelity evidence, promotion, deployment, and runtime resolution form one closed loop. | Layered R system schematic with a single evidence-feedback loop |
| `performance_summary` | Resident Shapes reach 14.49× geometric-mean speedup, while absolute latency remains strongly Shape-dependent. | Speedup bars and baseline-to-deployed latency dumbbells |
| `workload_landscape` | The official workloads occupy distinct launch, throughput, working-set, and capacity regimes. | Log-log regime map |
| `workload_sensitivity` | Independently deployed plans respond non-monotonically across key Shape dimensions. | Four discrete point panels without interpolating lines |
| `useful_throughput` | Relative speedup and estimated useful throughput answer different performance questions. | Per-Shape lollipop and speedup-throughput scatter |
| `deployed_program_matrix` | The 14 Shapes resolve to materially different complete programs rather than one universal policy. | Single-panel categorical matrix across seven structural decision axes |
| `component_ablation` | Complete speedup and performance retained after mechanism removal differ by Shape; family effects are not additive shares. | Full-speedup log bars beside a retained-performance percentage heatmap |
| `shape14_streaming` | Online tiled attention avoids a global score tensor and makes Shape 14 runnable. | Graphviz dense-invalid versus streamed-valid lanes |
| `shape14_capacity` | The measured streamed inner-forward peak fits the RTX 4080 while dense analytical bounds do not. | Horizontal log-scale capacity plot |
| `search_evidence` | Four resident cycles aggressively narrow evidence while Screen dominates measured stage time. | Log retention plot and 100% stage-time bars |

## Visual contract

- Primary navy `#356C95` identifies resident/search evidence; teal `#1F8A78` identifies deployed or measured execution; orange `#D9772B` is reserved for Shape 14 and true decision exceptions; gray is neutral context.
- Text uses Arial. Most figures use an 8–9 pt report-width baseline; the dense deployment matrix uses 6.0–6.8 pt labels and remains above the 5 pt publication floor.
- Only left and bottom quantitative axes remain. Grids are limited to major guides, and labels use dedicated space or deterministic avoidance.
- Color is redundant with direct text, marker shape, fill state, or boundaries; no conclusion depends on hue alone.
- SVG is the primary editable asset, PDF is the print vector asset, and PNG is a high-resolution preview.
- Composite figures were split when a single 760–1100 px Markdown viewport could not preserve readable text and clean panel gutters.

## Evidence boundaries

- P90 markers are percentiles, not uncertainty intervals.
- Shape 14 is absent from speedup panels because no dense baseline is measured.
- The 6.56 GiB Shape-14 value is the maximum allocated memory of one `B=2` inner forward, not a materialized full-`B=32` call and not CUDA reserved memory.
- Project-estimated TFLOP/s uses dominant matrix-operation FLOPs; it is not official MFU or a hardware-counter roofline measurement.
- Search counts are stage entries, not globally unique candidates; the figure does not combine incompatible study histories into a convergence claim.
- Sensitivity panels compare independently deployed programs and are descriptive rather than causal Kernel ablations.
- Component ablations report \(S_{ablated}/S_{complete}=L_{deployed}/L_{ablated}\). Below 100% means removal reduced performance; above 100% means the legal fallback was faster. These are legal family-level counterfactuals, not an additive causal decomposition. `*` marks a required dependency closure, `+` marks partial isolation, `CAP` marks an unsafe capacity counterfactual, `CPL` marks an active but inseparable family, and `N/A` means the deployed plan already uses the fallback.
