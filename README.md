# Learning-Guided Program Search for Shape-Specialized Transformers

**TikTok TechJam 2026 · Hardware Efficiency**

A closed-loop optimizer synthesizes legal Transformer execution programs, measures them on the target GPU, and deploys one winner per workload Shape. On the validated NVIDIA GeForce RTX 4080, Shapes 01–13 pass the supplied comparator and achieve a **14.49× equal-Shape geometric-mean speedup** over the official PyTorch baseline.

[Three-minute video](https://youtu.be/rItQ3x4iHBc) · [Interactive walkthrough](https://wendystar0628.github.io/ML-Guided-Program-Search-for-Shape-Specialized-Transformer-Optimization/) · [Technical report](docs/technical_report/README.md) · [Machine-readable result](result/20260831T083857.848038Z/final_performance.json) · [Quick reproduction](#quick-reproduction)

## Verified results

| Evidence | Result |
| --- | ---: |
| Shapes 01–13 correctness | **13/13 PASS** with the supplied comparator |
| Shapes 01–13 geometric-mean speedup | **14.49×** |
| Resident speedup range | **3.41×–36.09×** |
| Shape 14 correctness evidence | **Local `B=1` semantic PASS; official `B=32` I/O pending** |
| Shape 14 full logical `B=32` streamed latency | **17.24 s** (one timed sample) |
| Shape 14 streamed inner-forward peak allocation | **6.56 GiB** at `B=2` |

![RTX 4080 performance summary](docs/technical_report/figures/performance_summary.svg)

For Shapes 01–13, speedup is `baseline median / deployed median`; the headline value is the geometric mean of the 13 per-Shape ratios, giving each Shape equal weight in log-speedup space. Shape 14 is excluded because no dense `S × S` baseline was executed.

### Shape 14: streamed execution

![Shape 14 streamed execution](docs/technical_report/figures/shape14_streaming.svg)

Shape 14 (`B=32`, `S=100,000`, `D=1,024`, 16 heads, two layers) runs as sixteen ordered `B=2` inner forwards. A tiled causal-attention Kernel maintains online softmax statistics, emits each output chunk, and discards score tiles instead of materializing a global `[B, H, S, S]` tensor.

The measured 17.244 s covers one complete 16-chunk loop. The 6.56 GiB value is the maximum allocated memory of one `B=2` inner forward, not a materialized `B=32` peak. A local `B=1` semantic check passes; the official `B=32` input/output pair is unavailable, so no official full-batch correctness or dense-baseline speedup is claimed. Full per-Shape values and protocol boundaries are in the [evaluation report](docs/technical_report/04_evaluation_and_results.md).

## Architecture and method

![Closed-loop architecture](docs/technical_report/figures/architecture_overview.svg)

The runtime follows one loop: **construct → reject illegal plans → measure → compare → register the approved winner → execute**. Its three central design choices are:

1. **Typed executable-program search.** A `ConfigSpec` covers operators, layouts, precision boundaries, fusions, runtimes, and schedules. `PlanBuilder` rejects illegal combinations before GPU work.
2. **Learning-guided hardware evidence.** Shapes 01–13 use persistent branch-local conditional TPE and fixed-budget survivor racing. Screen, Enhanced, and Formal stages spend progressively stronger measurement only on increasingly competitive candidates.
3. **Separate capacity regime for Shape 14.** A finite no-replacement search selects bounded streamed programs without forcing the 100,000-token workload through the resident search lifecycle.

The method transfers three established principles:

- **TPE density-ratio search:** compatible branches favor proposals with high estimated `l(x)/g(x)` under measured feasibility constraints.
- **Fixed-budget best-arm identification:** broad inexpensive evidence is narrowed toward promising branches while retaining a small exploration reserve.
- **Group-sequential paired promotion:** alternating incumbent/challenger blocks allow clear wins to promote after 6 or 9 blocks; close decisions use up to 13. The pre-specified three-look sign-test construction has a conservative per-comparison false-promotion bound below 0.05 under its documented null and independence assumptions.

Deployment entries are scoped to an exact device, software environment, and Shape. An unmatched environment uses a portable fallback rather than inheriting the RTX 4080 result. The [search report](docs/technical_report/03_search_and_optimization_method.md) gives the equations, implementation correspondence, and theoretical boundaries.

## What was deployed and why it helped

![Shape-specialized deployed-program matrix](docs/technical_report/figures/deployed_program_matrix.svg)

Each row is a measured composition of runtime, attention, layout, projection, FFN, normalization/fusion, and precision choices—not a manually named policy.

![Complete deployment speedup and retained performance after mechanism-family removal across Shapes 01–13](docs/technical_report/figures/component_ablation.svg)

The right panel reports `retained performance = ablated speedup / complete speedup`. Values below 100% indicate a loss after removing one legal mechanism family; values above 100% mean the fallback was faster in this compact run. These are non-additive removal sensitivities because runtime, layout, precision, and fusion interact. Runtime scheduling is the broadest dependency, norm/boundary specialization is the next most consistent, and attention or projection/precision becomes decisive only for particular Shapes. The [evaluation report](docs/technical_report/04_evaluation_and_results.md#44-coherent-mechanism-family-ablation) documents the protocol and coupled cases.

## Quick reproduction

The validated path uses Windows 11 PowerShell, Python 3.12, an NVIDIA RTX 4080-class CUDA GPU, CUDA Toolkit 13.2, and Visual Studio C++ Build Tools. PyTorch, Triton, and compiled paths build lazily on first use; compilation is excluded from timed samples.

```powershell
git clone https://github.com/Wendystar0628/Learning-Guided-Program-Search-for-Shape-Specialized-Transformers.git
cd Learning-Guided-Program-Search-for-Shape-Specialized-Transformers

py -3.12 -m venv .venv
.\environments\activate_windows_rtx4080.ps1
python -m pip install -r environments\windows-rtx4080.txt
```

Run tests and the low-cost pipeline check, then reproduce the declared measurement:

```powershell
python -m pytest -q
.\.venv\Scripts\python.exe scripts\run_final_performance.py --preset smoke
.\.venv\Scripts\python.exe scripts\run_final_performance.py --preset final
```

Shapes run serially in fresh processes under one project-local GPU lease. Close unrelated GPU workloads before the final run. Results are appended to:

```text
result/<UTC completion time>/
  final_performance.json
  final_performance.md
```

A matching environment should remain near the declared result, although clocks, temperature, drivers, and background activity affect latency. Other device signatures intentionally use the portable fallback. See the [result guide](result/README.md) for presets and output fields.

## Repository guide

```text
solution/       typed programs, plan compiler, operators, kernels, runtimes
autotune/       conditional search, staged evidence, promotion, outer loop
benchmarking/   correctness, timing, profiling, hardware and GPU isolation
deployment/     exact-environment identity and deployed local winners
official/       supplied benchmark semantics and 14 official Shapes
scripts/        optimization and final-result entrypoints
result/         timestamped final-performance and ablation artifacts
tests/          tests grouped by production responsibility
docs/           technical report and official-material translations
```

`observations/` stores local study and diagnostic history and is intentionally excluded from GitHub.

## Report, AI disclosure, and deliverables

The English [technical report](docs/technical_report/README.md) covers problem framing, architecture, search mathematics, measurement, environment, AI collaboration, and limitations.

| Item | Used in this project |
| --- | --- |
| Core libraries | PyTorch, Triton, Optuna, Ninja |
| Supplied assets | Official PyTorch benchmark semantics and 14 workload Shapes |
| External dataset or hosted runtime API | None; inputs come from the supplied benchmark path |
| AI tools and models | OpenAI Codex; ChatGPT GPT-5.6 sol Pro; Deep Research |
| Actually used Skills | Stop That Shit, Deep Research, Browser Control, Nature Figure |

The human participant set the problem framing, architecture directions, constraints, priorities, and acceptance decisions. AI tools researched and adapted those directions, implemented alternatives, ran the bounded workflows, and summarized measured feedback. Deterministic project code performed correctness, timing, promotion, and deployment. The report records the detailed ownership model and two [representative interaction histories](docs/technical_report/05_environment_ai_tools_and_limitations.md#56-representative-interaction-histories).

Competition deliverables:

- source code plus setup, compilation, and run instructions: this repository;
- written technical report: [`docs/technical_report/`](docs/technical_report/README.md);
- timestamped result evidence: [`result/`](result/README.md);
- AI tools, models, Skills, human guidance, and interactions: [technical report §5](docs/technical_report/05_environment_ai_tools_and_limitations.md);
- public three-minute demo: [YouTube](https://youtu.be/rItQ3x4iHBc);
- Devpost project page: to be linked after publication.

## Limitations and next steps

- Results cover one native-Windows RTX 4080 and the competition forward Transformer core, not training, KV-cache decoding, distributed execution, or production serving.
- Approximately 12 hours of search explored only part of the combinatorial program space; deployed plans are best-so-far results, not proven global optima.
- The report includes legal one-family removal tests, not a full combinatorial interaction decomposition.
- Shape 14 still requires the final official `B=32` input/output oracle, and the project has not learned a transferable cross-device performance model.

The highest-value next steps are official Shape 14 validation, longer best-so-far search studies, new profiled Kernel families, and multi-device evidence for a transferable cost model.

<details>
<summary><strong>Optional autotuning workflow</strong></summary>

These commands search for new local winners and are not required to reproduce the declared result:

```powershell
# Shapes 01–05 and 07–13; add -IncludeShape06 for the expensive large-batch case
.\scripts\optimize_shapes_01_13.ps1

# Independent bounded Shape-14 search
.\scripts\optimize_shape_14.ps1
```

Compatible studies resume from persistent local evidence. Use `python cli.py --help` for lower-level search, benchmark, profile, and probe commands.

Run report-facing component ablation separately; it never updates the deployment registry:

```powershell
.\.venv\Scripts\python.exe scripts\run_component_ablation.py
```

</details>
