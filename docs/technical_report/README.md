# Technical Report

## Learning-Guided Program Search for Shape-Specialized Transformers

This report describes a closed-loop GPU optimization system that synthesizes legal Transformer execution programs, evaluates them on the target device, and deploys only statistically supported per-Shape winners. The current validated platform is an NVIDIA GeForce RTX 4080 on native Windows.

## Report structure

1. [Problem framing and contributions](01_problem_and_contributions.md)
2. [System architecture](02_system_architecture.md)
3. [Search and optimization method](03_search_and_optimization_method.md)
4. [Evaluation and results](04_evaluation_and_results.md)
5. [Environment, AI tools, and limitations](05_environment_ai_tools_and_limitations.md)

The separation is deliberate: a reviewer can first understand the problem and the contribution, then inspect the executable architecture, the learning-guided search method, the measured evidence, and finally the disclosed environment, human–AI collaboration, and project boundaries.

## Headline result

On the validated RTX 4080 snapshot, Shapes 01–13 pass the supplied full comparator and achieve a **14.49× geometric-mean speedup** over the measured PyTorch baseline. Shape 14 passes a local `B=1` semantic-equivalence check and completes the full logical `B=32` streamed execution at **17.24 s** median latency and **6.56 GiB** peak allocated memory. Validation against the final official `B=32` input/output artifact remains pending; Shape 14 has no materialized dense baseline and is excluded from the speedup aggregate.

![RTX 4080 performance summary](figures/performance_summary.svg)

The numeric source is the timestamped [final performance artifact](../../result/20260831T083857.848038Z/final_performance.json). These are local engineering results, not an official competition score.

## Scope of this draft

This version intentionally does not contain a dedicated result-reproduction section. That submission component is reserved for a later owner-guided revision.

## Figure bundle

The ten report figures are rendered through the R/ggplot2 and Graphviz pipeline in [`figures/`](figures/). They cover the closed-loop architecture, official workload regimes and measured sensitivity, final performance, useful-work throughput, deployed program structure, all-resident mechanism-family ablations, Shape-14 streamed execution and capacity, and the observed search funnel. Each figure is available as editable SVG, PDF, and a high-resolution PNG preview. The compact CSV tables under `figures/source_data/` preserve the plotted values and their evidence boundaries.
