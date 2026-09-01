# Technical Report

## Learning-Guided Program Search for Shape-Specialized Transformers

This report presents a closed-loop GPU optimizer that constructs legal Transformer execution programs, measures them on the target device, and deploys statistically supported per-Shape winners.

## Headline result

On the validated native-Windows RTX 4080, Shapes 01–13 pass the supplied comparator and achieve a **14.49× geometric-mean speedup** over the measured PyTorch baseline. Shape 14 passes a local `B=1` semantic check and completes one logical `B=32` streamed execution in **17.24 s** with a **6.56 GiB** `B=2` inner-forward peak. Official `B=32` I/O validation remains pending, and Shape 14 is excluded from the speedup aggregate.

![RTX 4080 performance summary](figures/performance_summary.svg)

The numeric source is the timestamped [final performance artifact](../../result/20260831T083857.848038Z/final_performance.json). These are local engineering results, not an official competition score.

## Report structure

1. [Problem framing and contributions](01_problem_and_contributions.md)
2. [System architecture](02_system_architecture.md)
3. [Search and optimization method](03_search_and_optimization_method.md)
4. [Evaluation and results](04_evaluation_and_results.md)
5. [Environment, AI collaboration, and limitations](05_environment_ai_tools_and_limitations.md)

Setup, installation, and reproduction commands are in the repository [README](../../README.md#quick-reproduction). Editable figures and source data are under [`figures/`](figures/).
