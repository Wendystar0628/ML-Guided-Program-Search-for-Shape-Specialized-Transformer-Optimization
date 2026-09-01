# 4. Evaluation and Results

## 4.1 Headline result

On the validated RTX 4080, Shapes 01–13 all pass the supplied comparator and achieve a **14.49× equal-Shape geometric-mean speedup**, ranging from 3.41× to 36.09×. Shape 14 completes one logical `B=32` streamed loop in 17.244 s with a 6.56 GiB `B=2` inner-forward peak; local `B=1` semantics pass, while official `B=32` I/O validation remains pending.

The declared artifact is dated `2026-08-31T08:38:57.848038Z` and uses FP32 input/output, `padding_ratio=0.0`, and `input_scale=1.0` on the [disclosed environment](05_environment_ai_tools_and_limitations.md#51-validated-runtime-environment).

![RTX 4080 performance summary](figures/performance_summary.svg)

*Figure 6. Shapes 01–13 achieve a 14.49× geometric-mean speedup; absolute latency distinguishes relative acceleration from execution time.*

## 4.2 Metrics and measurement protocol

For resident Shape (i), speedup and the equal-Shape aggregate are

\[
s_i = \frac{\text{baseline median latency}_i}
           {\text{deployed median latency}_i},
\qquad
G = \exp\left(\frac{1}{13}\sum_{i=1}^{13}\log s_i\right)=14.4926.
\]

The geometric mean gives every Shape equal weight in log-speedup space; per-Shape values remain necessary because one aggregate cannot describe workload variability ([Fleming and Wallace, 1986](https://doi.org/10.1145/5666.5673)).

P90 is a percentile of deployed timing samples, not a confidence interval. Peak memory is maximum allocated GPU memory reported by the project. Shapes 01–13 use the supplied elementwise absolute-or-relative comparator.

The resident final preset normally uses five correctness trials, 20 warmups, 100 repeats, and three rounds. Shape 06 uses one correctness trial, two warmups, five repeats, and three rounds because its very large batch makes repeated validation disproportionately expensive. Shape 14 uses one local `B=1` semantic trial and one timed full-logical-batch run with no additional warmup.

Measurements use CUDA Events, fixed inputs per comparison, one GPU lease, and a fresh process per Shape. Baseline/candidate and incumbent/challenger comparisons alternate order where applicable. Compilation and first-run work are excluded from timed samples.

The report also estimates dominant matrix-operation work:

\[
\widehat{F}=L\left(2BS^2D+8BSD^2+4BSDF\right),\qquad
\widehat{T}=\frac{\widehat{F}}{10^9t_{ms}}.
\]

This estimate covers the principal attention, projection, and FFN matrix multiplications; it is not executed FLOPs, a hardware-counter measurement, roofline efficiency, or official MFU.

## 4.3 Resident Shapes 01–13

| Shape | Baseline median (ms) | Deployed median (ms) | Deployed P90 (ms) | Speedup | Peak memory (GiB) | Correctness |
| ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 01 | 1.7271 | 0.2181 | 0.2191 | 7.92× | 0.027 | PASS |
| 02 | 1.8714 | 0.0717 | 0.0717 | 26.11× | 0.020 | PASS |
| 03 | 1.8665 | 0.0819 | 0.0829 | 22.78× | 0.021 | PASS |
| 04 | 1.9101 | 0.0973 | 0.0983 | 19.64× | 0.016 | PASS |
| 05 | 3.9045 | 0.4209 | 0.4466 | 9.28× | 0.019 | PASS |
| 06 | 491.9541 | 36.8179 | 37.0465 | 13.36× | 1.251 | PASS |
| 07 | 1.7935 | 0.0737 | 0.0737 | 24.33× | 0.012 | PASS |
| 08 | 21.4340 | 6.2807 | 6.6192 | 3.41× | 0.266 | PASS |
| 09 | 1.5126 | 0.2017 | 0.2029 | 7.50× | 0.027 | PASS |
| 10 | 1.6846 | 0.1976 | 0.1987 | 8.52× | 0.027 | PASS |
| 11 | 7.9422 | 0.2427 | 0.2448 | 32.73× | 0.027 | PASS |
| 12 | 1.7558 | 0.1014 | 0.1024 | 17.32× | 0.016 | PASS |
| 13 | 120.4296 | 3.3372 | 3.4683 | 36.09× | 0.199 | PASS |

Shape 08 is the weakest relative result at 3.41× and remains the clearest resident optimization target; Shapes 13 and 11 lead at 36.09× and 32.73×.

![Project-estimated useful throughput](figures/useful_throughput.svg)

*Figure 7. Relative speedup and project-estimated useful throughput identify different optimization regimes.*

Shape 08 has the lowest speedup but the highest resident project-estimated useful throughput (about 67.0 TFLOP/s). Shape 02 reaches 26.11× yet only about 1.64 estimated TFLOP/s because its absolute workload is small. Relative acceleration and useful-work scale are therefore complementary, not interchangeable.

## 4.4 Coherent mechanism-family ablation

For each resident Shape, a counterfactual replaces one mechanism family in the deployed `ConfigSpec` with its nearest legal fallback, recompiles the plan, checks correctness and execution-path identity, and measures the pair on the same input in a fresh process.

![Complete deployment speedup and mechanism-family ablations](figures/component_ablation.svg)

*Figure 8. The right panel reports retained performance, (S_{ablated}/S_{complete}=L_{deployed}/L_{ablated}). Values below 100% indicate loss after removal; values above 100% mean the fallback was faster. Gray cells are inapplicable, coupled, or capacity-excluded. Asterisks mark dependency closures and plus signs mark partial isolation.*

The compact protocol uses one correctness trial, two warmups, five timed calls, and five alternating paired rounds; Shape 06 uses three rounds. All 56 executable counterfactuals passed correctness and path checks. Coupled mechanisms were not presented as independent effects, and the unsafe Shape-06 runtime-off case was capacity-excluded.

The main pattern is:

- runtime scheduling is the broadest dependency, especially for small and medium Shapes;
- norm/boundary specialization is the next most consistent mechanism;
- attention and projection/precision have strong Shape-specific effects, while isolated FFN and layout removals are usually smaller under the deployed compositions.

These leave-one-family-out sensitivities are non-additive. A complete interaction decomposition would require a larger factorial or Shapley-style experiment.

## 4.5 Shape 14 streamed result

| Evidence | Result |
| --- | ---: |
| Full logical `B=32` latency | **17.244 s**, one timed sample |
| `B=2` inner-forward peak allocation | **6.56 GiB** |
| Local semantic check | **`B=1` PASS** |
| Official `B=32` I/O validation | **Pending; artifact unavailable** |
| Dense-baseline speedup | **Not measured** |

The logical batch executes as 16 ordered, distinct `B=2` forwards. Tiled causal attention retains online softmax statistics and emitted output chunks rather than a global `S × S` score tensor. The sampled digest records execution reproducibility; it is not a correctness oracle.

The project estimates `1,391,250,636,800,000` model FLOPs for the logical workload, corresponding to approximately **80.7 project-estimated TFLOP/s** at the measured latency.

![Shape 14 streamed execution](figures/shape14_streaming.svg)

*Figure 9. Ordered microbatches emit output chunks while discarding causal score tiles.*

![Shape 14 capacity evidence](figures/shape14_capacity.svg)

*Figure 10. The measured streamed inner-forward allocation fits the validated device; analytical dense-score lower bounds exceed its capacity by orders of magnitude.*

The analytical bounds establish why streaming is necessary; they are not dense-baseline latency measurements.

## 4.6 Evidence boundaries

This snapshot supports Shape-specialized acceleration on the disclosed RTX 4080 and memory-feasible streamed execution for Shape 14. It does not establish official MFU or score, cross-hardware performance, confidence intervals across independent machine runs, or official Shape-14 `B=32` correctness.

## 4.7 Sources

- Machine-readable result: [final_performance.json](../../result/20260831T083857.848038Z/final_performance.json)
- Human-readable result: [final_performance.md](../../result/20260831T083857.848038Z/final_performance.md)
- Raw mechanism-family ablation: [ablation.json](../../result/ablation/20260901T002918.287640Z/ablation.json)
- Figure source data: [`figures/source_data/`](figures/source_data/)
