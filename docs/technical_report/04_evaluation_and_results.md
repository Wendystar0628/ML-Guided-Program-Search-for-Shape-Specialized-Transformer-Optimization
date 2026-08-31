# 4. Evaluation and Results

## 4.1 Declared snapshot

This section reports the completed local artifact dated `2026-08-31T08:38:57.848038Z`. The test variant uses FP32 input/output, `padding_ratio=0.0`, and `input_scale=1.0`. All GPU work was run through the project's exclusive device lease and fresh-process Shape suite.

![RTX 4080 performance summary](figures/performance_summary.svg)

## 4.2 Metrics

For Shapes 01–13, per-Shape speedup is

\[
s_i = \frac{\text{baseline median latency}_i}
           {\text{deployed median latency}_i}.
\]

The headline aggregate is the equal-Shape geometric mean

\[
G = \exp\left(\frac{1}{13}\sum_{i=1}^{13}\log s_i\right)=14.4926.
\]

P90 is a latency percentile from the deployed timing samples; it is not an error bar or confidence interval. Peak memory is the project-reported maximum allocated GPU memory. Correctness uses the official elementwise absolute-or-relative tolerance logic. This aggregate is an internal engineering summary, not the unpublished official MFU-weighted competition score.

To complement relative speedup with useful-work scale, the report also uses the project's dominant-matrix-operation estimate

\[
\widehat{F}=L\left(2BS^2D+8BSD^2+4BSDF\right),\qquad
\widehat{T}=\frac{\widehat{F}}{10^9t_{ms}}.
\]

This estimate covers the principal attention, projection, and FFN matrix multiplications. It excludes elementwise work and is not executed FLOPs, hardware-counter throughput, roofline efficiency, or official MFU.

## 4.3 Per-Shape results

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
| 14 | — | 17244.3359 | 17244.3359 | — | 6.563 | PASS |

Shapes 01–13 all pass and produce a **14.49×** geometric mean. The largest measured speedups are Shape 13 (36.09×), Shape 11 (32.73×), and Shape 02 (26.11×). Shape 08 is the lowest at 3.41×, making it a more important optimization target than Shapes already dominated by launch-overhead removal or a highly favorable specialized path.

![Project-estimated useful throughput](figures/useful_throughput.svg)

The throughput view prevents a high speedup from being mistaken for high absolute device utilization. Shape 08 has the lowest relative speedup but the highest resident project-estimated useful throughput (about 67.0 TFLOP/s); Shape 02 has a large 26.11× speedup yet only about 1.64 estimated TFLOP/s because its absolute workload is small. The attention-share encoding also separates attention-heavy Shapes 13 and 14 from projection/FFN-dominated regimes.

## 4.4 Shape 14

Shape 14 does not materialize the dense `S × S` reference baseline. The full logical-batch latency loop executes 16 ordered, distinct `B=2` streamed forwards and retains only a compact summary between chunks. The measured 17.244 s covers that complete logical-batch loop and passes correctness. The reported 6.56 GiB is the maximum allocated memory of one `B=2` inner streamed forward; it excludes allocator-reserved memory and is not a measurement of one materialized `B=32` call. Shape 14 is excluded from the speedup geometric mean.

The result artifact records a project estimate of `1,391,250,636,800,000` model FLOPs. Dividing this estimate by the measured latency gives approximately **80.7 project-estimated TFLOP/s**. This is not official MFU: the official useful-FLOP definition, device denominator, Shape weights, and bandwidth correction have not been published.

![Shape 14 streamed-capacity evidence](figures/shape14_streaming.svg)

For scale, a single FP32 dense attention-score tensor at `B=2` would require at least 1,192 GiB; at the full logical `B=32`, it would require at least 19,073 GiB. The 1,192 GiB lower bound is about 182 times the measured 6.56 GiB inner-forward peak, before accounting for QKV, outputs, FFN state, and temporary buffers. This is capacity evidence for online/streamed attention, not a dense-baseline latency comparison.

## 4.5 Measurement protocol interpretation

The resident final preset normally uses five correctness trials, 20 warmups, 100 repeats, and three rounds. Shape 06 deliberately uses one correctness trial, two warmups, five repeats, and three rounds because its very large batch makes repeated validation disproportionately expensive. Shape 14 uses one correctness trial, no additional warmup, one full-batch repeat, and one round in the final reporting script; its P90 therefore equals its median by construction and is not tail-latency evidence.

The project measures with CUDA Events, fixed inputs per comparison, alternating baseline/candidate order where applicable, one GPU lease, and a fresh process per Shape. Previous development runs showed that external GPU activity can materially distort sub-millisecond measurements. The result above should therefore be read as a declared exclusive-device snapshot, not averaged with older, differently loaded runs.

## 4.6 What the evidence supports

The results support three claims:

1. shape-specialized execution materially outperforms the supplied PyTorch reference across all resident workloads on the disclosed RTX 4080 stack;
2. no single speedup trend explains batch, width, head-count, and sequence variation;
3. a streamed implementation makes Shape 14 feasible without a dense attention matrix.

The current artifact does not support resident MFU, a cross-hardware performance claim, a confidence interval across independent machine runs, or an official score.

## 4.7 Source

- Machine-readable result: [final_performance.json](../../result/20260831T083857.848038Z/final_performance.json)
- Human-readable result: [final_performance.md](../../result/20260831T083857.848038Z/final_performance.md)
- Figure source data: [`figures/source_data/`](figures/source_data/)
