# Final Performance Results

- Status: `completed`
- Resident task: `completed`
- Shape 14 task: `completed`
- Device: NVIDIA GeForce RTX 4080
- Compute capability: 8.9
- PyTorch / CUDA: 2.12.1+cu132 / 13.2
- Measurement preset: `final`
- Completed at: 2026-08-31T08:03:47.883417Z
- Speedup: baseline median latency / deployed median latency
- Shapes 01-13 geometric mean speedup: 14.4264x

## Shapes 01-13

| Shape | Baseline median (ms) | Deployed median (ms) | Deployed P90 (ms) | Speedup | Peak VRAM (GiB) | Correct |
| --- | ---: | ---: | ---: | ---: | ---: | :---: |
| official_01 | 1.848 | 0.218 | 0.220 | 8.474x | 0.03 | PASS |
| official_02 | 1.901 | 0.072 | 0.072 | 26.526x | 0.02 | PASS |
| official_03 | 1.880 | 0.082 | 0.083 | 22.950x | 0.02 | PASS |
| official_04 | 1.841 | 0.106 | 0.107 | 17.286x | 0.02 | PASS |
| official_05 | 3.918 | 0.425 | 0.450 | 9.219x | 0.02 | PASS |
| official_06 | 490.313 | 36.709 | 36.999 | 13.357x | 1.25 | PASS |
| official_07 | 1.858 | 0.074 | 0.074 | 25.201x | 0.01 | PASS |
| official_08 | 21.435 | 6.256 | 6.507 | 3.426x | 0.27 | PASS |
| official_09 | 1.435 | 0.202 | 0.204 | 7.114x | 0.03 | PASS |
| official_10 | 1.608 | 0.198 | 0.199 | 8.137x | 0.03 | PASS |
| official_11 | 7.913 | 0.243 | 0.245 | 32.608x | 0.03 | PASS |
| official_12 | 1.822 | 0.101 | 0.102 | 17.973x | 0.02 | PASS |
| official_13 | 120.457 | 3.305 | 3.452 | 36.442x | 0.20 | PASS |

## Shape 14

| Shape | Latency kind | Deployed median (ms) | Deployed P90 (ms) | Peak VRAM (GiB) | Local B=1 | Full B=32 path | Official B=32 I/O |
| --- | --- | ---: | ---: | ---: | :---: | :---: | :---: |
| official_14 | end_to_end_distinct_microbatches | 17257.926 | 17257.926 | 6.56 | PASS | COMPLETE | NOT AVAILABLE |

Shape 14 uses the memory-efficient streamed path and is excluded from the Shapes 01-13 speedup geometric mean.
Its final/formal latency covers distinct streamed microbatches; smoke latency is explicitly reported as a model-compute estimate.
The sampled execution digest is a compact reproducibility marker, not an official correctness oracle.
