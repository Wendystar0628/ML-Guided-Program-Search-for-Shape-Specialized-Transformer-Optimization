# Final Performance Results

- Status: `completed`
- Resident task: `completed`
- Shape 14 task: `completed`
- Device: NVIDIA GeForce RTX 4080
- Compute capability: 8.9
- PyTorch / CUDA: 2.12.1+cu132 / 13.2
- Measurement preset: `final`
- Completed at: 2026-08-31T08:38:57.848038Z
- Speedup: baseline median latency / deployed median latency
- Shapes 01-13 geometric mean speedup: 14.4926x

## Shapes 01-13

| Shape | Baseline median (ms) | Deployed median (ms) | Deployed P90 (ms) | Speedup | Peak VRAM (GiB) | Correct |
| --- | ---: | ---: | ---: | ---: | ---: | :---: |
| official_01 | 1.727 | 0.218 | 0.219 | 7.918x | 0.03 | PASS |
| official_02 | 1.871 | 0.072 | 0.072 | 26.107x | 0.02 | PASS |
| official_03 | 1.866 | 0.082 | 0.083 | 22.784x | 0.02 | PASS |
| official_04 | 1.910 | 0.097 | 0.098 | 19.635x | 0.02 | PASS |
| official_05 | 3.905 | 0.421 | 0.447 | 9.277x | 0.02 | PASS |
| official_06 | 491.954 | 36.818 | 37.046 | 13.362x | 1.25 | PASS |
| official_07 | 1.793 | 0.074 | 0.074 | 24.326x | 0.01 | PASS |
| official_08 | 21.434 | 6.281 | 6.619 | 3.413x | 0.27 | PASS |
| official_09 | 1.513 | 0.202 | 0.203 | 7.498x | 0.03 | PASS |
| official_10 | 1.685 | 0.198 | 0.199 | 8.524x | 0.03 | PASS |
| official_11 | 7.942 | 0.243 | 0.245 | 32.726x | 0.03 | PASS |
| official_12 | 1.756 | 0.101 | 0.102 | 17.320x | 0.02 | PASS |
| official_13 | 120.430 | 3.337 | 3.468 | 36.087x | 0.20 | PASS |

## Shape 14

| Shape | Latency kind | Deployed median (ms) | Deployed P90 (ms) | Peak VRAM (GiB) | Correct |
| --- | --- | ---: | ---: | ---: | :---: |
| official_14 | end_to_end_distinct_microbatches | 17244.336 | 17244.336 | 6.56 | PASS |

Shape 14 uses the memory-efficient streamed path and is excluded from the Shapes 01-13 speedup geometric mean.
Its final/formal latency covers distinct streamed microbatches; smoke latency is explicitly reported as a model-compute estimate.
