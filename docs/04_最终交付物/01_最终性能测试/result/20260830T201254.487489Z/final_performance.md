# Final Performance Results

- Status: `completed`
- Resident task: `completed`
- Shape 14 task: `completed`
- Device: NVIDIA GeForce RTX 4080
- Compute capability: 8.9
- PyTorch / CUDA: 2.12.1+cu132 / 13.2
- Measurement preset: `final`
- Completed at: 2026-08-30T20:12:54.487489Z
- Speedup: baseline median latency / deployed median latency
- Shapes 01-13 geometric mean speedup: 18.0810x

## Shapes 01-13

| Shape | Baseline median (ms) | Deployed median (ms) | Deployed P90 (ms) | Speedup | Peak VRAM (GiB) | Correct |
| --- | ---: | ---: | ---: | ---: | ---: | :---: |
| official_01 | 4.241 | 0.274 | 0.286 | 15.455x | 0.04 | PASS |
| official_02 | 4.411 | 0.119 | 0.144 | 37.134x | 0.02 | PASS |
| official_03 | 4.422 | 0.140 | 0.159 | 31.524x | 0.02 | PASS |
| official_04 | 4.416 | 0.147 | 0.184 | 29.948x | 0.02 | PASS |
| official_05 | 4.015 | 0.427 | 0.634 | 9.403x | 0.02 | PASS |
| official_06 | 538.996 | 71.213 | 73.551 | 7.569x | 0.62 | PASS |
| official_07 | 4.243 | 0.141 | 0.143 | 30.024x | 0.02 | PASS |
| official_08 | 21.998 | 6.395 | 6.640 | 3.440x | 0.27 | PASS |
| official_09 | 3.832 | 0.306 | 0.386 | 12.517x | 0.02 | PASS |
| official_10 | 4.370 | 0.233 | 0.236 | 18.719x | 0.04 | PASS |
| official_11 | 8.072 | 0.299 | 0.301 | 26.995x | 0.03 | PASS |
| official_12 | 4.340 | 0.156 | 0.255 | 27.884x | 0.02 | PASS |
| official_13 | 116.234 | 3.698 | 3.969 | 31.430x | 0.11 | PASS |

## Shape 14

| Shape | Latency kind | Deployed median (ms) | Deployed P90 (ms) | Peak VRAM (GiB) | Correct |
| --- | --- | ---: | ---: | ---: | :---: |
| official_14 | end_to_end_distinct_microbatches | 19950.711 | 19950.711 | 3.32 | PASS |

Shape 14 uses the memory-efficient streamed path and is excluded from the Shapes 01-13 speedup geometric mean.
Its final/formal latency covers distinct streamed microbatches; smoke latency is explicitly reported as a model-compute estimate.
