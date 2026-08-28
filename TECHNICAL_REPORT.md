# Shape-Aware Transformer Optimization on an RTX 4080

## Executive summary

One Transformer implementation is not fastest for every published shape. Small
shapes are dominated by launch and framework overhead, wide shapes benefit from
Tensor Core execution, very large batches expose residual-normalization traffic,
and the extreme long-sequence shape cannot materialize dense attention. This
project therefore treats optimization as a measured routing problem:

1. profile the real device and classify each workload;
2. rank a small, explicit set of eligible execution compositions;
3. reject candidates that fail correctness or do not execute the requested path;
4. let Formal GPU measurements select the winner; and
5. dispatch deterministically at runtime with no online search.

On the recorded NVIDIA GeForce RTX 4080 stack, all 13 resident shapes pass the
official-style comparator. Their unweighted geometric-mean speedup is **10.98x**.
The independent Shape 14 path also runs the complete logical workload without a
dense `S x S` attention matrix: its measured Target median is **15,987.85 ms** at
**87.02 useful TFLOP/s**, with **6.93 GiB** peak device allocation. Shape 14 is
reported separately because its current correctness reference is provisional
and no executable dense Baseline result exists.

The machine-readable evidence is bound to the current workload, official
snapshot, Solution implementation, route table, runtime and measurement
protocol by the [RTX 4080 verified bundle](verified_hardware/nvidia_geforce_rtx_4080/).

## 1. Environment and measurement protocol

### 1.1 Recorded platform

| Item | Recorded value |
| --- | --- |
| OS / driver model | Windows 11 / WDDM |
| GPU | NVIDIA GeForce RTX 4080, Ada, compute capability 8.9 |
| GPU resources | 76 SMs, 16 GiB VRAM, 64 MiB L2, 256-bit memory bus |
| Theoretical memory bandwidth | 716.864 GB/s |
| Driver | 610.88 |
| Python | 3.12.5 |
| PyTorch / CUDA runtime | 2.12.1+cu132 / CUDA 13.2 |
| cuDNN / Triton | 9.20.0 / 3.7.1 |
| Runtime math policy | `matmul_precision=high`, TF32 allowed |

The current [hardware profile](verified_hardware/nvidia_geforce_rtx_4080/profile.json)
also records measured anchors: 93.73 FP16 TFLOP/s, 93.91 BF16 TFLOP/s,
47.22 FP32/TF32 TFLOP/s, 270.88 GB/s bounded device-copy bandwidth, and
8.02 microseconds eager launch latency. These are local runtime anchors, not
vendor peak specifications.

### 1.2 Resident protocol: Shapes 1–13

- External input, output and reference semantics: FP32.
- Correctness: five seeded trials, `rtol=0.02`, `atol=0.002`.
- Timing: 20 warm-up iterations, 100 repeats per round, three rounds.
- Statistic: CUDA Event median and P90 of the complete Transformer forward.
- Comparison: independent Baseline and Solution models with identical weights
  and inputs, alternated within the same isolated worker process.
- Compilation and first-use setup are outside the timed region.
- Every reported specialized policy must provide observed execution evidence;
  a silent fallback is not counted as that candidate.

The stored result is the [Formal resident reference](verified_hardware/nvidia_geforce_rtx_4080/results/reference_formal.json).
It binds Solution implementation `07882d07...` and route table `abe29c3f...`.

### 1.3 Streamed protocol: Shape 14

Shape 14 has logical shape `B=32, S=100000, D=1024, H=16, L=2`. It uses an
independent Target-only protocol because a dense reference attention matrix and
the full resident logical batch do not fit this device:

- correctness is checked on a complete `B=1` long-sequence sample against an
  internal query-block reference;
- eligible attention policies and microbatch divisors are screened on the GPU;
- the selected schedule executes all 32 samples as one logical workload;
- three complete rounds are timed after two warm-up rounds;
- GPU Target latency and host-streamed end-to-end latency are both retained.

The stored [Shape 14 reference](verified_hardware/nvidia_geforce_rtx_4080/results/reference_streamed.json)
is explicitly marked `provisional` and `target_only`. It does not invent a dense
Baseline latency or a speedup.

### 1.4 Performance metrics

`Achieved TFLOP/s` uses the project's useful matmul FLOP count divided by Target
latency. `Project MFU` estimates segmented ideal compute time from measured
saturated GEMM roofs matched to the observed linear and attention compute
dtypes, then divides it by measured Target time. It is a diagnostic estimate,
not an official score and not a vendor peak utilization claim.

The result schema also contains a logical operator-traffic estimate. It is not
measured DRAM traffic and is intentionally not presented as hardware bandwidth
in the main result table.

## 2. Architecture

The implementation keeps the performance path small and separates four
responsibilities:

| Layer | Responsibility |
| --- | --- |
| Official contract | Supplied Transformer behavior, published shapes and source identity |
| Solution data plane | Packed QKV, attention backend, linear precision, residual-normalization backend, CUDA Graph replay and custom kernels |
| Measurement control plane | Hardware probe, workload analysis, candidate eligibility, correctness, GPU timing and observed-path validation |
| Deployment | Formal winner promotion, exact verified-hardware bundle and deterministic offline dispatcher |

[`policy_registry.py`](policy_registry.py) is the single source for execution
compositions. A policy describes orthogonal choices—attention backend, linear
compute dtype, residual-normalization backend and CUDA Graph wrapper—rather than
embedding shape decisions inside the Transformer. An immutable execution plan
resolves eligibility once; the forward pass consumes that plan and reports what
actually ran.

The calibration flow is:

1. **Probe:** collect architecture, memory, runtime features and measured
   performance anchors.
2. **Model:** analyze token count, dimensions, attention cost, launch pressure,
   precision support and fusion opportunity.
3. **Screen:** measure at most three theoretically ranked, eligible candidates
   per resident shape; `tune` remains an explicit experiment command with no
   hidden default set.
4. **Formal:** remeasure every surviving candidate under the complete protocol.
5. **Promote:** atomically write the measured winner for the exact hardware,
   runtime and shape key.
6. **Dispatch:** select that fixed route during execution; if no exact route
   matches, use the portable `eager-sdpa` control.

Shape 14 deliberately does not enter this resident route table. Its streamed
executor independently selects an eligible policy and a memory-safe timing
microbatch, while reusing the same Solution components and result metrics.

## 3. Contribution boundary

Clear ownership is essential because high-performance library kernels and
project-specific optimization are composed in the same execution plan.

### 3.1 Implemented by this project

- shape- and hardware-aware workload analysis, bounded candidate ranking, GPU
  screening, Formal remeasurement and exact route promotion;
- a single typed policy registry, immutable execution plans and observed-path
  evidence that prevents a fallback from being reported as a specialized win;
- packed QKV projection with official-weight conversion;
- FP32 interface with measured internal FP16 attention and linear execution;
- composition and guarding of CUDA Graph, compiled residual-plus-LayerNorm,
  attention backend and precision choices;
- a custom Triton residual-add plus LayerNorm kernel for the measured large-row,
  width-128 Shape 6 family;
- Shape 14 logical-batch streaming, memory guard, policy/microbatch screening,
  complete-workload scheduling and independent Target-only evidence;
- compact per-shape metrics, verified hardware bundles and deterministic
  runtime dispatch.

### 3.2 Provided by third-party runtimes

- PyTorch modules, autograd-free inference utilities, `torch.compile`, CUDA
  Graph APIs and Scaled Dot Product Attention dispatch;
- cuBLAS/cuBLASLt-backed matrix multiplication selected through PyTorch;
- PyTorch Efficient Attention and cuDNN SDPA kernels;
- the Triton language and compiler used to build the project's residual-norm
  kernel.

The project invokes PyTorch's `CUDNN_ATTENTION` backend for the winning Shape 14
policy. NVIDIA documents cuDNN SDPA as using a FlashAttention-2 algorithm
([cuDNN Attention documentation](https://docs.nvidia.com/deeplearning/cudnn/v1.22.0/operations/Attention.html)).
This is therefore a **library-provided FlashAttention-class backend**, not a
custom project attention kernel. The external `flash-attn` package is not an
installed dependency and no result is attributed to it.

## 4. Formal RTX 4080 results: resident Shapes 1–13

All latencies are complete Solution forwards. Peak memory is binary GiB. The
geometric mean is an equally weighted project summary, not a claim about an
unpublished official weighting formula.

| Shape | Target median (ms) | Target P90 (ms) | Speedup | Achieved TFLOP/s | Project MFU | Peak GiB | Actual policy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 01 | 0.492 | 0.493 | 9.25x | 15.31 | 30.1% | 0.084 | `graph-mixed-fp16-efficient-compiled-norm` |
| 02 | 0.140 | 0.143 | 33.09x | 0.84 | 1.8% | 0.018 | `graph-fused-norm` |
| 03 | 0.148 | 0.148 | 30.49x | 3.17 | 6.7% | 0.022 | `graph-fused-norm` |
| 04 | 0.238 | 0.238 | 18.87x | 7.92 | 16.8% | 0.034 | `graph-fused-norm` |
| 05 | 0.929 | 1.146 | 4.54x | 16.20 | 31.9% | 0.150 | `graph-mixed-fp16-efficient-compiled-norm` |
| 06 | 104.440 | 106.300 | 4.31x | 11.26 | 12.0% | 4.900 | `mixed-fp16-core-efficient-triton-norm` |
| 07 | 0.295 | 0.296 | 14.93x | 2.28 | 3.9% | 0.033 | `graph-mixed-fp16-efficient-compiled-norm` |
| 08 | 7.565 | 7.775 | 1.75x | 55.65 | 59.4% | 0.352 | `mixed-fp16-core-efficient` |
| 09 | 0.479 | 0.480 | 8.12x | 15.70 | 30.9% | 0.084 | `graph-mixed-fp16-efficient-compiled-norm` |
| 10 | 0.469 | 0.470 | 9.23x | 16.04 | 31.6% | 0.084 | `graph-mixed-fp16-efficient-compiled-norm` |
| 11 | 0.609 | 0.611 | 11.66x | 12.35 | 24.3% | 0.084 | `graph-mixed-fp16-efficient-compiled-norm` |
| 12 | 0.214 | 0.214 | 21.02x | 7.85 | 16.6% | 0.034 | `graph-fused-norm` |
| 13 | 5.302 | 5.482 | 20.45x | 22.69 | 24.2% | 0.259 | `mixed-fp16-core-efficient` |

Summary:

- 13/13 cases completed successfully;
- 10.98x unweighted geometric-mean speedup;
- zero failed output elements across 65 correctness trials;
- largest observed absolute error: 0.00177722;
- selected policies were observed as fully applied in every result.

Large relative-error maxima occur only where the reference value is close to
zero; pass/fail uses the specified combined `atol + rtol * |reference|` rule.

## 5. Shape 14: independent Target-only result

| Shape | Target median (ms) | Target P90 (ms) | End-to-end (ms) | Achieved TFLOP/s | Project MFU | Peak GiB | Actual policy | Timing schedule |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 14 | 15,987.85 | 15,988.45 | 19,333.10 | 87.02 | 92.8% | 6.93 | `mixed-fp16-core-cudnn` | microbatch 2 x 16 |

The full `B=1, S=100000, D=1024` comparator checks 102.4 million elements.
It passed with zero failed elements and maximum absolute error 0.000833869.
This is one provisional trial against the internal query-block reference. The
result demonstrates feasibility and measures the complete logical `B=32`
Target, but it is not yet evidence for an official Baseline speedup.

## 6. Insight Cards

### Insight Card 1 — Small fixed shapes are launch-bound

- **Observation:** Shapes 2–4 and 12 complete in 0.140–0.238 ms after
  optimization, yet project MFU remains only 1.8–16.8%. Their arithmetic volume
  is too small to saturate the GPU; framework and launch costs dominate.
- **Mechanism:** capture the complete fixed-shape forward in a CUDA Graph and
  fuse residual-add plus LayerNorm at a local compiler boundary.
- **Candidates:** `eager-sdpa`, `graph`, `graph-fused-norm`, and the mixed
  attention graph compositions where eligible.
- **Measured decision:** `graph-fused-norm` wins Shapes 2–4 and 12; the compiled
  mixed-attention graph wins Shapes 1, 5, 7 and 9–11.
- **Boundary:** these routes are exact to shape and runtime. Low MFU here does
  not imply that a larger custom GEMM would help.

### Insight Card 2 — Throughput and fusion must be composed selectively

- **Observation:** Shape 8 reaches 55.65 useful TFLOP/s and 59.4% project MFU,
  while Shape 6 has enough rows for residual-normalization traffic to become a
  material repeated cost.
- **Mechanism:** move attention and linear compute to an accuracy-checked FP16
  core; add a custom one-pass Triton residual-add plus LayerNorm only where the
  measured row count and width make it useful.
- **Candidates:** Efficient and cuDNN mixed-FP16 attention, FP16 core execution,
  compiled norm, and the project Triton norm kernel.
- **Measured decision:** `mixed-fp16-core-efficient` wins Shapes 8 and 13;
  `mixed-fp16-core-efficient-triton-norm` wins Shape 6.
- **Boundary:** the Triton kernel is guarded to its measured width-128,
  large-row family. Library GEMMs are retained instead of being rewritten
  without evidence of headroom.

### Insight Card 3 — Shape 14 is a capacity problem before it is a routing problem

- **Observation:** dense attention scales as `S^2`, and the full `B=32` inputs,
  outputs and intermediates cannot reside together on a 16 GiB GPU. Attention
  represents 94.2% of the project's useful matmul FLOPs for this shape.
- **Mechanism:** validate one complete long-sequence sample, use a memory-efficient
  SDPA backend that does not expose the dense attention matrix, and stream the
  logical batch through a measured microbatch schedule.
- **Candidates:** PyTorch Efficient SDPA and cuDNN SDPA combined with eligible
  microbatch divisors.
- **Measured decision:** cuDNN SDPA with FP16 core and microbatch 2 wins the
  current screen, completing 16 microbatches at 87.02 useful TFLOP/s and
  6.93 GiB peak allocation.
- **Boundary:** cuDNN supplies the attention kernel; the project's contribution
  is the correctness, scheduling, screening and evidence path. The current
  reference is provisional and no dense Baseline speedup is claimed.

## 7. Correctness and execution integrity

Correctness is a gate, not a weighted optimization objective. Each resident
candidate receives the same weights and seeded inputs as the Baseline, then is
tested over five trials before performance can be published. Shape 14 performs
a complete long-sequence `B=1` comparison before timing the streamed logical
batch.

Execution integrity is checked separately from numerical correctness. The
requested attention backend, linear dtype, residual-norm backend and runtime
wrapper are compared with the observed execution path. A backend that is
ineligible, throws, or silently falls back cannot be promoted under the
specialized policy name.

The internal `safe` policy preserves the explicit reference calculation order
for diagnosis and fallback testing. It is not routable. `eager-sdpa` is the
portable optimized control and is the deterministic dispatcher fallback when
no exact verified route matches.

## 8. Feasibility, impact and portability

### Feasibility and practicality

- Online execution performs a table lookup, not candidate search or profiling.
- Calibration uses a bounded candidate set and serial GPU ownership, so
  measurements do not compete for the device.
- Routes are tied to GPU, compute capability, OS, PyTorch, CUDA, driver, dtype,
  runtime math policy and workload shape.
- The verified bundle is updated only by Formal results that pass correctness
  and observed-execution checks.
- Shape 14 stays outside the resident route table, avoiding fake generality and
  keeping its capacity-specific scheduler isolated.

### Impact and relevance

The method targets the common practical case in which a consumer GPU does not
match the data-center hardware assumptions of a universal kernel package.
Rather than asking one backend to win everywhere, it extracts repeatable value
from fixed production-like shapes: launch-bound cases use graphs and fusion,
throughput cases use mixed precision, and capacity-bound cases use streaming.
The same measurement loop can be applied to inference services, local ML tools
and other fixed-shape Transformer deployments.

### Cross-hardware status

Cross-hardware support is retained as an executable cold-start mechanism:
probe a new device, theoretically rank eligible candidates, measure them, and
write exact routes from the Formal winners. It is deliberately not advertised
as cross-hardware performance validation. The results in this report are
verified only on the recorded RTX 4080 stack; another GPU or software version
requires calibration.

## 9. AI-assisted development

OpenAI Codex was used as a development collaborator. Human guidance supplied
the competition contract, latest official shapes, hardware target, correctness
boundary, preference for a clean mainline, and the requirement that real GPU
measurements—not architectural enthusiasm—decide what remains.

The collaboration followed a repeatable pattern:

1. Codex inspected the official implementation, profiler evidence and
   structured results, then proposed bounded bottleneck hypotheses.
2. Candidate implementations and architecture changes were reviewed against
   the existing control/data-plane boundary.
3. Deterministic code performed comparator checks, observed-path validation,
   GPU timing and route promotion.
4. Measured outcomes decided whether to keep, specialize or reject a candidate.

Representative outcomes include retaining library GEMMs, adding the Shape 6
Triton residual-norm kernel, promoting the graph-plus-compiled-norm composition
for several small shapes, and choosing cuDNN plus microbatch 2 for Shape 14.
This report summarizes decisions rather than reproducing raw conversations or
private machine paths.

The planned in-project hardware-aware Agent runtime is **deferred and not
implemented**. It is not required in the inference hot path and is not claimed
as a completed contribution. The deterministic probe, measurement, calibration
and dispatch services are intentionally usable later by such an Agent without
changing the current competition implementation.

The repository does not encode an immutable Codex model-version identifier or
a verified list of packaged Skills, so this report does not infer them. The
submission metadata should name only the exact model and Skills actually used
at export time.

## 10. Reproduction

Create the recorded Windows environment and verify CUDA:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-windows-rtx4080.txt
. .\activate_windows_rtx4080.ps1
.\.venv\Scripts\python.exe environment_check.py --check-extension
```

Run the current exact RTX 4080 bundle:

```powershell
.\.venv\Scripts\python.exe `
  verified_hardware/nvidia_geforce_rtx_4080/run_verified.py `
  --scope all --preset formal
```

Run the two paths directly:

```powershell
.\.venv\Scripts\python.exe -m runner benchmark `
  --preset formal --device cuda:0

.\.venv\Scripts\python.exe -m runner benchmark-streamed `
  --case-id official_14 --preset formal --device cuda:0
```

Probe and recalibrate a changed hardware/software stack:

```powershell
.\.venv\Scripts\python.exe -m runner probe --device cuda:0
.\.venv\Scripts\python.exe -m runner calibrate `
  --preset formal --device cuda:0
```

Run control-plane, architecture and real-GPU tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -m "not gpu"
.\.venv\Scripts\python.exe -m pytest -q -m architecture tests\architecture
.\.venv\Scripts\python.exe -m pytest -q -m gpu tests\gpu\test_gpu_smoke.py
$env:RUN_SHAPE14_GPU='1'
.\.venv\Scripts\python.exe -m pytest -q -m gpu tests\gpu\test_shape14_gpu.py
```

## 11. Limitations

- The official final MFU weighting and any bandwidth correction were not fixed
  in the meeting material. Project MFU and geometric-mean speedup must not be
  presented as the official final score.
- Resident performance is exact to the recorded RTX 4080 software stack and
  source identities. Driver, PyTorch, CUDA, runtime-policy or implementation
  changes invalidate the verified route evidence.
- Shape 14 correctness is provisional until an external official full-shape
  reference is available. Its result is Target-only and has no Baseline
  speedup.
- Shape 14 uses library-provided cuDNN SDPA. The project has a custom Triton
  residual-norm kernel, but no custom online-attention kernel and no external
  `flash-attn` dependency.
- Peak allocation is a PyTorch device-allocation measurement, not total board
  memory consumption. Logical traffic is an operator estimate, not a profiler
  DRAM counter.
- Current accuracy evidence covers the recorded seeded FP32 variant. Different
  masks, padding, scales, dtypes or training/autograd behavior require separate
  validation.
- Cross-hardware calibration is implemented, but performance conclusions have
  not been verified on a second GPU in this report.
