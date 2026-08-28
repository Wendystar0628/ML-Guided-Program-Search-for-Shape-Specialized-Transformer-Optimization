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
official-style comparator. Their unweighted geometric-mean speedup is
**8.220830719x**.
The independent Shape 14 path also runs the complete logical workload without a
dense `S x S` attention matrix: its measured Target median is **17,136.172 ms**
at **81.1887 useful TFLOP/s**, with **7.307 GiB**
(**7,845,867,008 bytes**) peak device allocation. Shape 14 is reported
separately because its current correctness reference is provisional and no
executable dense Baseline result exists.

The [unified RTX 4080 performance result](results/final/nvidia_geforce_rtx_4080.json)
is bound to the current workload, official snapshot, Solution implementation,
route table, runtime and measurement protocol recorded by the
[RTX 4080 verified bundle](verified_hardware/nvidia_geforce_rtx_4080/).

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
also records saturated `4096 x 4096` GEMM anchors: **86.3223 FP16 TFLOP/s**,
**94.5265 BF16 TFLOP/s**, and **47.3050 FP32/TF32 TFLOP/s**. The bounded
device-copy anchor is **272.031 GB/s**, and the eager-launch anchor is
**6.841 microseconds**. These are local runtime anchors, not vendor peak
specifications.

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

The paired rows are stored in the
[unified performance result](results/final/nvidia_geforce_rtx_4080.json). They
bind Solution implementation `017f6028...` and route table `1ed68576...`.

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

The Shape 14 row in the
[unified performance result](results/final/nvidia_geforce_rtx_4080.json) is
explicitly marked `provisional` and `target_only`. It does not invent a dense
Baseline latency or a speedup.

### 1.4 Performance metrics

`Achieved TFLOP/s` uses the project's useful matmul FLOP count divided by Target
latency. `Project MFU` estimates segmented ideal compute time from measured
saturated GEMM roofs matched to the observed linear and attention compute
dtypes, then divides it by measured Target time. It is a diagnostic estimate,
not an official score and not a vendor peak utilization claim.

Every shape also carries a logical operator-traffic estimate. It is not measured
DRAM traffic and is intentionally not presented as hardware bandwidth in the
main result table.

## 2. Architecture

The implementation keeps the performance path small and separates four
responsibilities:

| Layer | Responsibility |
| --- | --- |
| Official contract | Supplied Transformer behavior, published shapes and source identity |
| Solution data plane | Packed QKV, attention backend, linear precision, cross-layer residual-to-normalization scheduling, full/batch-tiled CUDA Graph replay, compiled forward and custom kernels |
| Measurement control plane | Hardware probe, workload analysis, candidate eligibility, correctness, GPU timing and observed-path validation |
| Deployment | Formal winner promotion, exact verified-hardware bundle and deterministic offline dispatcher |

[`policy_registry.py`](policy_registry.py) is the single source for execution
compositions. A policy describes orthogonal choices—attention backend, linear
compute dtype, residual-normalization backend and outer runtime—rather than
embedding shape decisions inside the Transformer. The outer runtime can be
eager execution, one full-forward CUDA Graph, batch-tiled CUDA Graph replay, or
a cached fixed-plan compiled forward. An immutable execution plan resolves
eligibility once; the forward pass consumes that plan and reports what actually
ran.

The Transformer forward is organized as a normalization pipeline. Attention's
residual update is paired with the current block's `norm2`; the FFN residual is
paired with the next block's `norm1`, or with the final normalization after the
last block. Compiled and Triton residual-to-norm implementations return both the
updated residual stream and normalized stream. This exposes a useful fusion
boundary across Transformer blocks while preserving the supplied pre-norm
calculation.

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
microbatch divisor, while reusing the same Solution components and result
metrics. The selected microbatch size and resulting count are measured runtime
outputs, not fixed properties of Shape 14.

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
- composition and guarding of full-forward CUDA Graph, batch-tiled Graph,
  fixed-plan compiled forward, attention backend and precision choices;
- cross-layer residual-to-normalization scheduling, with compiled and Triton
  implementations sharing the same mathematical boundary;
- a custom Triton residual-add plus LayerNorm candidate guarded to the
  high-row, width-128 Shape 6 family;
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

The final Shape 14 artifact selected a policy that invokes PyTorch's
`CUDNN_ATTENTION` backend. NVIDIA documents cuDNN SDPA as using a
FlashAttention-2 algorithm
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
| 01 | 0.363520 | 0.398234 | 5.022x | 20.70 | 23.98% | 0.080 | `graph-mixed-fp16-core-efficient-compiled-norm` |
| 02 | 0.133120 | 0.133248 | 14.940x | 0.88 | 1.87% | 0.018 | `graph-fused-norm` |
| 03 | 0.139264 | 0.139264 | 14.279x | 3.38 | 7.14% | 0.021 | `graph-fused-norm` |
| 04 | 0.221184 | 0.222208 | 8.865x | 8.50 | 17.98% | 0.033 | `graph-fused-norm` |
| 05 | 0.685056 | 1.122509 | 3.442x | 21.97 | 25.45% | 0.142 | `graph-mixed-fp16-core-efficient-compiled-norm` |
| 06 | 105.774719 | 117.979649 | 10.820x | 11.12 | 12.88% | 4.973 | `batch-tiled-mixed-fp16-core-efficient-compiled-norm` |
| 07 | 0.195584 | 0.197632 | 9.481x | 3.44 | 3.99% | 0.032 | `graph-mixed-fp16-core-efficient-compiled-norm` |
| 08 | 6.416384 | 6.649037 | 2.196x | 65.61 | 76.00% | 0.401 | `compiled-mixed-fp16-core-efficient` |
| 09 | 0.353280 | 0.365568 | 4.531x | 21.30 | 24.67% | 0.080 | `graph-mixed-fp16-core-efficient-compiled-norm` |
| 10 | 0.344064 | 0.363622 | 5.225x | 21.87 | 25.33% | 0.080 | `graph-mixed-fp16-core-efficient-compiled-norm` |
| 11 | 0.484352 | 0.618496 | 15.529x | 15.54 | 18.00% | 0.080 | `graph-mixed-fp16-core-efficient-compiled-norm` |
| 12 | 0.198656 | 0.199680 | 10.031x | 8.46 | 17.88% | 0.033 | `graph-fused-norm` |
| 13 | 4.060672 | 4.133069 | 28.841x | 29.63 | 34.33% | 0.197 | `compiled-mixed-fp16-core-efficient` |

Summary:

- 13/13 cases completed successfully;
- 8.220830719x unweighted geometric-mean speedup;
- zero failed output elements across 65 correctness trials;
- largest observed absolute error: 0.00188206;
- selected policies were observed as fully applied in every result.

Large relative-error maxima occur only where the reference value is close to
zero; pass/fail uses the specified combined `atol + rtol * |reference|` rule.

## 5. Shape 14: independent Target-only result

| Shape | Target median (ms) | Target P90 (ms) | End-to-end (ms) | Achieved TFLOP/s | Project MFU | Peak GiB | Actual policy | Timing schedule |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 14 | 17,136.172 | 17,158.589 | 18,821.000 | 81.1887 | 94.0529% | 7.307 | `mixed-fp16-core-cudnn` | microbatch 2 x 16 |

The full `B=1, S=100000, D=1024` comparator checks 102.4 million elements.
It passed with zero failed elements and maximum absolute error 0.000833869.
This is one provisional trial against the internal query-block reference. The
result demonstrates feasibility and measures the complete logical `B=32`
Target, but it is not yet evidence for an official Baseline speedup.

## 6. Insight Cards

### Insight Card 1 — Small fixed shapes are launch-bound

- **Observation:** Shapes 2–4 and 12 complete in 0.133120–0.221184 ms, yet
  project MFU remains only 1.87–17.98%. Their arithmetic volume is too small to
  saturate the GPU; framework and launch costs dominate.
- **Mechanism:** capture the complete fixed-shape forward in a CUDA Graph and
  expose each residual-to-normalization pair as one local compiler boundary.
- **Candidates:** `eager-sdpa`, `graph`, `graph-fused-norm`, and the mixed
  attention or mixed-core Graph compositions where eligible.
- **Measured decision:** `graph-fused-norm` serves Shapes 2–4 and 12;
  `graph-mixed-fp16-core-efficient-compiled-norm` serves Shapes 1, 5, 7 and
  9–11.
- **Boundary:** these routes are exact to shape and runtime. Low MFU here does
  not imply that a larger custom GEMM would help.

### Insight Card 2 — Throughput and fusion must be composed selectively

- **Observation:** Shape 8 reaches 65.61 useful TFLOP/s and 76.00% project MFU,
  while Shape 6 has enough rows for residual-normalization traffic to become a
  material repeated cost.
- **Mechanism:** move attention and linear compute to an accuracy-checked FP16
  core; expose cross-layer residual-to-normalization fusion; use batch-tiled
  Graph replay for the exact independent high-batch case or compile a complete
  fixed-plan forward for guarded wide/long cases.
- **Candidates:** Efficient and cuDNN mixed-FP16 attention, FP16 core execution,
  compiled or Triton residual-to-norm, batch-tiled Graph, and compiled forward.
- **Measured decision:** Shapes 8 and 13 use
  `compiled-mixed-fp16-core-efficient`. Shape 6 uses
  `batch-tiled-mixed-fp16-core-efficient-compiled-norm`; its final Target
  median is **105.774719 ms**. This is the bound Formal result, without a claim
  of stable improvement over an earlier revision.
- **Boundary:** the Triton kernel is guarded to its measured width-128,
  large-row family. Library GEMMs are retained instead of being rewritten
  without evidence of headroom.

### Insight Card 3 — Shape 14 is a capacity problem before it is a routing problem

- **Observation:** dense attention scales as `S^2`, and the full `B=32` inputs,
  outputs and intermediates cannot reside together on a 16 GiB GPU.
- **Mechanism:** validate one complete long-sequence sample, use a memory-efficient
  SDPA backend that does not expose the dense attention matrix, and stream the
  logical batch through a measured microbatch schedule.
- **Candidates:** PyTorch Efficient SDPA and cuDNN SDPA combined with eligible
  microbatch divisors.
- **Measured decision:** the streamed Formal run selected
  `mixed-fp16-core-cudnn`, microbatch size 2, and 16 microbatches. Its Target
  median is 17,136.172 ms at 81.1887 useful TFLOP/s and 94.0529% project MFU.
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

Representative outcomes include retaining library GEMMs, adding guarded
compiled and Triton residual-to-norm paths, and introducing batch-tiled Graph
and fixed-plan compiled-forward candidates without bypassing the shared
correctness and measurement path. This report summarizes decisions rather than
reproducing raw conversations or private machine paths.

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
