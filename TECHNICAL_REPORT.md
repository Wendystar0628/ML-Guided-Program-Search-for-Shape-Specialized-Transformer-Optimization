# AI-Assisted Shape-Aware Transformer Optimization on an RTX 4080

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
supplied official comparator. Their unweighted geometric-mean speedup is
**9.192276080x**.
The independent Shape 14 path also runs the complete logical workload without a
dense `S x S` attention matrix: its measured Target median is **17,206.558 ms**
at **80.8566 useful TFLOP/s**, with **7.307 GiB**
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
| CPU / host memory | Intel Core i7-14700KF, 20 cores / 28 threads; 68,534,743,040 bytes (63.83 GiB) RAM |
| Working storage | SOLIDIGM SSDPFKNU020TZ, local NVMe, 2 TB class |
| GPU | NVIDIA GeForce RTX 4080, Ada, compute capability 8.9 |
| GPU resources | 76 SMs, 16 GiB VRAM, 64 MiB L2, 256-bit memory bus |
| Theoretical memory bandwidth | 716.864 GB/s |
| Driver | 610.88 |
| Python | 3.12.5 |
| PyTorch / CUDA runtime | 2.12.1+cu132 / CUDA 13.2 |
| cuDNN / Triton | 9.20.0 / 3.7.1 |
| Runtime math policy | `matmul_precision=high`, TF32 allowed |

The current [hardware profile](verified_hardware/nvidia_geforce_rtx_4080/profile.json)
also records saturated `4096 x 4096` GEMM anchors: **93.4278 FP16 TFLOP/s**,
**99.5565 BF16 TFLOP/s**, and **49.5361 FP32/TF32 TFLOP/s**. The bounded
device-copy anchor is **291.974 GB/s**, and the eager-launch anchor is
**6.120 microseconds**. These are local runtime anchors, not vendor peak
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
bind official snapshot
`d4f45c9336880b31ab1ae8a8f354aa05862772553162851257490bb936878762`,
Solution implementation
`57e014b8cbb626905e4a619e2fd468b7c7113b5d2b88217eac876c0fe256d4f4`,
workload set
`621c0f205180f303970ed9e7ce2ee1548cd6c1ac5d46fff1e69dc938039736e9`,
and route table
`440d6fa1f6ae86f41ccb5a83ec5029a1f9e84ab1344e28616d98bf2f7de419f9`.

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
- a custom mixed residual-add plus LayerNorm Triton kernel used by Shapes 5 and
  6; it keeps the
  accumulated residual stream in FP32, consumes each branch update in FP16 and
  emits the next intermediate branch input in FP16, with exact shape guards;
- version-aware unchanged-input staging for the Shape 5 CUDA Graph route, which
  reuses the captured input only when tensor identity and version both match;
- a Shape 8 compiled path that keeps the official FP32 parameters authoritative
  while using derived, non-persistent FP16 shadow weights for repeated inference;
- a custom forward-only Shape 13 Triton causal-attention kernel with blocked
  QK/PV computation and FP32 online-softmax state, guarded to the exact
  `B=64, H=4, S=1024, head_dim=32` tensor family and composed with the
  fixed-plan compiled forward;
- Shape 14 logical-batch streaming, memory guard, policy/microbatch screening,
  complete-workload scheduling and independent Target-only evidence;
- compact per-shape metrics, verified hardware bundles and deterministic
  runtime dispatch.

### 3.2 Provided by third-party runtimes

- PyTorch modules, autograd-free inference utilities, `torch.compile`, CUDA
  Graph APIs and Scaled Dot Product Attention dispatch;
- cuBLAS/cuBLASLt-backed matrix multiplication selected through PyTorch;
- PyTorch Efficient Attention and cuDNN SDPA kernels;
- the Triton language and compiler used to build the project's Shapes 5/6
  residual-normalization and Shape 13 attention kernels.

The final Shape 14 artifact selected a policy that invokes PyTorch's
`CUDNN_ATTENTION` backend. This is a library-provided memory-efficient
attention path, not the custom Shape 13 Triton attention kernel. The external
`flash-attn` package is not an installed dependency and no result is attributed
to it.

## 4. Formal RTX 4080 results: resident Shapes 1–13

All latencies are complete Solution forwards. Peak memory is binary GiB. The
geometric mean is an equally weighted project summary, not a claim about an
unpublished official weighting formula.

| Shape | Target median (ms) | Target P90 (ms) | Speedup | Achieved TFLOP/s | Project MFU | Peak GiB | Actual policy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 01 | 0.364544 | 0.383386 | 4.721x | 20.64 | 23.44% | 0.080 | `graph-mixed-fp16-core-efficient-compiled-norm` |
| 02 | 0.133120 | 0.134144 | 13.839x | 0.88 | 1.87% | 0.018 | `graph-fused-norm` |
| 03 | 0.139264 | 0.140288 | 13.809x | 3.38 | 7.16% | 0.021 | `graph-fused-norm` |
| 04 | 0.221184 | 0.222208 | 8.769x | 8.50 | 18.03% | 0.033 | `graph-fused-norm` |
| 05 | 0.501760 | 0.759091 | 4.710x | 29.99 | 34.05% | 0.049 | `graph-mixed-fp16-core-efficient-triton-mixed-norm-reuse-input` |
| 06 | 45.323263 | 46.174311 | 10.730x | 25.94 | 29.45% | 1.257 | `batch-tiled-mixed-fp16-core-efficient-triton-mixed-norm` |
| 07 | 0.140288 | 0.185549 | 14.445x | 4.80 | 5.45% | 0.012 | `compiled-mixed-fp16-core-efficient` |
| 08 | 6.147072 | 6.366208 | 2.334x | 68.48 | 77.76% | 0.274 | `compiled-shape08-fp16-shadow-weights` |
| 09 | 0.352256 | 0.373248 | 6.937x | 21.36 | 24.25% | 0.080 | `graph-mixed-fp16-core-efficient-compiled-norm` |
| 10 | 0.348160 | 0.406410 | 5.093x | 21.61 | 24.54% | 0.080 | `graph-mixed-fp16-core-efficient-compiled-norm` |
| 11 | 0.387072 | 0.586752 | 19.602x | 19.44 | 22.07% | 0.017 | `compiled-mixed-fp16-core-efficient` |
| 12 | 0.197632 | 0.198656 | 9.767x | 8.50 | 18.02% | 0.033 | `graph-fused-norm` |
| 13 | 3.202560 | 3.520819 | 36.685x | 37.57 | 42.66% | 0.181 | `compiled-mixed-fp16-core-shape13-triton-attention` |

Summary:

- 13/13 cases completed successfully;
- 9.192276080x unweighted geometric-mean speedup;
- zero failed output elements across 65 correctness trials;
- largest observed absolute error: 0.00180167;
- selected policies were observed as fully applied in every result.

Large relative-error maxima occur only where the reference value is close to
zero. The exact elementwise rule is
`abs(target - reference) <= 0.002` **or**
`abs(target - reference) / abs(reference) <= 0.02`; the two allowances are not
added together.

Measured method allocation:

| Method composition | Formal winner for | Why it remains separate |
| --- | --- | --- |
| Full Graph + fused residual/LayerNorm | 02, 03, 04, 12 | Very small fixed shapes are launch-bound and do not need mixed-precision compute to win. |
| Full Graph + mixed-FP16 core + Efficient Attention + compiled residual/LayerNorm | 01, 09, 10 | Moderate fixed shapes benefit from both launch amortization and branch compute reduction. |
| Full Graph + mixed-FP16 core + Efficient Attention + custom Triton mixed residual/LayerNorm + version-aware unchanged-input staging | 05 | The fixed input can safely avoid repeated graph staging when tensor identity and version are unchanged, while the mixed residual boundary removes extra traffic. |
| Fixed-plan compiled forward + mixed-FP16 core + Efficient Attention | 07, 11 | Whole-forward compilation wins for narrow/small-head families where an explicit Graph composition is not the best measured route. |
| Fixed-plan compiled forward + non-persistent FP16 shadow weights | 08 | Authoritative weights remain FP32 while repeated inference avoids rematerializing their FP16 compute representation. |
| Batch-tiled Graph + mixed-FP16 core + custom Triton mixed residual/LayerNorm | 06 | The independent `B=10000` case needs bounded tiles and a dual-dtype residual boundary. |
| Fixed-plan compiled forward + mixed-FP16 core + custom Triton online-softmax attention | 13 | The exact `S=1024, head_dim=32` family benefits from a shape-specialized causal attention kernel. |

## 5. Shape 14: independent Target-only result

| Shape | Target median (ms) | Target P90 (ms) | End-to-end (ms) | Achieved TFLOP/s | Project MFU | Peak GiB | Actual policy | Timing schedule |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 14 | 17,206.558 | 17,208.098 | 18,879.743 | 80.8566 | 91.8066% | 7.307 | `mixed-fp16-core-cudnn` | microbatch 2 x 16 |

The full `B=1, S=100000, D=1024` comparator checks 102.4 million elements.
It passed with zero failed elements and maximum absolute error 0.000833869.
This is one provisional trial against the internal query-block reference. The
result demonstrates feasibility and measures the complete logical `B=32`
Target, but it is not yet evidence for an official Baseline speedup.

## 6. Insight Cards

### Insight Card 1 — Small fixed shapes are launch-bound

- **Observation:** Shapes 2–4 and 12 complete in 0.133120–0.221184 ms, yet
  project MFU remains only 1.87–18.03%. Their arithmetic volume is too small to
  saturate the GPU; framework and launch costs dominate.
- **Mechanism:** capture the complete fixed-shape forward in a CUDA Graph and
  expose each residual-to-normalization pair as one local compiler boundary.
- **Candidates:** `eager-sdpa`, `graph`, `graph-fused-norm`, and the mixed
  attention or mixed-core Graph compositions where eligible.
- **Measured decision:** `graph-fused-norm` serves Shapes 2–4 and 12;
  `graph-mixed-fp16-core-efficient-compiled-norm` serves Shapes 1, 9 and 10.
  Shape 5 adds the custom mixed residual/LayerNorm boundary and version-aware
  unchanged-input staging. Shapes 7 and 11 instead select the complete fixed-plan compiled forward,
  showing that small latency alone does not determine the best outer runtime.
- **Boundary:** these routes are exact to shape and runtime. Low MFU here does
  not imply that a larger custom GEMM would help.

### Insight Card 2 — Shape 6 needs a dual-dtype residual boundary

- **Observation:** Shape 6 has `B=10000`, so repeated residual writes,
  normalization reads and branch conversions remain material even after the
  GEMMs move to FP16. A full resident graph is also impractical for this
  independent high-batch workload.
- **Mechanism:** execute fixed tiles of 128 samples, keep the accumulated
  residual stream in FP32, consume the FP16 attention/FFN branch update, and
  fuse residual addition with LayerNorm in one Triton kernel. Intermediate
  normalized values are emitted directly in FP16 for the next branch; the
  final boundary remains FP32.
- **Measured decision:**
  `batch-tiled-mixed-fp16-core-efficient-triton-mixed-norm` reaches
  **45.323263 ms**, **25.94 useful TFLOP/s**, **29.45% project MFU** and
  **1.257 GiB** peak allocation.
- **Boundary:** the custom kernel is intentionally guarded to the measured
  `tile_B=128, S=128, D=128` inference family. It does not replace library GEMMs
  or claim general LayerNorm coverage.

### Insight Card 3 — Attention and compilation must follow head geometry

- **Observation:** Shape 7 (`D=32, H=4`) and Shape 11 (`D=128, H=16`) both have
  `head_dim=8`, while Shape 13 has `S=1024, head_dim=32`. These geometries do
  not share one best execution composition. Shape 8 is wide enough to reach
  **68.48 useful TFLOP/s** with the library attention path.
- **Mechanism:** use a whole fixed-plan compiled forward for Shapes 7 and 11.
  Shape 8 keeps FP32 owner weights and supplies its compiled path with derived,
  non-persistent FP16 shadow weights.
  For exact Shape 13, use a custom Triton causal-attention kernel that processes
  QK and PV in blocks, maintains row maxima and normalization sums in FP32, and
  therefore avoids materializing the full score/probability matrix.
- **Measured decision:** Shapes 7 and 11 reach **0.140288 ms** and
  **0.387072 ms** with `compiled-mixed-fp16-core-efficient`; Shape 8 reaches
  **6.147072 ms** with `compiled-shape08-fp16-shadow-weights`. Shape 13 reaches
  **3.202560 ms**, **36.685x** speedup and **42.66% project MFU** with
  `compiled-mixed-fp16-core-shape13-triton-attention`.
- **Boundary:** the Triton attention specialization accepts only causal,
  no-mask, forward-only FP16 tensors with the exact measured Shape 13 geometry.
  Other shapes continue to use measured library attention routes.

### Insight Card 4 — Shape 14 is a capacity problem before it is a routing problem

- **Observation:** dense attention scales as `S^2`, and the full `B=32` inputs,
  outputs and intermediates cannot reside together on a 16 GiB GPU.
- **Mechanism:** validate one complete long-sequence sample, use a memory-efficient
  SDPA backend that does not expose the dense attention matrix, and stream the
  logical batch through a measured microbatch schedule.
- **Candidates:** PyTorch Efficient SDPA and cuDNN SDPA combined with eligible
  microbatch divisors.
- **Measured decision:** the streamed Formal run selected
  `mixed-fp16-core-cudnn`, microbatch size 2, and 16 microbatches. Its Target
  median is **17,206.558 ms** at **80.8566 useful TFLOP/s** and **91.8066%
  project MFU**.
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
On the disclosed machine, the measured benefit is concrete: resident complete
forwards are 2.334x–36.685x faster than the supplied baseline, and the streamed
schedule completes Shape 14 with 7.307 GiB peak allocation on a 16 GiB GPU.
Those improvements can reduce latency or make a previously non-resident fixed
workload executable for local ML tools, offline inference and shape-stable
services. They are Transformer-kernel measurements, not an unmeasured claim
about application-level throughput, cost, or energy. The same calibration loop
is applicable to other devices, but performance value outside this RTX 4080
stack remains to be measured.

### Cross-hardware status

Cross-hardware support is retained as an executable cold-start mechanism:
probe a new device, theoretically rank eligible candidates, measure them, and
write exact routes from the Formal winners. It is deliberately not advertised
as cross-hardware performance validation. The results in this report are
verified only on the recorded RTX 4080 stack; another GPU or software version
requires calibration.

## 9. AI-assisted development

### 9.1 Tool, model and Skill disclosure

| Item | Actual use in this project |
| --- | --- |
| AI coding tool | OpenAI Codex Desktop was used to inspect the repository, edit code and documentation, run PowerShell/Git commands, coordinate bounded parallel reviews, and interpret deterministic GPU results. |
| LLM | A GPT-5-based Codex model. The repository did not persist a stable point-version alias for every development session, so this report does not invent one. |
| Runtime AI API | None. The benchmark, calibration and deployed dispatcher do not call an LLM or external AI service. |
| Project-local Agent Skills | None were packaged or claimed for the verified optimization sequence. Direct task-level instructions guided Codex; the planned device/profile/kernel Skills belong to a deferred Agent design and are not presented as used deliverables. |
| Non-AI development tools | PowerShell, Git, Python, PyTorch, Triton, pytest and Ruff. |

This is a complete disclosure of the AI tooling that can be verified from the
project record. “No project-local Skill package” is an intentional truthful
entry, not a missing generated artifact. Built-in repository, shell and
multi-agent capabilities are tool functions rather than separately authored
competition Skills.

### 9.2 Human guidance and control

Human guidance supplied the competition contract, latest official shapes,
hardware target, correctness boundary and four persistent engineering choices:

1. prioritize the measurable performance mainline and avoid over-engineering;
2. use real GPU results rather than CPU timing or theoretical rank as the final
   selection authority;
3. keep project-owned kernels, library primitives and provisional evidence
   explicitly separated; and
4. delete obsolete architecture instead of preserving compatibility layers that
   would obscure the final implementation.

Codex proposed hypotheses and implementation candidates. Deterministic code—not
the model—performed comparator checks, observed-path validation, CUDA Event
timing and route publication. Human direction and measured outcomes decided
whether a candidate was kept, narrowed to one shape, or removed.

### 9.3 Representative interaction history

The following entries are concise, privacy-safe summaries of real development
interactions. They preserve the decision chain without publishing raw chats,
system prompts, account data or private machine paths.

| Human goal or constraint | AI-assisted action | Deterministic evidence and decision | Repository evidence |
| --- | --- | --- | --- |
| Replace RTX-4080-only hard-coded routing with a hardware-aware cold start whose measured winner is published automatically. | Refactored probing, bounded candidate ranking, calibration and exact route promotion around shared contracts. | The theoretical model only narrows candidates; Smoke and Formal complete-forward GPU measurements own promotion. | `d257a4a`, [policy registry](policy_registry.py), [calibration service](runner/calibration.py) |
| Remove verbose result artifacts and keep only information useful to judges and later tuning. | Consolidated public output into one final JSON per hardware identity and isolated regenerable experiments. | The [final result](results/final/nvidia_geforce_rtx_4080.json) contains the current per-shape latency, correctness, throughput, MFU estimate, memory and policy; intermediate runs remain ignored. | `0b08a6a`, [result contract](results/README.md) |
| Migrate completely to the published 14 shapes and make the extreme long-sequence case run instead of mixing it with ordinary resident cases. | Rebuilt the workload contract, isolated Shape 14 in a streamed executor, and screened policy/microbatch pairs without a dense `S x S` allocation. | Shapes 1–13 remain paired; Shape 14 completes the logical batch in 17.207 s with 7.307 GiB peak allocation and is labelled provisional/Target-only. | `37a4d94`, `59a10b5`, [final result](results/final/nvidia_geforce_rtx_4080.json) |
| Separate workloads needing a new method from those needing engineering refinement, while preserving existing winners. | Added guarded compiled-forward routes, mixed residual-normalization Triton paths for Shapes 5/6, version-aware Shape 5 Graph staging, non-persistent Shape 8 FP16 shadow weights, and a Shape 13 online causal-attention Triton kernel. | All 13 resident shapes pass five comparator trials; measured routes reach 9.192x geometric-mean speedup. Library GEMMs and cuDNN remain attributed to their providers. | [current implementation](solution/transformer.py), [final result](results/final/nvidia_geforce_rtx_4080.json) |
| Keep the architecture readable after repeated optimization rounds. | Removed legacy policy lists and hidden control channels, centralized policy definitions, and made the forward consume one immutable execution plan. | Tests protect registry/plan/route identity and observed execution; no compatibility copy of the old design remains in the public tree. | `9ba848b`, [policy registry](policy_registry.py), [execution plan](solution/execution_plan.py) |

### 9.4 Agent boundary

The planned in-project hardware-aware Agent runtime is **deferred and not
implemented**. It is not required in the inference hot path and is not claimed
as a completed contribution. The deterministic probe, measurement, calibration
and dispatch services can later be called by such an Agent without changing
the current competition implementation.

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
- Shape 14 uses library-provided cuDNN SDPA. The project custom online-attention
  kernel is an exact Shape 13 specialization and is not used for Shape 14; the
  repository has no external `flash-attn` dependency.
- Peak allocation is a PyTorch device-allocation measurement, not total board
  memory consumption. Logical traffic is an operator estimate, not a profiler
  DRAM counter.
- Current accuracy evidence covers the recorded seeded FP32 variant. Different
  masks, padding, scales, dtypes or training/autograd behavior require separate
  validation.
- Cross-hardware calibration is implemented, but performance conclusions have
  not been verified on a second GPU in this report.
