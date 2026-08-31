# 5. Environment, AI Tools, and Limitations

## 5.1 Validated runtime environment

| Component | Validated value |
| --- | --- |
| Operating system | Microsoft Windows 11 Pro, version 10.0.26200, build 26200 |
| CPU | Intel Core i7-14700KF |
| System memory | 63.8 GiB |
| GPU | NVIDIA GeForce RTX 4080, compute capability 8.9, 76 SMs |
| GPU memory | 16 GB-class; 16,376 MiB from `nvidia-smi`, 17,170,956,288 bytes from PyTorch |
| NVIDIA driver | 610.88 |
| Python | 3.12.5 |
| PyTorch | 2.12.1+cu132 |
| CUDA runtime | 13.2 |
| Triton | `triton-windows` 3.7.1.post27 |
| Search backend | Optuna 4.9.0 |
| Build support | Ninja 1.13.0 and Visual Studio C++ toolchain |

The two memory values are API-specific reports of the same 16 GB-class device. The project environment script keeps Torch extension, Triton, and TorchInductor caches inside the repository workspace and targets `sm_89`. Nsight Compute and Nsight Systems are exposed when installed.

## 5.2 Software design dependencies

The implementation deliberately uses a small external dependency set. PyTorch provides the reference semantics, tensor operations, native SDPA, compilation, and CUDA Graph integration. Triton provides focused custom GPU kernels. Optuna supplies the mature TPE implementation. Ninja and the native compiler toolchain support compiled paths. The project avoids adding a separate orchestration framework around the search loop.

## 5.3 AI tools and models actually used

| Tool or capability | Use in this project |
| --- | --- |
| OpenAI Codex | Primary code implementation, refactoring, test execution, repository inspection, and multi-agent task coordination |
| ChatGPT GPT-5.6 sol, Pro reasoning | Deep repository and methodology reviews used as design input |
| Deep Research | Broader method and performance-strategy exploration |
| Browser control | Submitted the accessible private repository to the separate Pro review workflow and retrieved the analysis |

The exact model used in every historical Codex task is not encoded reliably in the repository, so this report does not assign one model name to all Codex sessions. Multi-agent delegation is described as a Codex workflow capability, not as a separate project Agent Skill.

## 5.4 Agent Skills actually used

| Skill | Role |
| --- | --- |
| Stop That Shit | Kept implementation work bounded and prevented audit, packaging, or speculative infrastructure from displacing the performance mainline |
| Deep Research | Supported evidence-oriented exploration of candidate algorithms and optimization methods |
| Browser Control | Operated the authorized web review workflow |
| Nature Figure | Generated the editable academic architecture, performance, and workload figures in this report |

Only tools and Skills actually used are listed. The repository does **not** claim to contain an autonomous LLM Agent runtime. Candidate generation, correctness, timing, comparison, persistence, and deployment are deterministic program operations; AI assistance was used during development to propose, review, and refine changes.

## 5.5 Human guidance and decision ownership

The collaboration followed a human-directed engineering loop. The human participant supplied the problem framing, high-level architectural concepts, optimization directions, competition priorities, hardware and time constraints, and explicit preferences such as keeping the implementation performance-centred and avoiding unnecessary infrastructure. The AI then inspected the official requirements, current repository, GPU evidence, and relevant established methods; expanded the proposal into implementable alternatives; identified likely failure modes; adapted the design to the actual Shapes and RTX 4080 environment; implemented and exercised the selected change; and returned measured results and remaining trade-offs for the next human decision.

| Stage | Human responsibility | AI-assisted responsibility |
| --- | --- | --- |
| Direction | State the objective, constraints, candidate idea, and acceptable trade-offs | Relate the idea to the official task and current code; research established methods; expose missing assumptions and alternatives |
| Design | Select or redirect the high-level approach | Convert the direction into a minimal executable architecture, implementation plan, and falsifiable performance hypothesis |
| Execution | Authorize the experiment scope and persistent GPU work | Implement changes, run correctness and timing programs, monitor bounded searches, and summarize evidence |
| Decision | Retain ownership of project priorities and final claims | Recommend keep, reject, or revise from measured evidence rather than model confidence |

Candidate correctness, timing, paired comparison, and registry updates were performed by deterministic project code. AI suggestions did not become deployments by assertion: they had to enter the same generated program space and pass the same GPU evidence path as any other challenger. The practical loop was therefore **human hypothesis and constraints → AI research, adaptation, and implementation → deterministic correctness and GPU measurement → AI synthesis of the evidence → human continuation or redirection**. This boundary is also why the project is not described as an autonomous LLM Agent runtime.

## 5.6 Representative interaction histories

The following are concise thematic reconstructions of two continuous development threads, not verbatim chat transcripts. They omit private machine paths and unrelated conversation while preserving the actual human direction, AI reasoning, implementation consequences, and measured feedback.

### 5.6.1 From a TPE proposal to conditional program search

**Human direction.** The human proposed replacing the growing hand-written policy catalogue with a mathematically grounded search method, initially framed as TPE plus Bayesian optimization. The additional constraints were that the method must search real executable choices, remain understandable, and avoid building a large control plane around performance work.

**AI expansion and correction.** The AI reviewed established TPE, sequential model-based optimization, conditional-space, and racing methods against the repository. It clarified that TPE is itself a model-based Bayesian optimization method, then identified the weaknesses of applying one flat TPE model to this project: different program structures expose different parameters; illegal combinations can dominate the nominal Cartesian product; cold-start branches can be starved; duplicate finite proposals carry no new information; and cheap Screen evidence must not be confused with stronger deployment evidence.

**Implemented adaptation.** The proposal became the current typed search architecture. `ConfigSpec` separates structural `ProgramConfig` choices from `ScheduleConfig`; `PlanBuilder` rejects semantically or physically invalid combinations; structurally distinct branches use independent persistent Optuna TPE studies; only Screen results train TPE; and repeated proposals are replaced by unseen finite points when possible. Shape 14 uses a separate finite no-replacement search because its streamed execution cost and parameter domain do not match resident Shapes. These mechanisms are implemented in [`search_space.py`](../../autotune/search_space.py), [`optuna_backend.py`](../../autotune/optuna_backend.py), and [`search_engine.py`](../../autotune/search_engine.py).

**Feedback and outcome.** GPU measurements, rather than the original concept label, determined whether the design was useful. Persistent compatible studies supplied prior evidence, Enhanced measurement locked one challenger, and Formal paired comparison controlled deployment. The result preserved the human goal—programmatic Bayesian search instead of manually extending policy names—while correcting the parts that would have caused conditional-space bias, repeated exploration, and misleading evidence reuse.

### 5.6.2 Monitored optimization, bottleneck diagnosis, and restart

**Human direction.** After the base architecture was executable, the human required the AI to run the complete optimization workflow, continue observing it until completion, distinguish repeated bottlenecks from one-off noise, modify the project when the closed loop was blocked, and then restart the optimization. Resident Shapes and Shape 14 were kept as separate lifecycles so that the very expensive streamed workload could not dominate the ordinary search cycle.

**Observed evidence.** The first four resident cycles spread measurements too thinly: they created 1,240 Studies, but only 51 of 956 decision-capable branches reached the TPE startup threshold, and the median Study contained one completed trial. The search appeared busy while most branches never accumulated enough evidence for a model-guided proposal. Earlier workflow runs could also accumulate Screen evidence without reliably reaching Enhanced measurement, the locked Formal comparison, and deployment. The AI used the compact JSONL records, persistent Studies, and the [optimization-cycle note](../../notes/2026-08-31_optimization_cycle.md)—not terminal impressions alone—to compare stage timings, branch coverage, new-trial counts, failure categories, configuration identities, and decision outcomes.

**Implemented correction.** The AI changed the resident scheduler to fixed-budget survivor TPE. Required branches still receive coverage, ranked survivors are funded through their startup set and at least one genuinely TPE-guided ask, and a small exploration reserve prevents all non-survivors from being starved. The workflow was also closed so compatible Screen evidence could advance through Enhanced and Formal instead of triggering another unnecessary Screen-only pass; persistent Studies were scoped by evidence identity; duplicate configurations were tracked; resident Shapes were serialized under one device lease and measured in fresh processes; Shape 06 became an explicit opt-in for routine structure rotation; and Shape 14 retained its separate bounded script. The relevant boundaries are visible in [`search_engine.py`](../../autotune/search_engine.py), [`device_isolation.py`](../../benchmarking/device_isolation.py), [`promotion.py`](../../autotune/promotion.py), and the two [`scripts`](../../scripts/).

**Restart and measured feedback.** The workflow was rerun with rotating structure seeds and compatible prior evidence. Three subsequent cycles published winners for Shapes 10/13, 07/09, and 02/07; the next cycle produced no deployment, which correctly triggered a shift from repeating the same broad sweep toward adding new executable primitives. A later [recorded fourth resident iteration](../../observations/search/resident/logs/20260830T222524.582018Z_optimize_resident.jsonl) promoted the Shape 11 challenger after a 13-block paired comparison with a 1.0406 median paired ratio, while weaker challengers retained their incumbents. Subsequent full-result measurement produced the currently declared 14.49× geometric-mean speedup for Shapes 01–13. The important outcome was not that every restart found a winner, but that each completed run produced a reproducible search-to-deployment decision and useful evidence for the next human hypothesis.

These histories describe AI-assisted engineering, not AI authority over the result. The AI monitored, diagnosed, implemented, and summarized; deterministic optimization and promotion code made measurement-based deployment decisions; and the human continued to set priorities, constraints, and project direction.

## 5.7 Limitations

1. **Single validated platform.** Performance conclusions are specific to the disclosed native-Windows RTX 4080 stack. Hardware probing and portable fallbacks exist, but another device needs fresh search and measurement.
2. **No official score claim.** MFU weights, useful-FLOP accounting, peak denominator, bandwidth correction, and cross-device normalization remain unpublished.
3. **Shape-14 baseline boundary.** The dense `S × S` baseline is intentionally not materialized; Shape 14 currently supports feasibility, latency, memory, and a project FLOP estimate rather than baseline speedup.
4. **Project-defined measurement details.** Median, P90, warmup, repeat counts, and isolation are explicit, but the final official timing protocol remains incomplete.
5. **Limited input variant.** The declared snapshot uses FP32 input/output with zero padding and unit input scale. Future official mask or input updates require fresh validation.
6. **Finite implementation vocabulary.** Program search can combine only the library and Triton primitives currently implemented. New primitives can enlarge the legal search space without changing the search architecture.
7. **Bounded high-level coverage.** Resident runs retain at most 36 structure branches; seeded rotation improves coverage across cycles but does not make one run exhaustive.
8. **No learned cross-device model.** Cross-Shape warm starts use deterministic similarity, and cross-hardware infrastructure is retained, but no trained performance-transfer model is claimed.

## 5.8 High-value future work

- add a second physical GPU architecture and rebuild deployment evidence rather than extrapolating RTX 4080 winners;
- report official MFU and bandwidth-normalized metrics once their definitions are published;
- extend the program vocabulary where profiler evidence shows a structural ceiling, especially the weakest resident Shapes;
- collect repeated exclusive-device final snapshots if confidence intervals across independent runs become a submission requirement;
- validate any final official padding, mask, input-scale, and Shape-14 reference data.
