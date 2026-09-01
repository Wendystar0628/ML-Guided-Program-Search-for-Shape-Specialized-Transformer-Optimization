# 5. Environment, AI Collaboration, and Limitations

## 5.1 Validated runtime environment

| Component | Validated value |
| --- | --- |
| Operating system | Microsoft Windows 11 Pro, version 10.0.26200, build 26200 |
| CPU | Intel Core i7-14700KF |
| System memory | 63.8 GiB |
| GPU | NVIDIA GeForce RTX 4080, compute capability 8.9, 76 SMs |
| GPU memory | 16 GB-class; 16,376 MiB from `nvidia-smi`, 17,170,956,288 bytes from PyTorch |
| Storage | SOLIDIGM SSDPFKNU020TZ 2 TB NVMe; NTFS workspace volume |
| NVIDIA driver | 610.88 |
| Python | 3.12.5 |
| PyTorch | 2.12.1+cu132 |
| CUDA runtime | 13.2 |
| Triton | `triton-windows` 3.7.1.post27 |
| Search backend | Optuna 4.9.0 |
| Build support | Ninja 1.13.0 and Visual Studio C++ toolchain |

The two GPU-memory values are API-specific reports of the same device. Compiled paths target `sm_89`.

## 5.2 Software dependencies

PyTorch supplies the reference semantics, tensor operations, native SDPA, compilation, and CUDA Graph integration. Triton implements focused custom kernels; Optuna provides TPE; Ninja and the native compiler toolchain support compiled paths.

## 5.3 AI tools and models actually used

| Tool or capability | Use in this project |
| --- | --- |
| OpenAI Codex | Primary implementation, refactoring, testing, repository inspection, and multi-agent coordination |
| ChatGPT GPT-5.6 sol, Pro reasoning | Independent repository and methodology reviews used as design input |
| Deep Research | Broader exploration of search, measurement, and performance strategies |
| Browser control | Operated the authorized private-repository review workflow and retrieved its analysis |

Historical Codex tasks do not reliably encode one model name for every session, so no single model is attributed to all Codex work.

## 5.4 Agent Skills actually used

| Skill | Role |
| --- | --- |
| Stop That Shit | Kept implementation focused on requested and performance-relevant work |
| Deep Research | Supported evidence-oriented exploration of algorithms and optimization methods |
| Browser Control | Operated the authorized web review workflow |
| Nature Figure | Guided scientific-figure structure, accessibility, and export QA |

AI assistance proposed, reviewed, and implemented candidate changes. Deterministic project code performed candidate generation, correctness checks, timing, promotion, persistence, and deployment.

## 5.5 Human guidance and decision ownership

The human participant set the problem framing, high-level architecture, optimization directions, hardware and time constraints, priorities, and acceptance decisions. AI tools related those directions to the official task and current repository, researched established methods, adapted them to the actual Shapes and RTX 4080, implemented selected changes, and returned measured evidence.

| Stage | Human responsibility | AI-assisted responsibility |
| --- | --- | --- |
| Direction | State the objective, constraints, candidate idea, and acceptable trade-offs | Connect the idea to the task and code; identify assumptions and alternatives |
| Design | Select or redirect the high-level approach | Produce a minimal executable design and falsifiable performance hypothesis |
| Execution | Authorize experiment scope and persistent GPU work | Implement, run, monitor, and summarize bounded workflows |
| Decision | Own project priorities and final claims | Recommend keep, reject, or revise from measured evidence |

The working loop was **human hypothesis and constraints → AI research, adaptation, and implementation → deterministic correctness and GPU measurement → human continuation or redirection**.

## 5.6 Representative interaction histories

These two thematic reconstructions preserve the decision sequence and measured consequences without reproducing unrelated conversation or private paths.

### 5.6.1 From a TPE proposal to conditional program search

**Human direction.** Replace the growing hand-written policy catalogue with a mathematically grounded search method, initially described as TPE plus Bayesian optimization. The search still had to generate real executable choices and remain understandable.

**AI adaptation.** The AI related the proposal to TPE, conditional spaces, and racing, then identified the failure modes of one flat model: incompatible active parameters, a large illegal Cartesian product, branch starvation, duplicate finite proposals, and conflation of cheap search evidence with deployment evidence.

**Implementation and outcome.** The result is the current typed architecture: `ConfigSpec` separates program and schedule choices; `PlanBuilder` rejects illegal combinations; compatible branches own persistent constrained TPE studies; only Screen evidence trains TPE; duplicate proposals are replaced by unseen points when possible; and Shape 14 uses a separate finite search. Enhanced measurement locks one challenger before the paired Formal comparison, preserving the original goal of programmatic search while correcting conditional-space and evidence-reuse errors.

### 5.6.2 Monitored optimization, bottleneck diagnosis, and restart

**Human direction.** Run the complete optimization loop to completion, distinguish repeated bottlenecks from one-off noise, repair blocking behavior, and restart. Resident Shapes and Shape 14 had to remain separate so the long-sequence case could not consume the ordinary search cycle.

**AI diagnosis.** Four resident cycles created 1,240 Studies, but only 51 of 956 decision-capable branches reached TPE startup; the median Study contained one completed trial. Earlier runs could also accumulate Screen observations without reliably reaching Enhanced, Formal, and deployment. The diagnosis used persistent Studies and compact decision records rather than terminal impressions.

**Implementation and measured outcome.** Fixed-budget survivor TPE now guarantees broad required coverage, funds ranked survivors through startup and a guided ask, and preserves a small exploration reserve. Compatible evidence can advance through the closed loop; measurements are serialized under one GPU lease in fresh processes. Subsequent cycles deployed new winners across several Shapes, including a Shape-11 promotion after 13 paired blocks at a 1.0406 median ratio. The declared full-result snapshot reached the current 14.49× resident geometric mean.

## 5.7 Limitations

1. **Competition-core scope.** The project optimizes the forward-only Transformer core, not end-to-end training or serving systems with tokenization, embeddings, KV-cache decoding, backward computation, communication, concurrency, or application I/O.
2. **Finite search and synthesis vocabulary.** Approximately 12 hours of cumulative search explored only part of the combinatorial space. Results are best-so-far rather than global optima, and the system cannot discover a fundamentally new Kernel family until that primitive is implemented. Compilation and isolated benchmarking also require dedicated offline GPU time.
3. **Incomplete interaction ablation.** Legal one-family removals cover every resident Shape, but the available competition time did not permit a full combinatorial decomposition. The reported values are removal sensitivities, not additive contribution shares.
4. **Shape 14 official oracle unavailable.** The streamed path passes a local `B=1` semantic check and completes the logical `B=32` workload, but the official `B=32` input/output artifact is absent. The execution digest is not a substitute for that comparison.
5. **Single-platform evidence.** Performance was measured on one native-Windows RTX 4080 system. Shape similarity and compatible history provide warm starts, but no transferable cross-device performance model has been learned.

## 5.8 High-value future work

- run longer searches across additional structure seeds and report best-so-far improvement curves;
- validate Shape 14 against the official full-batch artifact when available;
- profile structural ceilings and introduce genuinely new executable Kernel families;
- collect multi-device evidence for a transferable cost model and evaluate selected plans in a realistic serving workload.
