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

The two memory values are API-specific reports of the same 16 GB-class device. The
project environment script keeps Torch extension, Triton, and TorchInductor caches
inside the repository workspace and targets `sm_89`. Nsight Compute and Nsight
Systems are exposed when installed.

## 5.2 Software design dependencies

The implementation deliberately uses a small external dependency set. PyTorch
provides the reference semantics, tensor operations, native SDPA, compilation, and
CUDA Graph integration. Triton provides focused custom GPU kernels. Optuna supplies
the mature TPE implementation. Ninja and the native compiler toolchain support
compiled paths. The project avoids adding a separate orchestration framework around
the search loop.

## 5.3 AI tools and models actually used

| Tool or capability | Use in this project |
| --- | --- |
| OpenAI Codex | Primary code implementation, refactoring, test execution, repository inspection, and multi-agent task coordination |
| ChatGPT GPT-5.6 sol, Pro reasoning | Deep repository and methodology reviews used as design input |
| Deep Research | Broader method and performance-strategy exploration |
| Browser control | Submitted the accessible private repository to the separate Pro review workflow and retrieved the analysis |

The exact model used in every historical Codex task is not encoded reliably in the
repository, so this report does not assign one model name to all Codex sessions.
Multi-agent delegation is described as a Codex workflow capability, not as a separate
project Agent Skill.

## 5.4 Agent Skills actually used

| Skill | Role |
| --- | --- |
| Stop That Shit | Kept implementation work bounded and prevented audit, packaging, or speculative infrastructure from displacing the performance mainline |
| Deep Research | Supported evidence-oriented exploration of candidate algorithms and optimization methods |
| Browser Control | Operated the authorized web review workflow |
| Nature Figure | Generated the editable academic architecture, performance, and workload figures in this report |

Only tools and Skills actually used are listed. The repository does **not** claim to
contain an autonomous LLM Agent runtime. Candidate generation, correctness, timing,
comparison, persistence, and deployment are deterministic program operations; AI
assistance was used during development to propose, review, and refine changes.

## 5.5 Limitations

1. **Single validated platform.** Performance conclusions are specific to the
   disclosed native-Windows RTX 4080 stack. Hardware probing and portable fallbacks
   exist, but another device needs fresh search and measurement.
2. **No official score claim.** MFU weights, useful-FLOP accounting, peak denominator,
   bandwidth correction, and cross-device normalization remain unpublished.
3. **Shape-14 baseline boundary.** The dense `S × S` baseline is intentionally not
   materialized; Shape 14 currently supports feasibility, latency, memory, and a
   project FLOP estimate rather than baseline speedup.
4. **Project-defined measurement details.** Median, P90, warmup, repeat counts, and
   isolation are explicit, but the final official timing protocol remains incomplete.
5. **Limited input variant.** The declared snapshot uses FP32 input/output with zero
   padding and unit input scale. Future official mask or input updates require fresh
   validation.
6. **Finite implementation vocabulary.** Program search can combine only the library
   and Triton primitives currently implemented. New primitives can enlarge the legal
   search space without changing the search architecture.
7. **Bounded high-level coverage.** Resident runs retain at most 36 structure branches;
   seeded rotation improves coverage across cycles but does not make one run exhaustive.
8. **No learned cross-device model.** Cross-Shape warm starts use deterministic
   similarity, and cross-hardware infrastructure is retained, but no trained
   performance-transfer model is claimed.

## 5.6 High-value future work

- add a second physical GPU architecture and rebuild deployment evidence rather than
  extrapolating RTX 4080 winners;
- report official MFU and bandwidth-normalized metrics once their definitions are
  published;
- extend the program vocabulary where profiler evidence shows a structural ceiling,
  especially the weakest resident Shapes;
- collect repeated exclusive-device final snapshots if confidence intervals across
  independent runs become a submission requirement;
- validate any final official padding, mask, input-scale, and Shape-14 reference data.
