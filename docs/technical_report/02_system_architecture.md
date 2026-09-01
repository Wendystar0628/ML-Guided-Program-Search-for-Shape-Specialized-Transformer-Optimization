# 2. System Architecture

## 2.1 Architectural overview

![Closed-loop system architecture](figures/architecture_overview.svg)

*Figure 3. Typed construction, isolated multi-fidelity measurement, paired promotion, exact-device deployment, and runtime resolution form one closed loop.*

Generated programs are statically validated, measured on the target GPU, and written to the deployment registry only after Formal approval. The runtime then executes the same plan that won the comparison.

## 2.2 Runtime flow

1. The official Shape, run variant, and detected environment fingerprint define the execution context.
2. The deployment registry resolves an exact environment-and-Shape incumbent or a conservative executable fallback.
3. The search space generates a typed configuration. `PlanBuilder` emits one immutable `ExecutionPlan` or rejects the configuration before GPU work.
4. The model executes the named library operators, Triton kernels, and runtime wrappers. Expected and actual traces must agree, preventing a fallback from masquerading as the candidate.
5. Three measurement fidelities—Screen, Enhanced, and Formal—narrow candidates. A paired Formal winner replaces only its matching entry in `deployment/deployed_configs.json`.
6. Ordinary model construction resolves the updated registry, closing the loop.

## 2.3 Responsibility boundaries

| Layer | Repository area | Responsibility |
| --- | --- | --- |
| Program representation and compilation | `solution/config.py`, `solution/plan.py`, `solution/plan_builder.py` | Typed choices, immutable plan, and static legality |
| Runtime execution | `solution/transformer.py`, `solution/operators/`, `solution/kernels/`, `solution/runtimes/` | Execute the selected plan without a second policy lookup |
| Search | `autotune/search_space.py`, `autotune/search_engine.py` | Generate structures, sample schedules, and allocate measurements |
| Measurement | `benchmarking/` | Correctness, CUDA-event latency, memory, device lease, and fresh processes |
| Promotion and deployment | `autotune/promotion.py`, `deployment/` | Paired decision and exact environment/Shape registry |
| Entrypoints and evidence | `cli.py`, `scripts/`, `observations/` | Bounded workflows, persistent studies, and compact decision records |

## 2.4 Resident execution

Shapes 01–13 combine four classes of building blocks:

- mature PyTorch linear algebra and SDPA with focused Triton attention, projection, FFN, and normalization kernels;
- layout/materialization choices and cross-operator fusions;
- FP32 boundaries with legal internal precision choices;
- eager, compiled, CUDA Graph, or batch-tiled CUDA Graph runtimes.

Only combinations accepted by `PlanBuilder` become runnable candidates.

![Shape-specialized deployed programs](figures/deployed_program_matrix.svg)

*Figure 4. Each official Shape maps to a measured combination of runtime, attention, layout, projection, FFN, normalization/fusion, and precision choices.*

## 2.5 Shape 14 as a separate execution regime

Shape 14 (`B=32`, `S=100000`, `D=1024`, 16 heads, two layers) cannot materialize a dense attention score tensor on the validated 16 GB-class GPU. It therefore uses:

- an independent 34-point finite search over streamed Triton and bounded native-SDPA branches;
- ordered microbatch execution into one output tensor, with tiled online attention that never stores a global `S × S` matrix;
- separate search evidence and benchmarking, rejoining the common system at Formal comparison, deployment, and the output contract.

This is a capacity-driven execution regime within the same architecture, not an unrelated second system.

## 2.6 Isolation and evidence

All GPU-facing commands acquire one device lease. Shape suites run serially, with each Shape in a fresh process. Resident and Shape-14 search histories are stored separately; compact JSONL records retain coverage, failures, stage time, challenger identity, paired ratios, and the deployment decision. `deployment/deployed_configs.json` stores current winners rather than historical runs.
