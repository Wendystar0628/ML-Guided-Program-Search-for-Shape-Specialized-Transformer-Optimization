# 1. Problem Framing and Contributions

## 1.1 Challenge

The benchmark fixes one pre-normalized causal Transformer semantics across 14 public Shapes. Implementations may change internal precision and execution only if FP32 outputs satisfy the supplied elementwise tolerance.

The Shapes span batch size 1–10,000, width 32–1,024, head count 1–16, and sequence length 32–100,000. The dominant cost therefore shifts among launch overhead, matrix throughput, layout conversion, memory traffic, attention working set, and device capacity.

![Official workload regime map](figures/workload_landscape.svg)

*Figure 1. The official workloads occupy distinct launch-, throughput-, memory-, and capacity-limited regimes.*

![Measured workload sensitivity](figures/workload_sensitivity.svg)

*Figure 2. Independently deployed programs respond non-monotonically to batch size, width, head count, and sequence length. These points describe Shape sensitivity; they are not causal Kernel ablations.*

## 1.2 Core insight

The search object is a typed executable program: operator implementations; layout, materialization, and precision boundaries; compile or capture decisions; and schedule parameters. Candidate programs combine mature PyTorch operators with focused Triton kernels and are generated and validated by code rather than selected from a manually extended policy catalogue.

## 1.3 Contributions

1. **Typed program synthesis.** `ConfigSpec` separates structural choices from schedules, and `PlanBuilder` compiles each legal configuration into one immutable `ExecutionPlan` before GPU execution.
2. **Learning-guided conditional search.** Independent persistent TPE studies model structurally compatible branches; fixed-budget survivor racing concentrates measurements while retaining an exploration floor.
3. **Evidence-gated deployment.** Screen and Enhanced measurements narrow the frontier before an interleaved Formal comparison. Only an approved winner updates the exact-device registry; GPU work is serialized and each Shape is measured in a fresh process.
4. **Separate capacity regime for Shape 14.** The 100,000-token workload uses an independent finite search and streamed microbatch runtime rather than the resident execution lifecycle.

## 1.4 Scope

The artifact targets the 14 public Shapes on the validated RTX 4080. Its reusable contribution is an exact-device search-and-deployment workflow: a new environment is searched rather than assumed to inherit the RTX 4080 choices. This report makes no cross-hardware performance claim.
