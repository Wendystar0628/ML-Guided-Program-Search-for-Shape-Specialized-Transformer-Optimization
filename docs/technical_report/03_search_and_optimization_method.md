# 3. Search and Optimization Method

## 3.1 Search object

A candidate is represented as:

\[
\text{ConfigSpec} = \text{ProgramConfig} + \text{ScheduleConfig}.
\]

`ProgramConfig` describes the computation graph: attention backend, precision,
layout/materialization, output bridge, FFN implementation, normalization and fusion.
`ScheduleConfig` describes execution details such as runtime backend, compile mode,
Triton tile/warp/stage parameters, batch tiling, and microbatch size. A stable
configuration identifier follows from the complete typed value.

## 3.2 Generated resident search space

The resident generator forms high-level structures from legal primitive products,
applies semantic and hardware constraints, and exposes only schedule axes relevant to
that structure. It keeps at most 36 structure branches in one run. Required control
and incumbent-compatible branches are retained first; remaining capacity is selected
to cover primitive values, then pairwise primitive interactions, and finally a
seed-rotated remainder.

This is a bounded covering design, not exhaustive enumeration. Repeated outer cycles
can rotate optional structures by changing the structure seed. Explicit guards around
specialized kernels define their legal implementation domains; they do not hard-code
which complete program must win a Shape.

## 3.3 Constraint-aware branch-local TPE

Different structures expose different parameters, so each branch owns an independent
persistent Optuna study. The sampler is multivariate, grouped, reproducibly seeded,
and feasibility-aware. Startup size is:

\[
n_{startup}=\min(10, |\mathcal{X}_{branch}|).
\]

Only Screen measurements train TPE. The minimized objective is measured median
latency, while three continuous constraints independently encode:

1. comparator accuracy violation;
2. expected-versus-actual execution-path disagreement;
3. runtime feasibility.

Duplicate zero-information proposals are not rewarded or assigned synthetic timing.
They are replaced by an unseen finite point when one remains.

## 3.4 Fixed-budget survivor racing

The nominal per-Shape wall-time budget is divided into 65% Screen, 17% Enhanced, and
18% Formal time. Every mandatory structure first receives a Screen witness. Branches
are ranked by their best feasible measured Screen median; the scheduler does not
invent future learning-curve gains.

With a trial cap, the largest ranked survivor prefix that can reach TPE startup and
receive at least one guided proposal is selected. Roughly 10% of the remaining trial
budget is reserved for least-sampled alternatives, while the rest is distributed
round-robin over survivors. Without a trial cap, all branches stay active until the
soft deadline.

The method is accurately described as **fixed-budget survivor TPE**. It is not a
rising-bandit scheduler and does not claim a learned cross-Shape routing model.

## 3.5 Multi-fidelity selection

| Fidelity | Purpose | Persistence |
| --- | --- | --- |
| Screen | Broad, inexpensive candidate evidence and TPE training | Stored in the branch study |
| Enhanced | Remeasure a small feasible frontier with a stronger protocol | Reused only when evidence identities match |
| Formal | Lock one challenger and compare it with the incumbent | Always measured again |

For resident Shapes, the fastest 20% of eligible Screen candidates, capped at eight,
advance to Enhanced. Shape 14 advances only its best Screen candidate because one
full logical-batch evaluation dominates wall time. The fastest feasible Enhanced
candidate is locked before Formal measurement, preventing post-hoc challenger
selection from the Formal samples.

## 3.6 Sequential paired promotion

Formal comparison alternates incumbent/challenger ordering in paired blocks. Each
block produces the ratio

\[
r_i = \frac{\operatorname{median}(T_{incumbent,i})}
           {\operatorname{median}(T_{challenger,i})}.
\]

The pre-specified group-sequential rule is:

| Look | Promotion condition |
| ---: | --- |
| 6 blocks | 6 of 6 ratios are at least 1.10 |
| 9 blocks | at least 8 of 9 ratios are at least 1.05 |
| 13 blocks | at least 11 of 13 ratios are at least 1.02 |

Under the documented null assumption that a block has at most 0.5 probability of
meeting the 2% target, the union of the three one-sided false-promotion bounds remains
below 0.05. A challenger is rejected early when it can no longer reach the final
11-win condition. Both programs must remain feasible. When no incumbent exists, one
feasible Formal result may establish the first deployment.

This design spends fewer measurements on a clear improvement and reserves the full
13 blocks for close decisions.

## 3.7 Transfer, memory, and stopping

Cross-Shape warm starts are deterministic nearest-neighbour seeds. Shape distance is
standardized Euclidean distance over log-scaled batch, sequence length, model width,
head count, and FFN width, plus layer count. Up to four compatible seeds are selected
round-robin from the nearest three source Shapes. The priority is a current Formal
winner, then a checked-in deployment, then historical feasible Screen evidence.

Search identity includes Shape, branch, measurement environment, and evidence
semantics. Resident and Shape-14 histories are stored separately. The outer loop
increments the seed per sweep and stops after an absolute iteration limit, failure,
interruption, or a configured plateau with neither a deployment update nor new Screen
evidence. Time limits are soft at the measurement boundary: an already-started GPU
evaluation is allowed to finish.

## 3.8 Shape-14 finite search

Shape 14 uses no TPE. Its high-value finite space contains 34 points before static
rejection:

- two Triton streaming-Dh64 branches, each with four attention tiles, two warp
  counts, and two microbatch choices (16 points per branch);
- one native causal-SDPA branch with two microbatch choices.

All surviving points are mandatory and are enumerated without replacement across
persistent runs. The portable streamed implementation remains a correctness fallback,
not a performance challenger.
