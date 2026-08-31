# 3. Search and Optimization Method

## 3.1 Search object

A candidate is represented as:

\[
\text{ConfigSpec} = \text{ProgramConfig} + \text{ScheduleConfig}.
\]

`ProgramConfig` describes the computation graph: attention backend, precision, layout/materialization, output bridge, FFN implementation, normalization and fusion. `ScheduleConfig` describes execution details such as runtime backend, compile mode, Triton tile/warp/stage parameters, batch tiling, and microbatch size. A stable configuration identifier follows from the complete typed value.

For Shape \(s\), the measured search problem is the constrained black-box objective

\[
\min_{x\in\mathcal{X}_s}\; \operatorname{median} T_s(x)
\quad \text{subject to} \quad
g_{acc}(x)\le 0,\; g_{path}(x)\le 0,\; g_{run}(x)\le 0.
\]

Here \(x\) is a complete `ConfigSpec`. Static `PlanBuilder` checks remove structurally illegal points before GPU work; the three measured constraints then encode comparator accuracy, expected-versus-actual execution-path agreement, and runtime feasibility. This formulation makes the optimization target explicit without assigning invalid programs an attractive synthetic latency.

## 3.2 Generated resident search space

The resident generator forms high-level structures from legal primitive products, applies semantic and hardware constraints, and exposes only schedule axes relevant to that structure. It keeps at most 36 structure branches in one run. Required control and incumbent-compatible branches are retained first; remaining capacity is selected to cover primitive values, then pairwise primitive interactions, and finally a seed-rotated remainder.

This is a bounded covering design, not exhaustive enumeration. Repeated outer cycles can rotate optional structures by changing the structure seed. Explicit guards around specialized kernels define their legal implementation domains; they do not hard-code which complete program must win a Shape.

## 3.3 Constraint-aware branch-local TPE

Different structures expose different parameters, so each branch owns an independent persistent Optuna study. The sampler is multivariate, grouped, reproducibly seeded, and feasibility-aware. Startup size is:

\[
n_{startup}=\min(10, |\mathcal{X}_{branch}|).
\]

Only Screen measurements train TPE. The minimized objective is measured median latency, while three constraint coordinates separately encode:

1. comparator accuracy violation;
2. expected-versus-actual execution-path disagreement;
3. runtime feasibility.

Let \(y^*\) be the TPE split threshold for observed Screen latency \(y\). The original TPE construction models

\[
\ell(x)=p(x\mid y<y^*),\qquad
g(x)=p(x\mid y\ge y^*).
\]

For a minimization problem, its expected-improvement criterion is proportional to

\[
\left[\gamma+(1-\gamma)\frac{g(x)}{\ell(x)}\right]^{-1},
\qquad \gamma=p(y<y^*),
\]

so candidate ranking favors a large \(\ell(x)/g(x)\). The project delegates this estimator to Optuna's constrained, multivariate, grouped [`TPESampler`](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html); keeping one persistent study per structural branch avoids fitting one density model across incompatible active parameters. This is sequential model-based optimization, not a Gaussian-process posterior or a learned cross-Shape performance model.

Duplicate zero-information proposals are not rewarded or assigned synthetic timing. They are replaced by an unseen finite point when one remains.

## 3.4 Fixed-budget survivor racing

The nominal per-Shape wall-time budget uses cumulative soft deadlines at 65% for Screen, 82% for Enhanced, and 100% for Formal. An evaluation already in flight is allowed to finish. Every mandatory structure first receives a Screen witness. Branches are ranked by their best feasible measured Screen median; the scheduler does not invent future learning-curve gains.

With a trial cap, the largest ranked survivor prefix that can reach TPE startup and receive at least one guided proposal is selected. Roughly 10% of the remaining trial budget is reserved for least-sampled alternatives, while the rest is distributed round-robin over survivors. Without a trial cap, all branches stay active until the soft deadline.

The method is accurately described as **fixed-budget survivor TPE**. It is not a rising-bandit scheduler and does not claim a learned cross-Shape routing model.

The allocation follows the mature fixed-budget best-arm-identification principle: spread cheap evidence broadly, then spend scarce measurements on the most promising alternatives while retaining an exploration floor. The implementation is nevertheless a racing-inspired engineering adaptation, not Successive Halving, Hyperband, or F-Race: fidelity here changes the GPU measurement protocol rather than training resources, and no published regret or sample-complexity guarantee is claimed for the chosen 65/17/18 split.

## 3.5 Multi-fidelity selection

| Fidelity | Purpose | Persistence |
| --- | --- | --- |
| Screen | Broad, inexpensive candidate evidence and TPE training | Stored in the branch study |
| Enhanced | Remeasure a small feasible frontier with a stronger protocol | Reused only when evidence identities match |
| Formal | Lock one challenger and compare it with the incumbent | Always measured again |

For resident Shapes, the fastest 20% of eligible Screen candidates, capped at eight, advance to Enhanced. Shape 14 advances only its best Screen candidate because one full logical-batch evaluation dominates wall time. The fastest feasible Enhanced candidate is locked before Formal measurement, preventing post-hoc challenger selection from the Formal samples.

![Observed multi-fidelity search flow](figures/search_evidence.svg)

*Figure 5. Four resident optimization cycles reduce 3,933 Screen stage entries to 381 Enhanced entries, 50 Formal comparisons, and six deployment updates, while Screen measurement consumes most observed stage time.*

Across four consecutive resident cycles, 3,933 Screen stage entries narrowed to 381 Enhanced entries, 50 Formal comparisons, and six deployment updates. One complete cycle shows that Screen measurement dominates wall time for most Shapes, which is exactly where branch-local TPE and survivor allocation must spend their budget carefully. These are stage-entry counts rather than globally unique programs, and Enhanced evidence may be reused only when its evidence identity still matches. Because historical studies span code versions and evidence identities, the figure intentionally reports the auditable funnel and stage cost rather than claiming a single project-wide convergence curve.

## 3.6 Sequential paired promotion

Formal comparison alternates incumbent/challenger ordering in paired blocks. Each block produces the ratio

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

Under the working null

\[
H_0:\;P(r_i\ge 1.02)\le \tfrac{1}{2},
\]

and independent paired-block indicators, the three pre-specified looks have the conservative union bound

\[
\begin{aligned}
P(\text{false promotion at any look})
&\le 2^{-6}
+\frac{\binom{9}{8}+\binom{9}{9}}{2^9}
+\frac{\binom{13}{11}+\binom{13}{12}+\binom{13}{13}}{2^{13}}\\
&=0.0463867<0.05.
\end{aligned}
\]

The stricter 1.10 and 1.05 early events are subsets of the base 1.02 event used by the null. This is a conservative **per-comparison false-promotion bound**, not a latency confidence interval, a probability that the challenger is faster, or a family-wise guarantee across every Shape and challenger searched by the project. The 1.02 ratio is a pre-specified minimum-effect gate (about 1.96% lower challenger latency), not the significance level. Alternating AB/BA order reduces shared temporal drift but does not itself prove block independence.

A challenger is rejected early when it can no longer reach the final 11-win condition. Both programs must remain feasible. When no incumbent exists, one feasible Formal result may establish the first deployment; that initialization does not receive the paired-comparison guarantee.

This design spends fewer measurements on a clear improvement and reserves the full 13 blocks for close decisions.

## 3.7 Transfer, memory, and stopping

Cross-Shape warm starts are deterministic nearest-neighbour seeds. Shape distance is standardized Euclidean distance over log-scaled batch, sequence length, model width, head count, and FFN width, plus layer count. Up to four compatible seeds are selected round-robin from the nearest three source Shapes. The priority is a current Formal winner, then a checked-in deployment, then historical feasible Screen evidence.

Search identity includes Shape, branch, measurement environment, and evidence semantics. Resident and Shape-14 histories are stored separately. The outer loop increments the seed per sweep and stops after an absolute iteration limit, failure, interruption, or a configured plateau with neither a deployment update nor new Screen evidence. Time limits are soft at the measurement boundary: an already-started GPU evaluation is allowed to finish.

## 3.8 Shape-14 finite search

Shape 14 uses no TPE. Its high-value finite space contains 34 points before static rejection:

- two Triton streaming-Dh64 branches, each with four attention tiles, two warp counts, and two microbatch choices (16 points per branch);
- one native causal-SDPA branch with two microbatch choices.

All surviving points are mandatory and are enumerated without replacement across persistent runs. The portable streamed implementation remains a correctness fallback, not a performance challenger.

## 3.9 Theoretical lineage and scope

The implementation transfers a small number of established ideas into GPU program search:

- Bergstra et al., [*Algorithms for Hyper-Parameter Optimization*](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization), NeurIPS 2011: TPE density-ratio and expected-improvement derivation.
- Jamieson and Talwalkar, [*Non-stochastic Best Arm Identification and Hyperparameter Optimization*](https://proceedings.mlr.press/v51/jamieson16.html), AISTATS 2016: fixed-budget allocation toward promising alternatives.
- Fleming, Harrington, and O'Brien, [*Designs for Group Sequential Tests*](https://doi.org/10.1016/S0197-2456(84)80014-8), 1984: the general principle of pre-specified interim looks with stricter early evidence. The project's exact discrete thresholds are its own binomial/union-bound construction above, not an implementation of a named clinical-trial boundary.

These sources justify the method structure, not the numerical optimality of the branch cap, startup count, 65/17/18 deadlines, 20% Enhanced frontier, or 2% deployment margin. Those are pre-specified engineering choices whose value must be judged by measured search efficiency and deployment stability.
