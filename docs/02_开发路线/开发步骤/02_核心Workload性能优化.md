# 2. 核心 Workload 性能优化

> 本步骤覆盖全部官方 Shape，但拆成两个清晰机制：Shape 1–13 使用普通 Paired Benchmark/Sweep，Shape 14 使用独立 `benchmark-streamed`。二者共享 Solution 与 Policy/Kernel Registry。

## 1. Shape 视图

| 视图 | 对应 Shape | 变量 | 主要瓶颈假设 |
| --- | --- | --- | --- |
| Batch Sweep | 1–6 | `B=1,4,16,64,128,10000` | 小 Batch 的 Launch/Occupancy；大 Batch 的吞吐和峰值显存 |
| Width Sweep | 1、7、8 | `D=32,128,1024` | 小 GEMM 效率；宽 GEMM/FFN 与 Tensor Core |
| Head Sweep | 1、9–11 | `H=1,2,4,16` | `head_dim=128,64,32,8` 对 Attention Backend 和布局的影响 |
| Sequence Sweep | 1、12、13 | `S=32,128,1024` | 短序列 Launch；长序列 $S^2$ 中间量和 Softmax |
| Extreme Sequence | 14 | `B=32,S=100000,D=1024,H=16` | 完整 Batch 驻留、Attention 内存复杂度和长时计算吞吐 |

所有形状均为 Causal，不再为 Padding、Non-Causal 或自造 Dtype 组织另一条主线。

## 2. Hardware Profile 与白盒分析

Probe 采集用于候选选择的紧凑信号：

- GPU 型号、Compute Capability、SM、L2、显存容量与总线信息；
- Driver、PyTorch、CUDA Runtime 和操作系统；
- CUDA Launch、Graph Replay、Copy、代表性 GEMM、FP32 Softmax 和可用 SDPA Backend Anchor。

Shape 分析计算：

- QKV、Attention、Output Projection 与 FFN 的运算量；
- Input/Output、QKV、Score/Probability 与 FFN Activation 的数据规模；
- Head Dim、Token 数、潜在 Tensor Core 对齐、并行块与预计 Launch；
- 显式 Causal Mask 和 $S^2$ Attention 对峰值显存的影响；
- 官方 Dense Baseline、当前 Solution 与设备安全余量之间的可执行性差异。

成本模型只输出瓶颈类别、可读信号、排除原因和候选顺序。它不伪造精确延迟、预测 Speedup 或置信度。

## 3. 候选与 Policy

候选围绕少量可组合机制构建：通用 SDPA、CUDA Graph、Compiled/Triton Norm、Mixed-FP16 或 FP16 Shadow Core、Batch-tiled Graph、Compiled Forward，以及严格 Shape Guard 的专用 Triton Attention/Norm。具体组合随实测结果演进，因此完整 Policy ID 只在 `policy_registry.py` 维护，完整 Candidate ID、适用范围和执行证据只在 `runner/candidates.py` 维护。

Hardware Router 只根据 Shape、设备能力和瓶颈信号缩小候选范围；它不把理论排序当作赢家。每个组合仍需真实命中预期 Execution Plan、通过官方 Comparator，并在完整 Forward Formal 测量中胜出后才可部署。

`safe` 是官方等价的内部保守路径，只用于接口诊断、内部 Reference 和特化路径 Fallback；它不可部署，也不参与路由晋升。

Shape 14 的 `batch_streamed` 是 Runner 执行形态，不加入 Policy Registry。它用 `B=1` 完成 provisional Comparator，再比较 Mixed-Core Efficient/cuDNN 等可用 Policy 与 `1/2/4/8/16/32` 中合法且内存安全的 Timing Microbatch，按完整逻辑 Batch 估算延迟选择赢家，最后用同一锁定计划正式计时。

Candidate Registry 说明 Policy 映射、Shape Applicability、Hardware Capability、是否可晋升和必须观测到的 Execution Evidence。Policy Registry 说明 Transformer 如何将 Policy 解析为 Execution Plan。新候选不在多个白名单重复登记。

## 4. 按瓶颈推进

### Launch 与 Runtime

- 优先分析 `official_02`、`official_03`、`official_12`；
- 对比 `eager-sdpa`、`graph`，并在 Token 数不超过 2048、`D=FFN=128` 时加入 `graph-fused-norm`；
- 观察完整 CUDA Event 延迟与 ATen Kernel Self Time 差值；
- 避免为解决 Launch 问题先重写大型 GEMM。

### Attention 与 Causal 语义

- 所有 Shape 首先避免模型构建期的完整 Causal Mask；
- 核对 SDPA 实际 Backend、Head Dim 支持、转置和 Contiguous 开销；
- 在 `official_13` 观察 Score/Probability 物化与 Softmax 占比；
- 对 `S>=1024`、`head_dim=32/64` 的 FP32 Shape 比较适用的 Efficient、cuDNN 与通用 SDPA；
- 外部 FlashAttention 不作为安装依赖；借鉴其不物化完整 Score/Probability Matrix 的算法原则，优先使用当前 PyTorch 运行栈真实可用的内存高效后端；
- 自定义 Online Attention 或其他 Triton/CUDA Kernel 只在 Profile 和理论上限共同显示现有库后端仍有明确剩余空间时作为独立候选。

### Batch 与峰值显存

- `official_05`、`official_06` 重点分析 Activation 峰值、Attention Workspace 和吞吐；
- 在构建大张量前比较预计峰值与设备安全余量；
- 大 Batch 的 Norm Fusion 只作为 Mixed-Core 的局部组合候选，不单独复制一条 Transformer；自主 Triton 路径需要真实 Kernel 命中、Comparator 和完整 Forward 净收益，编译时间不进入正式计时；
- Shape 14 不运行完整 Dense Baseline；使用 `batch_streamed` 完成全部 32 个样本，并记录 Target-only Latency 与峰值显存；
- Batch Streaming 不改变 Transformer Policy。每个 Microbatch 仍由已注册 Attention Candidate 执行，避免把调度策略和数学 Kernel 混成一层。

### Shape 14 流式主线

1. 用 `B=1` 模型实例和输入建立内存有界的正确性工作集。
2. 用 Query-block Safe Reference 检查一个完整样本，结果标记为 `provisional`。
3. 对 Efficient、cuDNN 及其 Mixed-Core 组合运行完整样本 Comparator，淘汰后端或观测证据失败的候选。
4. 在显存守卫下短测 Policy × Timing Microbatch，按 `单次 Forward Median × Microbatch Count` 选择完整逻辑 Batch 赢家。
5. 使用选中计划串行处理完整 Batch，以 CUDA Event 记录设备时间，并另记 Host-to-Device、Compute、Device-to-Host 端到端时间。
6. 保存 Target-only Median/P90、峰值显存、Useful FLOPs、Achieved FLOP/s、项目估算 MFU、实际后端和 Timing Microbatch。
7. 后续接入正式全量正确性来源时替换 provisional Comparator，再决定是否进入设备精确路由。

### GEMM、FFN 与中间写回

- `official_08` 用于观察宽 QKV/Output/FFN GEMM 与 Exact GELU；
- 保留 cuBLAS/cuBLASLt 作为 GEMM 吞吐基线，重点比较 Layout、Epilogue、临时张量和局部 Compile；
- Residual/Norm 局部融合与 GEMM/FFN 分开评估；Compiled、Triton Mixed Norm 和 Initial Norm 均由严格 Guard 与完整 Forward 实测决定是否组合；
- 一般不重写通用 GEMM，除非 Profile 和理论分析同时说明库 Kernel 存在大幅空间。

### Head Dim 和布局

- `official_09`–`official_11` 区分 `head_dim=128,64,8`；
- 检查 QKV 重排、Stride、Alignment、Warp 利用和 SDPA 能力门控；
- 允许不同 Head Dim 选择不同现有算法族，不为每个 Head 数复制一套 Transformer。

## 5. GPU 迭代

1. 运行当前 Baseline 和 Dispatcher，找出慢、不稳定或未命中特化路径的 Shape。
2. 用 Profile 或局部实验区分计算、带宽、Launch、布局与峰值显存问题。
3. 选择一个机制和少量参数，不同时引入多个不可归因改动。
4. 运行官方 Comparator 和短完整 Forward 计时。
5. 有效时扩大到相关 Shape；无效时删除候选或回到理论上限分析。
6. 运行默认 1–13 Shape Sweep 检查路由回归，再单独运行 `benchmark-streamed` 确认 Shape 14 能完整结束。

## 6. 结果视图

不自造组权重或比赛总分。对每个 Shape 展示：

- Solution Median/P90；
- Baseline 可执行时的 Baseline Median/P90 与 Speedup；
- Shape 14 的 Target-only Latency、Host End-to-End、峰值显存、Useful FLOPs、Achieved FLOP/s、项目估算 MFU、Timing Microbatch 与 provisional Reference 范围；
- Correctness；
- Selected Policy 与 Observed Execution；
- 失败、超时或资源不可执行状态。

项目摘要使用 Shape 1–13 的 Paired 几何平均 Speedup 和最差 Shape。Shape 14 单独展示 Target-only 指标，不混入普通 Sweep 或 Speedup 汇总。项目估算 MFU 使用当前设备实测的、与实际 Attention/Linear Dtype 匹配的饱和 GEMM Roof；它明确标为项目解释指标，不冒充未知权重下的官方总分。逻辑算子流量也明确不是 Nsight 或实际 DRAM Counter。

## 7. 主要落点

| 文件 | 职责 |
| --- | --- |
| `official/test_shapes.json` | 官方 Shape |
| `policy_registry.py` | 可部署 Policy 与内部 `safe` 路径的唯一事实源 |
| `solution/execution_plan.py` | 统一 Eligibility 与执行决策 |
| `solution/kernels/` | Attention、QKV、FFN、Norm/Residual 的局部实现 |
| `solution/cuda_graph.py` | 固定 Shape Graph Replay |
| `runner/candidates.py` | 候选、能力、适用范围与观测证据 |
| `runner/hardware_router.py` | Shape 与硬件白盒排序 |
| `runner/workload_execution.py` | Shape 驱动的 Resident/Batch-streamed 计划 |
| `runner/streamed_execution.py` | Shape 14 的 Microbatch、Comparator、后端筛选与计时 |
| `runner/performance_metrics.py` | Useful FLOPs、Achieved FLOP/s、项目估算 MFU 与逻辑算子流量 |
| `runner/tuning.py` | 串行候选测量与汇总 |
