# 6. Agent 调优闭环与扩展

> 本步骤让 Agent 能够长时间维持一个 GPU 性能目标，用真实运行反馈规划、修改、测量与收敛。它复用性能主线，不建设第二套 Benchmark 或路由系统。

## 1. Agent 消费的工程事实

- `official/torch_transformer_benchmark.py` 与 `official/test_shapes.json`；
- `solution/` 的当前 Transformer、Policy、Execution Plan 和 Kernel；
- `runner/` 的 Hardware Profile、白盒分析、候选、Benchmark、Profile、Tune 和 Calibrate；
- `verified_hardware/<device_id>/` 的当前 Incumbent 与精确路由；
- 最近有关 Run/Profile 的精确引用；
- 用户目标、时间/测量预算、允许修改的范围与停止条件。

Agent 不把旧设备赢家、旧对话摘要或模型推测当作当前 GPU 性能事实。

## 2. 调优循环

1. 用 Probe 和官方 Shape 建立设备—任务基线。
2. 运行当前 Dispatcher，选出性能差、未命中专用路径或资源压力高的 Shape。
3. 用必要 Profile 与白盒成本区分 Launch、Attention、Memory/Layout、GEMM/FFN 和 Runtime 问题。
4. 选择一个目标机制，创建有界 Patch 或固定参数集。
5. 运行静态检查、局部 GPU Smoke 和完整 Comparator。
6. 用 `tune` 测量显式候选，或用 `calibrate` 将新候选放入白盒规划。
7. 阅读 Median、P90、可用的 Speedup、Achieved FLOP/s、Observed Execution、显存和 Profile 变化。
8. 决定保留、调整、删除或转向新瓶颈。
9. 有稳定收益时运行 Formal Calibration，由 Runner 晋升精确路由。
10. 用完整 1–13 Shape Sweep 检查路由回归，必要时单独运行 Shape 14 `benchmark-streamed`，更新 Session 并继续下一轮。

模型负责提出假设、选择工具、编辑代码和解释结果。正确性、计时、Workload Execution Plan、路由晋升和 Bundle 发布由确定性程序决定。

## 3. 领域路由与 Skill

Skill 不是为每个技术名称建立固定工作流。它们对应会影响决策的调优层：

| Skill | 输入 | 典型任务 |
| --- | --- | --- |
| Device Profile | GPU、Runtime、Anchor | 建立身份、能力、显存与快速执行事实 |
| Shape Analysis | 官方 Shape、Dtype、Profile | 计算运算量、中间量、Head Dim、Launch 与 Tensor Core 信号 |
| Runtime/Compiler | 小 Shape、Kernel Gap、Graph/Compile 能力 | 试验 CUDA Graph、局部 Compile、Launch 整理与同步开销 |
| Attention | Seq Len、Head Dim、SDPA Backend、显存 | 分析 Causal SDPA、现有 Mixed-FP16 Efficient/cuDNN Attention、布局、Softmax 和有明确空间时的自定义 Online 候选 |
| GEMM/FFN | QKV/FFN Dim、Dtype、GEMM Anchor、Profile | 筛选库算法、Layout、Epilogue、Buffer Reuse 与 Tensor Core 路径 |
| Batch/Memory | Batch、中间量、可用显存 | 判断执行形态，分析 Workspace、布局与现有 Batch Streaming 路径 |
| Route/Regression | 候选结果、Incumbent、完整 Sweep | 决定 Formal 调用、路由范围和回归处理 |

Skill 定义触发信号、所需上下文、允许工具、候选生成、停止信号和输出摘要。Skill 不写入固定赢家，不绕过 Runner 测量。

## 4. 上下文特化

每轮只组装与当前决策有关的信息：

- 官方语义与目标 Shape；
- 用户目标、预算和授权；
- 当前 GPU Profile 与白盒计划；
- 当前 Incumbent、Candidate 和 Execution Evidence；
- 相关代码、活动 Diff 与最近结果；
- 必要 Profile 摘要和已否定方向；
- 最近用户消息和待办。

不把全部仓库、所有旧 Runs、全部 Skills 或完整 Profiler Trace 放进每轮 Prompt。需要时通过工具重新读取权威文件。

## 5. 参数探索

几毫秒级 Workload 不由 Agent 逐个手工发起。固定程序根据候选类型生成少量合法配置，在 GPU 设备锁下串行执行，把精简摘要写入 Tuning Result。Agent 读取排名、稳定性、正确性和资源影响，再决定扩大、缩小或改变搜索区间。

搜索空间保持紧凑：

- Graph：捕获边界与静态 Buffer 方案；
- Memory：少量 Workspace、布局或新独立候选参数；
- CUDA/Triton：Tile、Warp、Stage 与布局；
- Compiler：少量局部 Compile Mode/Boundary；
- GEMM/FFN：库算法、Epilogue 与 Buffer 复用组合。

## 6. Patch 与反馈

每个优化 Patch 包含：

- 目标 Shape 或 Eligibility；
- 预期改变的瓶颈；
- 受影响文件和可恢复 Diff；
- 局部测试、GPU Comparator 与完整 Forward 计时计划；
- 保留、删除或继续调整的判定信号。

候选失败时，Agent 记录最小有用结论：编译不兼容、Comparator 风险、未实际命中、收益小于噪声、峰值显存恶化或瓶颈假设错误。不保留无限增长的失败日志库。

## 7. 停止与升级

Agent 可根据任务自主决定继续或切换方向。典型信号包括：

- 实测收益已连续接近噪声；
- 当前路径已接近理论上限；
- 瓶颈已迁移到其他调优层；
- 正确性或兼容性风险高于潜在收益；
- 完整 Sweep 出现新回归；
- 新的理论方案显示当前优化体系外仍有显著上限空间。

当已有明确路线可以提升时，继续正常迭代。当多轮失败但理论上限仍较远时，可以引入相对激进的新 Kernel 或新 Fusion Boundary，但作为独立候选，不破坏已有较好路由。

## 8. 非目标

- 通用 IDE 或 Claude Code/Codex 的完整功能复制；
- RAG 系统、向量数据库或全仓库长期索引；
- 通用自动 Rollback 平台或分布式事务系统；
- 常驻多 Agent Swarm、插件市场或多模型调度；
- 在 Forward 内调用 LLM、搜索候选或运行 Benchmark；
- 为了治理形式而建设大型实验平台、证据库或通用 MLOps。

## 9. 验收信号

- Agent 能从新 GPU 的空白状态开始 Probe、测量官方 Shape 并形成本设备路由；
- Agent 能从进展消息中分析、修改、运行、中断、恢复并继续多轮压缩后的目标；
- 领域 Skill 对应真实决策层，不把每个技术名称变成强制工作流；
- 参数搜索使用固定程序串行执行，Agent 读取精简结果而不手工循环调用；
- 候选只有经 Formal Calibration 门禁才影响部署路由；
- 文档和 Agent 明确区分现有 Batch Streaming、Mixed-FP16 Efficient/cuDNN Attention 与可能的自定义 Online Attention。
