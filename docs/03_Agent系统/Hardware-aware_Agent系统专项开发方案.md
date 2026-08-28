# Hardware-aware Agent 系统专项开发方案

> Agent 面向官方 Transformer Shape 的长时间分析与调优。该阶段推迟到性能主线稳定后实施；未来复用当前 Runner，不扩展为通用 Coding Agent。

## 1. 系统边界

Agent 负责：

- 理解用户目标、官方 Shape 和当前仓库；
- 决定 Probe、Benchmark、Profile、Tune 与 Calibrate 的时机；
- 从 Shape–Hardware 成本与实测中生成假设；
- 选择 Skill、修改代码、设计有界候选和分析反馈；
- 追踪长期目标，在多轮上下文压缩后继续行动；
- 向用户展示当前进度、决策、结果和需要授权的操作。

确定性程序负责：

- 官方 Shape 和 Measurement Protocol；
- Comparator、GPU Event 计时、Workload Execution Plan 与 Worker 隔离；
- Hardware Profile 结构、候选能力门控和白盒粗排；
- Formal 晋升、精确路由和 Verified Bundle 发布；
- 中断信号、进程退出码和结果持久化。

模型输出是提议，Tool 结果、代码、实际 Execution Path 和 GPU 测量是工程事实。

## 2. 运行形态

```text
用户 CLI
   ↓
Agent Host ── DeepSeek Client
   ↓
Context Builder / Session / Compaction
   ↓
Tool Gateway
   ├── File / Search / Patch / Diff
   ├── PowerShell / Bash / Job / Cancel
   ├── Probe / Benchmark / Profile / Tune / Calibrate
   └── Git Status / Diff / Log
   ↓
official/ + solution/ + runner/ + verified_hardware/
```

CLI、Agent 和未来其他调用方复用同一 Benchmark/Calibration Service，不解析终端文本作为主接口。

## 3. 启动、配置与交互

- 一条命令进入持续交互。
- 首版仅支持 DeepSeek API，模型设置位于单独可编辑配置文档。
- API Key 从环境变量读取，不落盘。
- 用户可输入消息、查看当前步骤和 Tool 进度、主动中断、保存退出和恢复 Session。
- 网络请求、Shell Job 和 GPU Service 共享 Cancellation Token。
- 只读和任务内可恢复操作可自主执行；外部影响、广泛删除、Git 发布和修改官方/测量合同按用户授权处理。

## 4. 上下文与记忆特化

稳定上下文保留：

- 官方 Transformer 语义与 `official_01`–`official_14`；
- 用户工程偏好、目标、预算、授权和停止信号；
- 当前 GPU、Workload、Solution、Incumbent 和 Active Patch 身份；
- 关键决策、已排除假设、结果/Profile 引用和待办。

可重读代码、大型 Shell 输出、完整 Profiler Trace 和无关历史不长期注入 Prompt。压缩使用“稳定核心 + 可替换摘要 + 最近尾部”，多轮压缩对旧摘要再摘要，保留身份和引用。

Session 保存状态、计划、决策、活动 Diff、引用与 Transcript Tail，不保存 Secret、全仓库副本或不可序列化运行对象。

## 5. 新 GPU 冷启动

1. 从仓库重新发现官方 Shape、Solution 和 Runner 能力。
2. 运行 Hardware Probe，获得身份、运行栈、容量、带宽相关信号和短性能 Anchor。
3. 为每个官方 Shape 计算运算量、中间量、Launch、Head Dim、Tensor Core 和峰值显存信号。
4. 对当前 Policy Registry 中适用的通用 SDPA、Graph、Fused Norm、Mixed-FP16 Efficient/cuDNN 等候选做能力门控与白盒粗排；`safe` 是不可部署的内部诊断与 Fallback 路径，不参与路由晋升。
5. 对有界候选集运行 Smoke，串行检查 Comparator 与实际执行路径。
6. 将所有通过 Smoke 的候选交给 Formal 计时并选择赢家。
7. 确定性晋升门禁自动创建或更新当前 GPU Bundle。
8. Shape 1–13 形成普通 Paired Sweep 与路由；Shape 14 由独立 `benchmark-streamed` 执行并保留 provisional 边界。Agent 分别阅读两类结果与必要 Profile。

Probe 必须发生在粗排之前；粗排不能代替候选实测。已验证 GPU 数据可以影响初始候选顺序，不能直接成为新设备路由。

## 6. 领域 Skills

| Skill | 主要决策 |
| --- | --- |
| Device Profile | 建立设备能力和运行身份 |
| Shape Analysis | 将官方 Shape 转化为 GPU 成本信号 |
| Runtime/Compiler | CUDA Graph、局部 Compile、Launch 与同步 |
| Attention | Causal SDPA、Head Dim、Layout、Softmax、现有内存高效后端与必要时的自定义 Online 候选 |
| GEMM/FFN | 库算法、Tensor Core、Epilogue、Buffer Reuse |
| Batch/Memory | Resident/Batch-streamed 选择、峰值显存与吞吐 |
| Route/Regression | Formal、Incumbent、路由范围与完整 Sweep |

Skill 只固定安全、测量口径和高重复价值流程；不强制 Agent 按技术名称逐层打卡。Agent 可以跨层选择最相关的决策，路由发布仍由 Runner 控制。

## 7. 参数程序与候选反馈

对 Tile、Warp、Stage、Compile Mode、库算法或其他已注册候选的小空间，固定程序生成合法配置，在 GPU 设备锁下串行运行并保存精简摘要。Agent 不为每个几毫秒实验单独发起一次模型循环。

每轮反馈保留：

- 目标 Shape 与瓶颈假设；
- Patch 或参数集；
- Correctness、Observed Execution、Median/P90/Speedup、显存影响；
- 保留、删除、调整或转向的决策；
- 精确的 Run/Profile/Diff 引用。

无需维护通用实验数据库或无限增长的失败记忆。

## 8. 非目标

- RAG、向量数据库或通用长期文档检索；
- 通用回撤平台、分支管理系统或复杂事务引擎；
- 多模型路由、插件市场、常驻 Swarm 或通用 IDE；
- 在 Transformer Forward 内运行 LLM、Probe 或 Benchmark；
- 为 Production-ready Deployment 建设运维平台；
- 将现有 Batch Streaming、Mixed-FP16 Efficient/cuDNN Attention 与可能的自定义 Online Attention 混为一谈。

## 9. 文件边界

```text
agent/
├── __main__.py
├── config/
│   └── deepseek.example.yaml
├── runtime/
│   ├── host.py
│   ├── model_client.py
│   ├── context.py
│   ├── compaction.py
│   ├── session.py
│   └── cancellation.py
├── tools/
│   ├── gateway.py
│   ├── files.py
│   ├── shell.py
│   ├── gpu.py
│   └── git.py
├── skills/
└── sessions/
```

Agent 使用 `runner/calibration.py` 和其他共享 Service，不在 `agent/` 复制 Benchmark、Comparator、Hardware Router 或 Route Promotion。

## 10. 验收

- 单一 CLI 可持续交互，支持进度、中断、保存、恢复和多轮压缩；
- DeepSeek 配置独立、Secret 不落盘、模型失败有界重试；
- File、Shell、GPU、Git 和 Session 工具具有明确 Schema、Timeout 和 Cancellation；
- 新 GPU 无历史数据时，Agent 可以从 Probe 开始规划 Shape 1–13 的 Paired 路由，并按需单独运行 Shape 14 流式 Benchmark；
- Agent 能完成假设、Patch、Comparator、实测、Profile、Formal 与收敛的多轮链条；
- 参数小空间由固定程序串行执行，路由晋升由确定性 Runner 决定；
- 首版没有 RAG、通用 Rollback、多模型路由、插件市场或常驻 Multi-Agent 系统。
