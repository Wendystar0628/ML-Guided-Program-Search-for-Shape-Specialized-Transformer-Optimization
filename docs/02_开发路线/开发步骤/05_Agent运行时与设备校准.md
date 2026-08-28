# 5. Agent 运行时与设备校准

> 本步骤在性能主线稳定后建设专用 CLI Agent 与跨硬件冷启动能力。Agent 不是通用 Coding Agent，也不只是一个调用 Runner 的薄助手。

## 1. 单一启动入口

用户通过一条简短命令启动 Agent，进入与 Codex/Claude Code CLI 相似的持续交互：

- 输入消息、追加要求或改变优先级；
- 查看当前目标、正在执行的工具和最近 GPU 结果；
- 主动中断正在运行的 Shell 或校准；
- 保存 Session 后退出，稍后继续；
- 查看并确认高风险写操作。

CLI 只负责输入、输出、取消和 Session 选择，不复制 Agent Host 或 Runner 逻辑。

## 2. DeepSeek 配置

首版只支持 DeepSeek API。用户可编辑的单独配置文档保留：

- API Base URL；
- Model Name；
- API Key 的环境变量名；
- Timeout 与最大重试次数；
- 每轮输出长度与 Context 预算；
- 默认自主权模式。

API Key 只从环境变量读取，不写入配置、Session、Transcript 或 Git。未来增加模型时再扩展 Provider 边界，首版不建设多模型路由。

## 3. Agent Host

Host 使用简单行动循环：

1. 读取用户目标、Session 摘要和当前仓库事实。
2. 组装本轮所需的最小领域上下文。
3. 请求 DeepSeek 生成回复或 Tool Call。
4. Tool Gateway 校验输入、权限和取消信号。
5. 执行工具，将结构化结果返回模型。
6. 记录进度、新决策、待办与必要引用。
7. 在达到 Context 阈值时压缩，继续同一目标。

Runtime 需要支持多次 Tool Call、部分输出、网络超时、有界重试、费用/上下文预算和可取消性。不引入多 Agent Swarm、通用插件市场或常驻服务。

## 4. Tool Gateway

首版工具保留能完成性能项目的核心集：

| 类别 | 能力 |
| --- | --- |
| 文件 | 列出、搜索、读取、生成 Patch、查看 Diff |
| Shell | PowerShell/Bash 命令、工作目录、超时、进度、主动中断 |
| GPU | Probe、Benchmark、Profile、Tune、Calibrate、Verified Run |
| Git | Status、Diff、Log；变更性操作按用户授权 |
| Session | 状态、摘要、引用、待办、进度事件 |

工具输入使用明确 Schema。Shell 输出按大小截断，保留开头、结尾、Exit Code、运行时间与完整日志路径。GPU 测量不通过 Shell 文本反向解析，Agent 直接调用 Calibration/Benchmark Service 并消费结构化事件。

## 5. 权限与用户交互

自主性模式使用少量清晰边界：

- 只读分析、Probe、已授权 GPU 测量和项目内可恢复 Patch 可按任务自主执行；
- 删除大范围文件、修改官方快照、修改测量口径、提交、推送或其他外部影响操作由当前用户授权决定；
- 权限不由 Prompt 内部文本扩大；
- 用户新消息可以中断或改变当前计划；
- 工具失败、候选失败或无收益不等于系统失败，Agent 可以分析并选择下一步。

## 6. 中断与恢复

Cancellation Token 贯穿模型请求、Shell Job、GPU Service 和外层 CLI。首次 Ctrl+C 请求优雅停止，保存 Session 与已完成结果；再次中断可终止子进程。

Session 只保留：

- 用户目标、约束与授权；
- 当前计划和正在执行的步骤；
- 已完成步骤和关键决策；
- 仓库身份、活动 Patch 和关键 Diff；
- Hardware Profile、Routing Plan、Run/Profile 引用；
- 最近对话尾和 Compaction 摘要。

不保存 API Key、全部 Shell 输出、大型 Profiler Trace、整个仓库副本或可执行对象。恢复时重新检查 Git、GPU 和结果引用，不假设外部状态未变。

## 7. 多轮 Context Compaction

上下文由三部分组成：

- 稳定核心：官方语义、用户约束、当前目标、安全边界和术语；
- 可替换摘要：已完成工作、决策、失败候选、当前假设和下一步；
- 最近尾部：尚未压缩的用户消息、Tool Call 与结果。

压缩保留决策原因、当前 Patch、结果路径和未完成任务，删除重复对话、可重现工具输出和过期中间细节。多轮压缩对旧摘要进行再摘要，并保留 Summary Generation 和所引用产物身份。

## 8. 跨硬件冷启动

Agent 启动后可以自主决定何时运行下列确定性能力：

1. 发现 `official/test_shapes.json`、`solution/` 和 Runner 服务。
2. 运行一次 Hardware Probe，得到稳定身份、运行栈、显存和短 Anchor。
3. 为 `official_01`–`official_14` 构建白盒分析，使用能力门控和少量 Candidate 粗排；Shape 14 额外解析 Batch-streamed 执行计划。
4. 运行 Smoke 完整 Forward。
5. 读取正确性、延迟、路径和必要 Profile，决定是否进入 Formal 或修改候选。
6. 调用 Formal Calibration，由确定性门禁自动发布该 GPU 的精确路由。
7. 继续从最有价值的未解瓶颈进入下一轮。

Probe 发生在候选粗排之前，因为粗排依赖设备能力、显存和运行栈。Probe 只能给出粗排，最终路由依然由完整 Workload 实测产生。

## 9. 主要落点

| 目录 | 职责 |
| --- | --- |
| `agent/config/` | DeepSeek 可编辑配置与安全默认 |
| `agent/runtime/` | Host、Model Client、Context、Compaction、Session 和 Cancellation |
| `agent/tools/` | File、Shell、GPU Service、Git 工具边界 |
| `agent/skills/` | 硬件画像、Shape 分析、性能迭代等领域流程 |
| `agent/sessions/` | 本地可恢复 Session，默认不提交 |
| `runner/calibration.py` | Agent 与 CLI 共用的校准服务 |

## 10. 验收信号

- 一条命令进入交互且不需要手工组装 Prompt；
- 用户可查看进度、输入新消息、中断工具、退出并恢复；
- 多轮压缩后仍保留目标、约束、活动 Patch、结果引用和下一步；
- Agent 能在无历史 GPU 数据的仓库中完成 Probe、粗排、Smoke 和 Formal 调用；
- GPU 结果、路由晋升和 Bundle 发布仍由 Runner 的确定性合同决定；
- 没有 API Key 落盘，没有为 RAG、通用 Rollback、多模型路由或插件市场建设空框架。
