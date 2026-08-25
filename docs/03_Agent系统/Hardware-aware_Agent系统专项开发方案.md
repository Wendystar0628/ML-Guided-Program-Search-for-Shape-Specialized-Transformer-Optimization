# Hardware-aware Agent 系统专项开发方案

> **定位**：本文件定义 Agent 的专题架构与工程取舍，不记录实现进度，也不规定性能工程与 Agent 工程之间不可调整的开发门槛。
>
> **步骤归属**：[第 5 步](../02_开发路线/开发步骤/05_Agent运行时与设备校准.md)承载 Runtime 与设备校准开发细节，[第 6 步](../02_开发路线/开发步骤/06_Agent调优闭环与扩展.md)承载领域调优闭环。完整目标文件树只在[开发步骤总览](../02_开发路线/开发步骤/00_开发步骤总览.md#完整目标文件树)维护。

## 1. 目标与范围

Agent 面向 Transformer GPU 优化，不扩展为任意仓库的通用 Coding Agent。它的能力集中在：

1. 在新设备上读取工程接口并建立本机环境、Baseline、Solution 和测量事实；
2. 根据 Workload、Profile、正式 Run、预算和权限选择调优方向；
3. 在隔离 Draft 中准备修改，冻结 Candidate，并调用统一 Runner；
4. 让后续模型推理真实消费结构化结果，形成继续、修订、换路、拒绝、停止或接受建议；
5. 支持用户 Follow-up、Shell、暂停、主动中断、状态保存、恢复和多轮上下文压缩；
6. 在 Agent 不可用时保留人工 Runner、Accepted Solution 和发布重放路径。

性能事实仍由确定性 Runner 产生。模型输出是提议，Tool 的真实结果、Candidate、Run、Profile、代码和 Policy 才是工程事实。最终 Transformer 热路径不依赖 Agent、DeepSeek、Session State 或运行时网络。

数字步骤用于分离文档关注面，Agent 可以根据仓库事实和用户目标并行读取、诊断、准备低风险工作或跳过不适用路线。真正需要固定的是官方数学语义、凭据隔离、保护路径、正式事实通道、进程监督和可恢复状态，而不是僵硬的开发顺序。

## 2. Agent 与确定性组件的分工

| 组件 | 主要职责 | 不承担的职责 |
|---|---|---|
| DeepSeek 模型 | 理解目标、解释证据、提出假设、选择路线、生成单一结构化动作、消费结果并调整计划 | 判定正式 Correctness、直接写正式结果、静默采用 Candidate |
| Agent Host | 管理配置、上下文、模型请求、行动调度、用户控制、状态与恢复 | 复制第二套 Benchmark 或替代 Runner |
| Skill | 描述某类调优的适用信号、证据、允许动作、预算、停止和回退 | 成为独立 Agent、扩大权限、直接执行副作用 |
| Program | 对结构化输入做枚举、过滤、聚合或派生 | 自行决定缺什么证据或绕过 Tool Gateway |
| Tool Gateway | 解析路径、权限、Secret、超时、进程和审批，执行受控副作用 | 把 Shell 输出认证为正式性能结果 |
| Runner | 生成环境、Correctness、Benchmark、Profile、Search、Compare 和 Replay 事实 | 解释机制、替用户决定是否采用 Candidate |
| 用户 | 决定目标、重要范围变化、风险接受、Candidate 采用和发布 | 逐项干预所有可逆低风险动作 |

首版默认采用一个本地 Host、一个 DeepSeek Client、一个 Console Control Reader 和一条可观察的 Action Loop。模型 Turn、GPU Job 和有写入副作用的动作由 Tool Gateway 协调；相互独立的只读检查和静态元数据查询可以安全并行，批量候选由 Runner 作为受管的单 GPU 串行批次执行。

## 3. Capability Handoff 与设备校准

Capability Handoff 是 Agent 定位性能工程能力的轻量引用图，不是性能数据包。它引用：

- Release、官方快照、Accepted Solution 和 Candidate Contract；
- Runner 的 Probe、Correctness、Benchmark、Profile、Query、Compare、Freeze、Replay 等能力；
- Workload、Run、Action、Tool 和 Policy Schema；
- 可用或不可用的 Search、Profiler、Compiler 与 Shell Backend；
- 项目根相对路径、版本和 Hash。

Handoff 不复制源码、Run、Profile、Insight Card 或设备历史。项目移动到不同绝对路径时，Runtime 从新项目根解析相对路径并重算 Hash。引用缺失、越界或不一致时，不派发依赖 Handoff 的 Runner、Candidate 执行、设备校准或其他工程副作用；配置修复、契约分析、代码审阅、规划和不依赖 Handoff 的最小 Fixture 仍可按任务权限执行。

新设备允许零条兼容 Run/Profile 启动。设备校准按目标 Workload 和已有事实裁剪，常用内容包括：

1. 静态环境身份：GPU、计算能力、显存、驱动、OS、Python、PyTorch、CUDA、编译器、Backend 与关键数值开关；
2. 动态健康信息：温度、功耗、时钟、显存占用、竞争进程和测量时间；
3. Runner/官方快照/Workload/Accepted Solution 的身份一致性；
4. Baseline A/A 控制、重复性和噪声；
5. 当前 Solution 的兼容、Correctness 和正式延迟；
6. 按证据决定的 Profile 或额外诊断。

历史事实按 Compatible、Reference Only 或 Stale 使用。旧设备数据可以提出假设，不能替代本机测量。Solution 不兼容时保留 Baseline-only 诊断路线；设备超出声明范围时形成明确 Unsupported 终态，不虚构 Best。

## 4. DeepSeek-only 配置与 Client

普通用户只编辑项目根 `agent_config.toml`。首版只支持 DeepSeek 官方 OpenAI-compatible Chat Completions，不建设 Provider Registry、模型路由、Fallback Model 或第二协议。

| 配置组 | 字段 |
|---|---|
| 配置身份 | Schema 版本、规范化配置 Hash |
| DeepSeek | 官方 HTTPS Endpoint、模型别名、API Key 环境变量名 |
| Thinking | Thinking 开关、Reasoning Effort |
| 输出 | 最大输出 Token |
| 请求 | 整体 Deadline、临时错误重试上限 |
| 上下文 | 模型窗口、活动预算、压缩阈值 |

模型别名和服务上限是用户可修改配置，不作为永久比赛事实。Runtime 启动时校验字段组合、Endpoint、模型能力和预算关系，不静默换模型。

Client 契约保持精简：

- 使用固定版本的 OpenAI-compatible SDK、Streaming、Usage 回收、取消传播和 Tool Schema 校验；
- 首版可以把单个结构化 Tool Call 作为易调试的默认动作粒度；当 Tool Schema、取消传播和结果归属已经稳定时，Runtime 可以批量执行相互独立的只读调用，GPU Job 和写入副作用仍受统一调度；
- Thinking Tool Cycle 的 `reasoning_content` 只在未闭合 Cycle 的内存请求链中回传，不写入 Transcript、摘要或正式事实；
- SDK 自动重试关闭；首个文本、Reasoning 或 Tool Delta 之前的临时连接错误、限流和服务端临时错误可以有界重试；
- 认证、参数、余额、用户取消、绝对 Deadline、已出现部分输出、断流或已派发 Tool 的请求不透明重试；
- 全部 Attempt、Backoff 和 Streaming 共享一个单调时钟绝对 Deadline；
- 保存模型名、非敏感配置 Hash、调用 ID、Usage、延迟和错误分类，不保存 API Key 或隐藏思维链。

API Key 只存在于 Agent Host 环境。配置、状态、Transcript、日志、错误、Runner、Candidate 和 Shell 子进程均不获得它。模型不可用时 Session 进入可恢复暂停状态，系统不部署离线模型或静默切换 Provider。

## 5. 单一 CLI 与用户控制

根目录 `start_agent.ps1` 是普通用户入口，内部入口为 `python -m agent`。Launcher 只定位项目根、`.venv-agent`、配置和内部模块并转发参数；依赖安装、配置修改、设备探测和业务逻辑属于独立实现，不进入 Launcher。

| 入口或控制 | 语义 |
|---|---|
| 默认交互 | 创建明确的新 Session，不静默选择最近 Session |
| `resume` | 恢复用户指定的 Session；条件不足时仍可离线查看并保持暂停 |
| `exec` | 执行一次目标并返回结构化终态；需要人工选择时明确返回 |
| `status`、`help` | 离线读取配置、State、Job、预算、权限和修复信息，不调用模型或 GPU |
| `/pause` | 在安全单元结束后停止派发新动作 |
| `/resume` | 重新校验 State、Handoff、环境和未决副作用后继续 |
| `/interrupt` | 取消活动模型 Stream 或终止登记 Job 的进程树，保存后保持暂停 |
| `/context`、`/compact` | 查看上下文预算或在完整动作边界触发压缩 |
| `!` | 用户经同一 Tool Gateway 请求 Shell 动作，不转化为 Agent 扩权或正式事实 |
| `/exit` | 在没有未确认进程树时结束 Host |

自然语言 Steering 在模型 Streaming 期间先持久化新消息，再取消旧请求；Tool/Job 正在产生副作用时，新消息在安全边界生效。Host 在 Tool Dispatch 前比较 Request、Input Revision 和 Context Generation，丢弃过期模型响应。

忙碌时 `Ctrl+C` 与 `/interrupt` 语义一致；空闲输入行中的 `Ctrl+C` 只清理当前输入。Signal Handler 只设置 Cancellation Token，短原子 State/Summary 提交完成后再传播取消。

CLI 展示目标、计划、动作、证据、预算、风险、Job、Candidate、Run 和下一步，不展示隐藏推理。公开帮助、错误和关键状态使用英文并保持 UTF-8 可读。

## 6. 工具、Shell 与权限

首版工具集中在以下能力：

| 工具组 | 用途 |
|---|---|
| Read/Search | 按明确路径或 ID 读取规则、代码、Run、Profile、Candidate、Diff 和 Handoff |
| File/Patch | 按任务范围、工作区策略和风险级别创建或修改工程文件，必要时使用隔离 Candidate |
| Process | 启动、轮询、超时、中断和终止登记的完整进程树 |
| Environment | 获取稳定环境身份和动态健康信息 |
| Runner | 调用 Correctness、Benchmark、Profile、Search、Compare、Freeze 与 Replay |
| Shell | 执行受控 PowerShell；Bash/WSL 在真实工具链需要时作为独立 Backend |
| Decision | 展示权限、请求批准、暂停和记录用户决定 |

有效权限取用户模式、Objective Envelope、WorkspacePolicy、Skill 声明、Program 元数据和 Tool Registry 的交集。Skill 与模型不能扩权。

长期保护面只覆盖 Secret、只读官方快照、已冻结 Candidate、已封存正式证据、版本控制元数据以及未经授权的发布或提交动作。`solution/` 的正式采用与 Release 由用户决定；`runner/`、`tests/`、`agent/`、文档、配置和环境脚本可以在任务范围、WorkspacePolicy 与风险控制内修改。正式 Run、State、Handoff、Acceptance 和 Seal 仍由相应确定性组件维护，避免模型直接伪造事实。

Shell 是实现与诊断能力，不是第二个 Runner。生成代码的构建和运行使用受管进程、有限超时、输出上限和明确环境；Windows 优先使用 Job Object，Fallback 按登记的 PID 与 Start Time 追踪子孙，禁止按进程名宽泛终止。无法确认进程树退出时，Session 标记 termination unknown 并停止派发新的 GPU Job。

## 7. Session State、打断与恢复

每个 Session 只保存四个文件：

| 文件 | 职责 |
|---|---|
| `LOCK` | 本机单写者保护 |
| `state.json` | Goal、预算、权限、活动 Candidate/Job、待处理消息、Pending Action 和下一安全动作 |
| `transcript.jsonl` | 用户可见消息、控制事件和紧凑 Tool Event |
| `context_summary.json` | 最新一代续作摘要，不是事实库 |

用户消息先以唯一 Message ID 追加并 Flush 到 Transcript，再原子更新 State 的待处理消息与提交边界。恢复扫描边界后的小段 Tail 并按 ID 去重；State 引用了不存在的消息时进入不一致暂停状态。

轻量 Pending Action 只用于 Candidate Freeze、正式 Run 和 Search Finalist 等不适合盲目重复的副作用：先登记 Action，再执行和核验 Candidate/Run/Hash，最后提交引用并清空。恢复时按 Action、Run 或 Candidate ID 对账；已完成则回链，不确定则显示事实并暂停。受管 Shell 或非原子 Tool 中断后留下部分效果时，Draft 标记 Dirty，系统不自动重试或回撤。

恢复流程核验 Lock、配置 Hash、Handoff、Policy、环境、源码、Candidate、正式 Run、活动 Job、进程树和 Pending Action。环境或协议变化会降低旧性能证据的比较资格，但不删除历史事实。首版只承诺同机本地恢复；新设备创建新 Session 并重新校准。

## 8. 多轮 Context Compaction

压缩服务同一 Session 的连续行动，不承担长期知识库或检索职责。权威事实始终来自 Policy、Handoff、State、代码、Candidate、Run 和 Profile；Summary 只保存进展脉络、重要决定及理由、已拒绝路线、开放问题和续作提示。

Runtime 依据活动上下文预算估算 System、Policy、Handoff、Tool Schema、Summary、Transcript Tail、当前输入和预留输出。达到阈值时可自动压缩，用户也可主动请求。

压缩只发生在完整 Model/Tool Action Cycle 结束的位置，避免切断 Assistant Tool Call、Tool Result、Thinking Reasoning Content 和最终 Assistant Response。下一代摘要消费上一代摘要、覆盖边界后的完整 Tail，以及重新读取的权威状态与精确 Artifact 引用。

Summary 写入采用唯一最新文件：先生成和校验新摘要，原子替换 Summary，再更新 State 的 Generation、Boundary 和 Hash。恢复时可验证的单代领先摘要可以补写 State，其他不一致进入暂停。压缩失败有界重试一次；仍失败时保留旧摘要和 Tail，进入 Context Exhausted 暂停，不循环压缩或继续普通模型动作。

验收使用固定 Client Fixture 和缩小的活动预算验证多次连续压缩，不向真实 API 填充数十万 Token。真实 API 只做有界 Chat 与单 Tool Call Smoke。

本项目不实现 RAG、Embedding、向量数据库、Retriever、Reranker、知识图谱或跨 Session 模型记忆库。按精确路径、ID 和文本搜索读取工程事实不属于 RAG。

## 9. 领域上下文、Skills 与路由

领域上下文按需组合：用户目标与预算、当前设备身份、Workload 形态、官方边界、Accepted Solution、活动 Draft/Candidate、实际 Dispatch/Fallback、兼容正式 Run、必要 Profile、相关失败和少量匹配 Scope 的 Insight Card。整个仓库、全部历史、所有 Skill 和完整 Trace 不进入每轮上下文。

调优路线直接引用[第 2 步的 1–6 调优层](../02_开发路线/开发步骤/02_性能分析与候选优化.md#4-按证据选择一个优化层)，不维护第二套编号。设备与测量事实刷新以及 Baseline 特征分析都属于第 1 层的证据准备；其余路线按第 2–6 层的名称与 Scope 选择对应 Skill。

这些调优层不是成熟度阶梯。Eligibility Filter 先排除设备、Backend、证据、正确性、权限、参数空间或预算不适用的路线；模型在剩余路线中选择一个活动 Skill 和一个回退。没有适用路线时，合法结果是补充最小事实、收窄目标或停止。

Skill 保存适用条件、所需证据、允许 Program/Tool、候选范围、正确性/测量要求、预算、停止与回退。Runtime 每轮只加载活动 Skill；`device_calibration` 是固定校准流程，其他 Tuning Skill 根据真实实现和证据创建，不预建多个空 Skill。

## 10. Candidate、Search 与反馈闭环

Agent 围绕单一证据链行动：

1. 读取权威上下文并选择一个有证据的方向；
2. 从明确父版本建立隔离 Draft，展示 Diff；
3. 静态检查并冻结完整 Candidate 源码闭包；
4. 由 Runner 生成 Correctness、Benchmark、Profile 或 Search Run；
5. 将紧凑结构化结果交给后续模型推理；
6. 形成继续、修订、换路、拒绝、停止或人工接受建议；
7. 用户采用的候选进入共享全量验证与发布流程。

参数 Search 复用 Runner 的固定 Driver。Agent 定义参数、约束、预算和停止信号；Program 生成并去重配置；Runner 在单 GPU 上串行执行并持久化一个 Search Run；Finalist 使用完整正式 Run 决胜。参数很少时直接形成少量 Candidate，不为展示搜索能力建设搜索平台。

新洞察先作为 Transcript 中的 Insight Card Proposal，引用 Observation、机制、预测、Candidate、正式 Run、否定条件、Scope 和 Fallback。Canonical Insight Card 由用户或获批准的文档流程维护；模型总结不能把假设自动改成 Supported。

Agent 找到更优 Candidate 时只提出采用建议。Accepted Solution、Execution Manifest、全量回归、Release 和 Submission 仍由共享性能与发布流程处理；被拒绝或无收益 Candidate 不改变既有正式版本。

## 11. 自主权与效果评价

Agent 自主性围绕可逆性和事实风险调整：

- 读取、搜索、汇总、静态检查和隔离 Draft 修改可以在声明范围内自主进行；
- 构建、受管测试、Profile 和有限 Search 使用预算、超时、权限与用户模式；
- 读取、修改和验证普通工程文件可以按任务授权自主完成；触及 Secret、只读官方快照、冻结或封存对象、扩大任务范围、安装高风险依赖、改变正式 Solution、发布和提交时再由用户决定；
- 用户可以随时 Follow-up、暂停、中断、恢复或结束。

Agent 效果使用现有 Session、Candidate 和 Run 评价，不建设独立 KPI 平台。值得关注的证据包括：

1. 是否在零兼容历史设备上建立可用本机事实；
2. 后续模型是否真实消费正式 Run 并改变决定；
3. 正确性与正式性能变化；
4. 无收益或错误 Candidate 的拒绝质量；
5. 模型调用、Token、GPU 时间、正式 Run 和墙钟成本；
6. 中断、恢复、保护路径和 Agent 关闭后的 Runner Replay；
7. 人工相同预算下的结果作为可选对照，不要求大型统计实验。

比赛展示若选择 Agent 路线，一条真实闭环已经能够证明其价值：一个领域 Skill、一个隔离 Candidate、一次 Runner 事实、一次后续模型反馈、一个错误候选拒绝和一次脱离 Agent 的 Replay。没有性能收益时可以如实展示校准、拒绝或停止能力，不把流程完成包装成加速。

## 12. 条件式扩展与非目标

扩展由真实问题触发：PowerShell 无法覆盖的工具链可以增加 Bash/WSL Backend；有限 Grid 明显不足且收益可测时可以改进搜索策略；执行任意生成代码需要更强隔离时再建设强制执行边界。三者都不要求预先进入基础 Runtime。

明确非目标：

- 面向任意项目的通用 Coding Agent；
- 自动写正式 Solution、自动接受、自动提交或自动发布；
- 多 Provider、模型路由、Fallback Model 或自动部署离线模型；
- RAG、Embedding、向量数据库、知识图谱和外部知识摄取管道；
- 通用 Undo、Workspace 历史快照、Checkpoint 历史、会话时间旅行或自动回滚；
- 数据库、远程 Session Store、Event Sourcing 和通用消息队列；
- 专业 Worker、多活动 Agent Turn、并发 GPU Job 或常驻 Agent Swarm；
- 每个 Skill 独立 Host、模型、Session、Runner 或结果库；
- 通用 AutoML、Bayesian 或进化搜索平台；
- 默认联网搜索、自动安装依赖、修改驱动或 GPU Power/Clock；
- Dashboard、IDE、Web 控制台、插件市场和在线 LLM 热路径。

## 13. 设计依据

- [DeepSeek Chat Completions API](https://api-docs.deepseek.com/api/create-chat-completion/)：Streaming、Usage、Thinking 与 Tool Call 接口；
- [DeepSeek multi-round conversation](https://api-docs.deepseek.com/guides/multi_round_chat/)：无状态 API 与客户端历史管理；
- [DeepSeek Thinking Mode](https://api-docs.deepseek.com/guides/thinking_mode/)：Thinking Tool Cycle 与 `reasoning_content` 回传；
- [DeepSeek Tool Calls](https://api-docs.deepseek.com/guides/tool_calls/)：Tool Schema 与调用往返；
- [OpenAI Compaction guidance](https://developers.openai.com/api/docs/guides/latest-model?model=gpt-5.2)：重复压缩和稳定上下文锚点；
- [OpenAI Codex CLI](https://developers.openai.com/codex/cli) 与 [security](https://developers.openai.com/codex/security)：终端交互、审批、Sandbox 与工具子进程边界；
- [Claude Code context window](https://code.claude.com/docs/en/context-window) 与 [interactive mode](https://code.claude.com/docs/en/interactive-mode)：结构化摘要、`/context`、`/compact` 和 `Ctrl+C` 交互参考。

这些公开资料只提供交互、权限和上下文管理原则，不公开成熟产品的全部内部状态、提示词与恢复算法。本项目实现的是 DeepSeek 驱动、面向 Transformer GPU 优化的本地结构化 Agent，不复制通用 Coding Agent 产品范围。
