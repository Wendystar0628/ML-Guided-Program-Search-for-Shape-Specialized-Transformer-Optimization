# 5. Agent 运行时与设备校准

> **文档用途**：定义 Agent Runtime 与设备校准的开发内容和验收信号，不记录仓库实现状态。
> **步骤目标**：形成 DeepSeek 驱动、终端优先、有副作用动作默认串行、可主动中断与恢复，并能在新设备上按需建立本机事实的领域 Agent。
> **文件树基准**：完整且唯一的目标结构见[开发步骤总览：完整目标文件树](./00_开发步骤总览.md#完整目标文件树)。本文只列本步骤主要落点。
> **开发方式**：四个节点可按仓库接口、风险和并行条件调整顺序或合并实现；编号不构成硬性开发前置。

---

## 1. 启动入口、DeepSeek 配置与 Capability Handoff

本节点形成单一用户入口、DeepSeek-only 模型配置、最小 Client Adapter 和性能工程接口引用。Agent 依赖与性能运行依赖分开维护，最终 Transformer 不依赖 Agent、模型服务或 Session State。

### 1.1 开发内容

| 组成 | 开发内容 |
|---|---|
| `start_agent.ps1` | 唯一用户 Launcher；定位项目根、Agent 配置和隔离解释器，并转发交互、恢复、一次性执行、状态与帮助请求 |
| `agent` 模块入口 | 统一参数解析、Preflight、交互与非交互返回；不再维护第二套 CLI |
| `agent_config.toml` | DeepSeek 官方 Endpoint、模型、Thinking、Reasoning、输出、Deadline、有限重试和 Context Budget；不保存 API Key |
| `requirements-agent.txt` | 精确固定 Agent 依赖，不改变 Accepted Solution 的性能运行锁 |
| `deepseek_client.py` | Streaming、Usage、Thinking Tool Cycle、Tool Schema、取消与错误分类 |
| `capability_handoff.json` | 发布锚点、官方快照、Solution、Runner、Schema、Policy 与可选能力的项目根相对引用 |

Launcher 只负责定位和转发，不安装依赖、不改配置、不执行设备探测，也不承载 Host 业务逻辑。默认交互入口创建新 Session；恢复显式指定 Session，不静默选择“最近一次”。`status` 与 `help` 保持离线可用。

### 1.2 DeepSeek-only 配置

| 配置组 | 关注点 |
|---|---|
| DeepSeek | 官方 HTTPS Endpoint、用户选择的模型别名、API Key 环境变量名 |
| Thinking | Thinking 开关及模型支持时的 Reasoning Effort |
| 输出 | 最大输出 Token，小于活动上下文预算 |
| 请求 | 一个模型 Turn 的绝对 Deadline 和有限临时错误重试 |
| 上下文 | 模型窗口、较小的活动预算和压缩阈值 |
| 配置身份 | Schema 版本与非敏感配置摘要，恢复时向用户展示配置变化 |

Client 保持以下语义：

- 只使用 DeepSeek 官方 API，不建设 Provider Registry、模型路由、Fallback Model 或自动离线替代；
- 固定启用流式响应、Usage 回收、取消传播和 Tool Schema 校验；
- Thinking 模式只发送支持的参数，不填充无效采样参数；
- Thinking Tool Cycle 所需的 Reasoning 内容只在未闭合请求链的内存中回传，不展示、不持久化；
- 首版默认一次模型回复派发一个 Tool Call；多个调用整体退回模型重提，不并发执行副作用；
- 临时错误只在首个流式事件到达前有界重试，所有尝试共享同一 Deadline；
- 认证、参数、余额、用户取消、部分输出后的断流和已派发工具的请求不透明重试；
- 模型不可用时保留离线审阅、配置修复、计划整理和已有证据查看，不静默切换模型。

### 1.3 凭据与 Handoff

API Key 只进入 Agent Host 进程。日志、Transcript、状态、错误输出和子进程环境统一清洗；Runner、Candidate、Shell 和用户直达命令不继承 DeepSeek Credential。

Handoff 是轻量引用图，不复制源码、Run、Profile 或 Insight Card。它保存发布锚点、项目根相对路径、必要身份摘要和 Capability 状态，并能在不同绝对路径下重新解析。

Handoff 或 API Key 缺失只限制依赖它们的动作：

- Handoff 无效时，正式 Runner 调用、Candidate 执行和设备校准保持不可用；本地文档审阅、契约修复、路径检查和计划整理仍可进行；
- API Key 缺失时，不发起模型请求；离线状态、人工编辑、证据审阅、Handoff 修复和已批准的确定性检查仍可进行；
- 发布锚点变化时重建 Handoff，新设备使用新 Session；旧 Session 保持可审阅，不尝试迁移旧进程。

---

## 2. Agent Host、Tool Gateway 与受控执行

本节点建立一个本地 Host、持续接收控制输入的 Console Reader、默认串行的 Action Loop，以及统一的 Tool Gateway。首版以清楚的动作边界和可见结果为重点，不引入通用 Agent Framework、Worker、消息队列或远程执行平台。

### 2.1 Host 与行动循环

Host 负责装配配置、Handoff、Session、上下文、模型请求、工具结果、用户控制和可见进度。一次典型循环包括：

1. 读取用户消息、Session State 与相关 Artifact；
2. 构建当前上下文并选择需要的 Skill 或事实；
3. 请求 DeepSeek 给出一个结构化动作；
4. 由 Tool Gateway 解析路径、资源、风险和授权范围；
5. 执行 Tool 或受管 Job，并把标准化结果送回下一轮；
6. 保存必要状态，展示结果、预算和下一动作。

模型 Tool Call 和有副作用 Job 首版默认单个串行，减少 GPU 竞争、状态冲突和取消歧义。彼此独立的只读文件检查、静态元数据读取和不会改变决策身份的查询可以并行执行，再由 Host 合并结果。并行不延伸到正式 GPU Job、Patch、Freeze、发布或共享状态写入。

### 2.2 Tool Gateway

| 工具组 | 作用 | 结果地位 |
|---|---|---|
| 文件与检索 | 列树、搜索、读取、Diff、按任务范围编辑 | 工程输入或变更 |
| Candidate | 创建 Draft、Patch、静态检查、Freeze | 实验源码与不可变 Candidate |
| Direct Process | 以明确参数启动已知程序 | 受管 Job，输出不自动成为性能事实 |
| Shell | PowerShell 主线；按真实工具链接入 Bash | 探索、构建胶水和诊断 |
| Runner | Probe、Correctness、Benchmark、Profile、Query；按需接入 Search | 正式事实的唯一通道 |
| 比较与提案 | 比较兼容 Run、提出 Candidate 采用或继续方向 | 派生判断，不自动发布 |

模型只提出工具名、类型化参数、用途和预期证据。Gateway 依据已解析路径、任务范围、资源消耗和副作用风险决定自动执行、请求批准或拒绝，不使用一套写死的路径权限交集替代具体判断。

### 2.3 写入与保护边界

以下对象保持不可越界：

- API Key 与其他未授权 Secret 不可读取、输出、持久化或传给子进程；
- `official/` 中的官方快照保持字节级只读；官方更新通过明确同步流程引入新快照；
- 已封存 Final Evidence 保持只读，不覆盖、不补写、不伪造“最新”；
- 未经用户授权不执行 Candidate 采用、Release Seal、公开提交或其他发布动作。

`runner/`、`tests/`、`agent/`、`docs/`、配置、依赖文件、环境辅助文件、`solution/` 和 Candidate 工作区不设永久禁止修改。Agent 根据用户任务范围、变更风险、证据需要和批准决定是否编辑；影响裁判、比较口径、依赖锁、正式 Solution 或发布材料的修改清楚展示 Diff 与后续重验范围。

读操作通常可自动进行；局部、可恢复且落在明确任务范围内的写入可批次授权；Raw Shell、网络、依赖变化、扩大写范围、长时间 GPU 工作和发布动作显示影响后再请求批准。用户收紧权限立即生效，模型、摘要和 Session 恢复不能自行放宽权限。

### 2.4 Shell 与 Job

- 能以参数数组表达的已知程序优先 Direct Process，减少 Shell 转义和环境差异；
- Windows 工程以 PowerShell 为规范 Backend；WSL 或 Git Bash 有真实工具链价值时再增加 Bash，并使用独立环境身份；
- 每个 Job 记录 Backend、cwd、非敏感环境摘要、进程身份、开始/结束、退出码和截断输出；
- 超时或主动中断作用于已登记进程树，不按进程名宽泛结束任务；
- 进程树无法确认退出时保持该资源槽占用，并向用户展示人工处置项；
- 不创建脱离监督的常驻后台进程，不自动结束未登记外部进程，不修改 GPU Power、Clock 或驱动；
- 正式 Runner 在受监督子进程中执行冻结 Target 的 Correctness 与计时；Shell 临时数字只作 Diagnostic；
- Build、OOM、Timeout 或 Runtime Error 形成明确失败结果，不在同一正式 Run 中静默切换到 Baseline。

---

## 3. 最小可恢复 Session、主动打断与多轮压缩

本节点只保存恢复和审阅真正需要的状态。性能数字、Candidate 源码和 Profile Artifact 继续留在各自权威目录，Session 不建设第二套事实库或生产级事件平台。

### 3.1 最小 Session 文件

| 文件 | 职责 |
|---|---|
| `LOCK` | 本机单写者保护，避免两个 Host 同时更新同一 Session |
| `state.json` | Session 状态、用户目标与约束、预算、权限、活动请求/Job、Candidate/Run 引用、待处理消息和下一动作 |
| `transcript.jsonl` | 用户可见消息、控制事件、Tool/Job 摘要、审批和 Artifact 引用 |
| `context_summary.json` | 最新一代续接摘要，只保存被压缩的判断脉络 |

State 与 Summary 使用原子替换，Transcript 追加写入。未完成动作只保存足以核对真实副作用的动作引用；恢复时先检查 Candidate、Run、文件和进程的实际结果，已完成则补回引用，不确定则暂停并展示事实，不自动回滚或盲目重做。

### 3.2 用户控制

| 控制 | 行为 |
|---|---|
| 自然语言 Steering | 模型生成时，新消息替换旧请求；Tool 或 Job 运行时，新消息排到安全边界 |
| `/pause` | 当前短动作收尾后停止派发新模型请求、Tool 和 Job |
| `/interrupt` 或忙碌时 `Ctrl+C` | 取消活动模型 Stream，或终止已登记 Job 的进程树，并保存可恢复状态 |
| 空闲输入时 `Ctrl+C` | 清除输入行，不丢弃 Session |
| `resume` | 从用户指定 Session 重新读取状态、Artifact 和环境后继续 |
| `/exit` | 在没有未确认进程时写入终态并退出 |

每个模型请求绑定请求身份和用户输入修订号。Steering、暂停或压缩使旧请求过期时，旧结果不进入 Tool Gateway。中断后的部分模型输出不形成动作；受影响的正式 Run 标记为不可比较或中断，不以历史 PASS 补位。

恢复聚焦以下检查：

1. Session 是否存在活跃写者；
2. Handoff、模型配置、环境和相关代码是否发生变化；
3. 活动 Candidate、Run 和 Job 的真实状态；
4. 外部编辑是否让 Draft、授权或比较条件变脏；
5. 待处理用户消息、剩余预算和下一安全动作。

恢复不是 Undo。已发生的文件或外部副作用不自动撤销；外部 Dirty 状态由用户决定保留、修订或重新验证。

### 3.3 Context Builder 与 Compaction

每轮上下文优先装配稳定规则、Policy、Handoff、State、用户目标、相关代码、活动 Candidate、精确 Run/Profile 引用和最近 Transcript Tail。旧设备事实与历史 Insight 只作为带 Scope 的参考，不覆盖本机测量。

压缩保留以下语义：

- 根据活动上下文预算、预留输出和配置阈值自动触发，也允许用户主动触发；
- 在完整 Model/Tool Cycle 边界执行，不切断 Tool Call、Tool Result 或未完成流式输出；
- 输入由上一代 Summary、新增 Transcript Tail 和重新读取的权威状态组成；
- Summary 保存进展、重要决定及原因、被拒绝路线、开放问题和续接提示；
- Goal、约束、预算、权限、活动 Candidate/Job、正式证据和下一动作始终从 State 与 Artifact 重新注入；
- 多轮压缩只保留最新 Summary，上一代作为生成输入而不是长期摘要数据库；
- 压缩失败时保留旧 Summary 与 Tail，暂停普通模型动作并提示收窄目标，不循环压缩或丢失上下文。

本步骤不增加 RAG、Embedding、向量库、知识图谱或跨 Session Agent Memory。精确路径、ID 和正式 Artifact 足以承担项目事实读取。

---

## 4. 新设备校准

本节点让 Agent 在没有兼容历史 Run/Profile 的设备上建立本机事实。旧设备数据可以启发路线，但不作为启动门槛，也不直接进入本机 Measured Best。

### 4.1 校准流程

以下步骤可根据已有兼容事实裁剪；证据充分时跳过重复测量，异常或缺口出现时增加最小必要动作。

1. 解析 Handoff、Runner 能力、项目根和用户目标；
2. 执行不初始化 GPU 的静态 Probe；
3. 明确目标 Workload、Measurement Protocol、预算与允许动作范围；
4. 由固定 Device Calibration Skill 调用类型化 Probe/Runner；
5. 形成稳定 `environment_id` 与独立的动态健康快照；
6. 将已有事实分类为 Compatible、Reference Only 或 Stale；
7. 根据缺口在本机测量 Baseline，并记录当前 Solution 的明确终态；
8. 汇总设备状态、可用证据和建议调优方向；
9. Profile 由证据不足、异常、回退或机制验证价值触发，不作为固定开工仪式。

### 4.2 环境与健康

| 稳定环境身份 | 动态运行健康 |
|---|---|
| OS、Shell Backend | 温度、实时利用率、空闲显存 |
| GPU Identity、Compute Capability、显存容量 | 外部 GPU 进程与短期竞争 |
| Driver、CUDA Runtime | 当前热状态、瞬时频率和系统负载 |
| Python、PyTorch | 本次 OOM、Timeout 与噪声观察 |
| 实际参与执行的 Compiler、Triton 与依赖版本 | 一次性健康告警 |

Benchmark、Runner、Solution/Candidate、Workload 和 Measurement Protocol 各自保留独立身份，不全部塞进环境指纹。Windows 与 WSL 使用不同环境身份，不混合比较。

### 4.3 设备结果与边界

| 结果 | 含义 | 后续判断 |
|---|---|---|
| `ready_for_tuning` | Baseline 与 Solution 已形成目标 Workload 的本机可比较事实 | 选择一个有证据的调优方向，或在收益已满足目标时停止 |
| `baseline_only` | Baseline 可运行，Solution 出现不支持、构建、显存、正确性或运行兼容问题 | 优先处理兼容性、Baseline Characterization 或 Backend 选择 |
| `unsupported_device` | 目标设备或 Workload 无法建立可信 Baseline/正式计时 | 报告能力缺口与可行人工路径，不生成性能结论 |
| `blocked_contract` | Handoff 或 Runner 接口无法支持相关校准动作 | 保留离线修复、审阅和规划；不把契约问题包装成设备性能问题 |

静态只读 Probe 可自动或并行执行。会初始化 CUDA、分配显存、写编译 Cache、运行 Profiler 或产生正式结果的动作纳入 GPU 预算与受管 Job。校准不自动安装或升级依赖、驱动，不静默切换 Windows/WSL，不修改 GPU Power/Clock，也不结束未登记外部进程。

---

## 5. 精炼验收与主要落点

### 5.1 验收信号

| 场景 | 预期结果 |
|---|---|
| 单一入口 | 新 Session、指定恢复、一次性执行、离线状态和帮助共享同一 Launcher 与内部 CLI |
| 缺少 API Key | 模型请求不可用；离线修复、计划、审阅和确定性检查仍可继续 |
| Handoff 无效 | 依赖 Handoff 的 Runner/Candidate/校准动作不可用；契约修复和本地审阅仍可继续 |
| DeepSeek Client | Streaming、Usage、Thinking Cycle、有限重试、单 Tool Call 默认与主动取消可复核 |
| Host 执行 | 有副作用动作与 GPU Job 默认串行；独立只读检查可安全并行并汇总 |
| 权限边界 | Secret、官方快照、封存证据和未经授权发布受到保护；其他路径按任务范围与风险授权 |
| 主动打断 | 模型 Stream 与登记 Job 可中断，部分输出不派发动作，Session 保存后可恢复 |
| 副作用恢复 | 已完成 Candidate/Run 不重复执行；未知或 Dirty 状态暂停并展示事实 |
| 多轮压缩 | 连续两次压缩后，目标、约束、预算、权限、Candidate/Job、Run 引用和下一动作保持一致 |
| 零历史设备 | 能建立本机环境、Baseline 与 Solution 终态，不借用旧设备 Best |
| 正式事实 | Shell 临时数字保持 Diagnostic；Correctness、Latency、Median、Speedup 和 Profile 由 Runner 产生 |

测试使用固定 DeepSeek Client Fixture、受控进程、缩小的 Context Budget 和零历史设备 Fixture。真实 DeepSeek 调用保持为有界能力 Smoke，不通过大量真实 Token 测试压缩。

### 5.2 主要文件落点

完整结构仍以[开发步骤总览：完整目标文件树](./00_开发步骤总览.md#完整目标文件树)为准。

| 路径 | 职责 |
|---|---|
| `start_agent.ps1`、`agent_config.toml`、`requirements-agent.txt` | 单一启动入口、DeepSeek-only 配置和独立依赖 |
| `agent/__main__.py`、`agent/host.py` | 内部 CLI、Console Reader 与默认串行 Action Loop |
| `agent/deepseek_client.py`、`agent/context.py` | DeepSeek 协议、上下文装配与多轮 Compaction |
| `agent/state.py`、`agent/contracts.py` | 最小 Session、恢复逻辑、Handoff/Action/Tool Schema |
| `agent/project_policy.json`、`agent/tools/registry.py` | 任务范围、风险判断与 Tool/Runner/Shell 绑定 |
| `agent/skills/device_calibration/SKILL.md` | 新设备校准、事实分类、设备结果与降级语义 |
| `agent_state/capability_handoff.json` | 性能工程与 Agent Runtime 的轻量引用图 |
| `agent_state/sessions/<session_id>/` | Lock、State、Transcript 和最新 Context Summary |
| `tests/test_agent_runtime.py` | 启动、Client、Tool、打断、恢复、压缩和校准契约 |

Agent 虚拟环境、Session 临时缓存、构建 Cache 和 Python Cache 属于忽略路径。正式 Runner 继续使用 Handoff 指定的性能环境，Agent 解释器不冒充性能运行环境。
