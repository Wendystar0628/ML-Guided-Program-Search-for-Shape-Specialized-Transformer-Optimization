# 6. Agent 调优闭环与扩展

> **文档定位**：说明 Agent 怎样消费性能分析、候选验证和运行时能力，形成一个真实的 Transformer GPU 调优反馈闭环。本文描述稳定设计，不记录实现进度。
> **执行方式**：下列数字只组织内容，不是硬阶段门。Agent 可以从已有 Candidate、Run、Profile 或 Insight 开始，也可以先做最小错误 Fixture；证据信号决定下一动作和可声明范围，不决定用户何时可以接受、拒绝或继续。
> **并行接入**：本步骤可以接入[第 5 步](./05_Agent运行时与设备校准.md)提供的完整能力，也可以先用满足同一输入输出契约的最小 Host、Tool Gateway 和固定 Fixture 验证领域闭环，因此运行时与领域调优可以并行开发。
> **文件树**：完整且唯一的目标文件树见[开发步骤总览：完整目标文件树](./00_开发步骤总览.md#完整目标文件树)。本文只列主要落点。

规则与总设计依据：[赛题中文翻译](../../01_赛题规则/赛题中文翻译.md)、[允许与不允许的改造范围](../../01_赛题规则/允许与不允许的改造范围.md)、[赛题全景开发地图](../赛题全景开发地图.md)和[Hardware-aware Agent 系统专项开发方案](../../03_Agent系统/Hardware-aware_Agent系统专项开发方案.md)。

执行接口依据：[第 2 步：性能分析与候选优化](./02_性能分析与候选优化.md)、[第 3 步：全量验证与优化收敛](./03_全量验证与优化收敛.md)和[第 5 步：Agent 运行时与设备校准](./05_Agent运行时与设备校准.md)。

## 1. 本步骤只解决什么

本步骤只解决一个问题：让模型的一项领域调优动作经过真实 Candidate 和共享 Runner，随后把结构化结果交给后续模型判断，形成可观察的继续、修订、换路、拒绝、停止或人工采用建议。

最小成果包括：

1. 从设备、Workload、代码和正式证据中选择一项有依据的 Tuning Skill。
2. 至少一个 Candidate 由模型依据设备、代码与正式证据，在隔离 Draft 中形成或实质修订并冻结。
3. 共享 Runner 挡住错误 Candidate，并为正确 Candidate 产生结构化结果。
4. 至少一次后续模型判断实际消费该结果，而不是只完成单次工具调用。
5. 用户可以随时接受、拒绝、要求补证据或继续；Agent 不替用户决定正式主线。

本步骤不重新定义测量协议、Candidate 生命周期、有限 Search、Insight Card、Agent Runtime 或设备校准。它们分别由第 2、3、5 步维护，本文只说明怎样把这些现有能力连接成领域反馈。

性能主线不依赖 Agent。模型不可用、闭环未完成、没有收益或用户拒绝时，Accepted Solution、Runner 和已有发布仍然可用；系统不为此切换第二模型或部署离线模型。

## 2. 从其他步骤消费什么

| 来源 | 本步骤消费的能力 | 本步骤不再复述 |
|---|---|---|
| 第 2 步 | 当前主假设、候选优化路线、Profile/Run 引用、Draft 与 Freeze 接口、正确性风险和候选结论 | 测量公式、Profile 方法、Candidate 字段、正式 Run Schema |
| 第 3 步 | 全 Workload 结果、Measured Best、Fallback、有限 Search capability、Finalist 和收敛结论 | SearchSpec、筛选协议、聚合规则、完整回归细节 |
| 第 5 步 | Handoff、设备事实、目标与预算、权限、模型调用、Tool Gateway、状态与中断语义 | CLI、配置、恢复、上下文压缩、Shell 和设备探测实现 |
| 规则文档 | 官方数学语义、允许与禁止的改造范围、交付边界 | 规则解释和官方脚本细节 |
| 已有工程事实 | Accepted Solution、冻结 Candidate、兼容 Run/Profile、Insight Card 和发布引用 | 不复制源码、样本、Trace 或历史日志 |

这些输入不是必须同时存在：

- 已有兼容 Candidate 和 Run 时，可以直接让 Skill 解释结果并提出下一动作。
- 已有 Profile 但没有 Candidate 时，可以从已证实热点形成最小候选。
- 只有 Candidate 时，可以先用 Runner 验证，再把结果交给模型。
- 没有可用候选时，可以从 Accepted Solution 派生一个最小 Draft。
- 领域链尚未接通时，可以先用固定错误 Fixture 证明 Correctness 拒绝路径。
- 缺少本机事实或证据不可比时，调用第 5 步校准能力；不从旧设备结果补位。

运行时与领域调优可按接口成熟度并行推进；接口可用即可接入，完整实现可以随后替换 Fixture，而不改变 Skill、Candidate、Runner 和结果反馈之间的契约。

## 3. 常见证据流，而不是一条固定流水线

| 起点 | 常见流向 | 合法结果 |
|---|---|---|
| 已有 Run/Profile | 读取兼容证据 → 选择 Skill → 解释机制与下一事实 | 继续、补 Profile、形成 Candidate、换路或停止 |
| 已有冻结 Candidate | Runner 验证 → 结构化摘要 → 后续模型判断 | 接受建议、修订、拒绝、比较或停止 |
| 只有 Accepted Solution | 选择 Skill → 最小 Draft → Freeze → Runner | 得到第一个真实 Candidate 反馈 |
| 固定错误 Fixture | Runner Correctness → 拒绝摘要 → 后续模型确认边界 | 证明错误不会进入有效性能比较 |
| 少量敏感配置 | 直接 Candidate 或复用第 3 步有限 Search → Finalist Run | 在冻结空间与预算内选择候选 |
| 证据不足或身份漂移 | 请求一项最有信息量的测量或设备事实刷新 | 补事实、降级 Claim 或停止 |

最小真实闭环的共同部分是：**一个真实 Skill → Candidate → Runner → 后续模型消费结果 → 可见决定**。

这不是要求每次都由 Agent 从零生成 Candidate。复用既有 Candidate、Run 或 Profile 可以验证领域解释和反馈契约，但不能单独支撑“自主调优闭环”声明；该声明需要至少一次由 Agent 形成或实质修订 Draft、冻结 Candidate 并送入真实 Runner 的证据。

模型结果与确定性 Policy 结果应区分。阈值规则自动挡错或排序可以作为工程能力，但不能冒充模型已经读取测量反馈。后续模型判断必须引用它实际收到的 Run 摘要。

## 4. 领域上下文与路线选择

### 4.1 只给模型本轮需要的上下文

| 上下文 | 最小内容 |
|---|---|
| 目标与边界 | 目标 Workload、主指标、正确性要求、允许改造、权限、预算和停止条件 |
| 设备与环境 | environment_id、设备终态、可用 Backend、健康摘要和兼容事实引用 |
| Workload | 本轮相关的 Shape、Dtype、Mask、Layer 和实际执行路径 |
| 代码 | Accepted Solution 的相关调用链、活动 Draft/Candidate 和 Diff |
| 证据 | 少量兼容 Run/Profile、失败摘要和必要 Artifact 引用 |
| 活动 Skill | 适用条件、允许工具、候选范围、正确性风险、测量要求和回退 |

不把整仓库、全部 Run、完整 Profile Trace、所有 Skills 或长 Transcript 塞给模型。正式数字和大 Artifact 留在原权威位置；模型只接收精确引用及会改变决定的紧凑摘要。

### 4.2 引用统一调优层，可直接跳转

本步骤只引用[第 2 步维护的 1–6 调优层](./02_性能分析与候选优化.md#4-按证据选择一个优化层)：

| 层 | 路线 | 主要信号 |
|---:|---|---|
| 1 | 设备与测量校准 | 新设备、缺少可证伪假设、性能异常、环境或比较身份需要刷新 |
| 2 | 编译与运行时 | Launch、Graph、Compile、SDPA 或 Runtime 选项可能改变端到端执行 |
| 3 | 成熟 Kernel 与 Backend 选择 | 至少两条语义等价且设备、Shape 兼容的既有路径 |
| 4 | Kernel 参数 | 参数敏感性已有证据，范围和组合数有限 |
| 5 | 图、融合与布局 | Profile/Graph 指向 Launch、中间写回、同步或 Layout 成本 |
| 6 | 自定义 Kernel 与专门化 | 热点集中、成熟路径收益到顶，潜在收益覆盖风险 |

确定性 Eligibility 先过滤不支持、不正确、证据不足、超权限或超预算的路线；模型在剩余路线中选择一项，也可以请求下一事实或停止。路线之间没有固定低到高顺序，不要求为展示完整性逐项执行。

Correctness 失败先修订或拒绝；证据不可比先刷新事实；已有充分证据时不重复 Profile；配置很少时不为展示 Search 而创建 Search。

## 5. Skill 的最小契约

一个 Skill 是一份领域工作流，不是一个独立 Agent。它至少说明：

- 解决的问题、版本和适用 Scope；
- 激活证据与不适用条件；
- 允许使用的 Program、Tool 和 Runner 操作；
- 可修改的 Candidate 范围与主要正确性风险；
- 需要读取或产生的正式证据；
- 成功、否定、停止和 Fallback 条件；
- 返回给模型的结构化结果。

Skill 规定何时做、为什么做和何时停止；固定 Program 只对结构化输入做派生、过滤、枚举或汇总；Tool 执行权限化副作用；Runner 产生正式事实。任何一层都不能扩大 Workspace Policy、直接写正式结果或自行接受 Candidate。

比赛证明只需一项由真实证据选中的执行 Skill。其他路线可以保留普通名称及 Unavailable 或 Not Implemented 原因，等出现真实消费者后再补；不预建每 Skill 独立模型、Session、Runner、Result Store 或 Router Agent。

## 6. Candidate、Runner 与模型反馈

Candidate 的 Draft、Freeze、身份和修改规则直接复用第 2 步；正式比较、有限 Search、Finalist 和 Measured Best 直接复用第 3 步。本步骤只增加 Agent Action、Skill 和后续 Decision 对这些既有对象的引用，不复制测量数据。

一次真实反馈应做到：

1. 模型提出一个有证据、可证伪且范围有限的动作。
2. Tool Gateway 在允许范围内形成或选择冻结 Candidate。
3. Runner 先执行 Correctness；失败 Candidate 不进入有效性能比较。
4. 正确 Candidate 的 Benchmark、Profile 或 Search Finalist 结果由既有 Runner 产生。
5. Runtime 把 Run ID、关键摘要、比较资格、失败和剩余预算送回模型。
6. 后续模型判断明确选择继续、修订、换路、拒绝、停止或提出采用建议。

结果解释保持简单：

| 结果 | 合理反馈 |
|---|---|
| 构建或 Correctness 失败 | 修订、换路或拒绝；不产生性能 Claim |
| 正确但无稳定收益 | 拒绝、收窄假设或停止 |
| 结果不稳定或不可比 | 检查身份、健康和协议，必要时补测 |
| 仅部分 Workload 有效 | 声明 Scope、反例和 Fallback |
| 正确且稳定改善 | 提出继续验证或人工采用 |
| 预算耗尽或无 Eligible Skill | 保存引用和停止原因，保留原方案 |

固定错误 Fixture 必须被 Correctness 挡住，不能进入 Measured Best，也不能改变 Accepted Solution。Transcript 只需保存 Action、Candidate、Run 和 Decision 的紧凑引用，不建设事件数据库或第二结果库。

有限 Search 的触发、SearchSpec、Screening 与 Finalist 规则见第 3 步。Insight Card 的模板、证据与维护规则见第 2、3 步；Agent 只在新证据改变机制判断时提出 Proposal，用户决定是否合并。

## 7. 人工接受与能力声明

用户可以在任意时点接受、拒绝、要求补证据或继续某个 Candidate、路线或建议。本文不把接受权绑定到章节、开发阶段或固定证据数量。

证据只限制能够声明什么：

- 用户可以接受一个实验方向，但没有正式 Run 时不能声明性能提升。
- 用户可以采用一个 Candidate，但未覆盖的 Workload、环境和 Fallback 必须继续标为未验证。
- 单一 Workload 改善不能冒充完整官方范围收益。
- Search Incumbent 或 Measured Best 不能自动冒充 Accepted Solution。
- 没有性能改善的真实闭环仍可证明校准、挡错、反馈或停止能力，但不能改写 Speedup Claim。

采用建议应简洁展示 Candidate、父版本、Diff、已有 Correctness/性能证据、Scope、风险、Fallback、预算和下一项验证。用户决定采用后，可随时调用既有的[全量验证与优化收敛](./03_全量验证与优化收敛.md)及[发布证据与提交材料](./04_发布证据与提交材料.md)流程；这些流程决定最终可以公开声明和封存的范围，而不是决定用户何时有权作出选择。

比赛材料中的最小 Agent 证据是：

1. 一项真实 Skill 参与路线选择；
2. 至少一个由 Agent 形成或实质修订的 Candidate 经过真实 Runner；
3. 后续模型实际消费结果并作出可见决定；
4. 一个错误 Candidate 被拒绝；
5. 人工接受、拒绝或继续的选择与证据范围清楚。

Fixture 用于证明挡错和控制路径；对外声明自主调优闭环时，至少一条正向闭环使用已配置的真实 DeepSeek Client 与真实 Runner 完成。

不需要为这一纵切建设 Dashboard、统计管道或治理平台。

## 8. 条件式扩展

| 扩展 | 触发信号 | 最小边界 |
|---|---|---|
| 增加 Skill | 新瓶颈反复出现且现有 Skill 无法覆盖 | 复用同一 Host、Tool Gateway、Candidate 和 Runner |
| 多步换路 | 单路线闭环显示继续投入有明确价值 | 每次只激活一个 Skill，保留总预算和停止条件 |
| 有限 Search | 参数敏感、空间有界且第 3 步能力可用 | Screening 不替代正式 Finalist Run |
| 自定义 Kernel | 成熟路径收益到顶且热点集中 | 一个最小 Candidate 和明确 Fallback |
| Bash | PowerShell 无法合理覆盖真实工具链 | 独立环境身份，不混用 Windows 与 WSL 结果 |
| 强制隔离 | 需要自动执行任意生成代码 | 未验证前保持 Ask First 与预注册模板 |

扩展由证据和实际消费者触发，不由“完整 Agent”外观触发。长期能力仍保持一个 Host、一个模型、一个活动行动循环和一个正式 Runner。

## 9. 明确非目标

- 通用 Coding Agent、IDE、Web 控制台、插件市场或团队后台；
- 多 Provider、多模型路由、Fallback Model 或自动部署离线模型；
- Worker、Agent Swarm、多活动 Agent Turn、后台服务或并发 GPU Job；
- RAG、Embedding、向量数据库、语义检索、知识图谱或跨 Session Memory Store；
- 通用 Undo、回撤、Workspace 快照、Checkpoint 历史、时间旅行或自动回滚；
- 数据库、消息队列、远程 Session Store、第二结果库或 Promotion 平台；
- 通用 AutoML、Bayesian/进化搜索或无约束参数空间；
- 每个 Skill 独立 Agent、模型、Session、Runner 或 Current Best；
- 自动写 Accepted Solution、自动接受 Candidate、自动提交或自动发布；
- Unrestricted Shell、默认联网、自动安装依赖、修改驱动或 GPU Power/Clock；
- 用 Shell 临时时间、Profiler、Search Screening 或模型总结替代正式 Runner；
- 让在线模型、Agent Runtime、Session State 或网络进入 Transformer 推理热路径。

## 10. 主要文件落点

完整结构只以[开发步骤总览：完整目标文件树](./00_开发步骤总览.md#完整目标文件树)为准。本步骤主要涉及：

| 路径 | 用途 |
|---|---|
| agent/skills/{skill_id}/SKILL.md | 领域调优工作流 |
| agent/programs/{program_id}.py | 必要的纯派生、过滤、枚举或汇总逻辑 |
| agent/tools/registry.py | 连接权限化 Tool 与共享 Runner |
| candidates/_drafts/{draft_id}/solution/ | 隔离 Draft |
| candidates/{candidate_id}/ | 冻结 Candidate |
| results/runs/{run_id}.json | Runner 产生的结构化结果 |
| agent_state/sessions/{session_id}/ | 最小 Action、Candidate、Run 与 Decision 引用 |
| docs/04_调优证据/Insight_Cards.md | 用户维护的机制解释与证据引用 |
| solution/ | 用户选择采用后由既有确定性流程更新的正式源码 |

本步骤不新增顶层调优目录、Search Store、Memory Store 或另一棵 Results Tree。
