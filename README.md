# TikTok TechJam 2026：Transformer GPU Kernel

本项目围绕官方 Transformer Benchmark，目标是在保持数学语义和接口兼容的前提下，获得正确、可重复、可解释的端到端 GPU 加速，并建设一条与性能热路径解耦的 Hardware-aware Agent 调优路线。

> **English summary:** This repository defines a reproducible Transformer GPU optimization workflow, including the official-baseline reference run, a single performance source tree, structured measurement and release evidence, and a decoupled DeepSeek-powered hardware-aware agent.

开发文档只描述目标、设计和可采用的步骤，不记录实现进度。实际状态以仓库中的文件、测试和结果为准；Bootstrap Baseline 或 A/A Control 只验证裁判与测量路径，不作为优化成绩。

## 官方 Baseline 参考运行

下面是作者参考环境的 PowerShell 路径，不是任意 GPU 的通用安装器。已有 `.venv` 时可直接从激活环境开始。

```powershell
Set-Location 'E:\Study\Msc_AAI\TechJam\TikTok_2026'

# Create the local environment when needed.
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Activate the reference Windows development environment.
. .\activate_dev_env.ps1

# Inspect the environment and baseline data flow.
python .\environment_check.py --skip-compile --skip-extension
python .\baseline_smoke_test.py --device cuda --dtype float32 --warmup 5 --repeats 20

# Run the bootstrap official benchmark workflow.
python .\torch_transformer_benchmark.py --device cuda --dtype float32
```

这组命令用于观察环境、输入、Mask、Correctness 和官方计时流程。终端中接近或偏离 `1.0x` 的结果可能只是同实现计时噪声；正式性能事实由后续统一 Runner 生成并写入 `results/`。完整编译能力检查可另行运行 `python .\environment_check.py`。

## 阅读顺序

1. [赛题中文翻译](docs/01_赛题规则/赛题中文翻译.md)：官方飞书赛题内容的中文整理、交付物与评分要求。
2. [允许与不允许的改造范围](docs/01_赛题规则/允许与不允许的改造范围.md)：数学语义、接口和优化边界。
3. [开发步骤总览](docs/02_开发路线/开发步骤/00_开发步骤总览.md)：六个数字步骤、文档路由和唯一完整目标文件树。
4. [赛题全景开发地图](docs/02_开发路线/赛题全景开发地图.md)：评分、工程主线和跨步骤设计原则。
5. [Hardware-aware Agent 系统专项开发方案](docs/03_Agent系统/Hardware-aware_Agent系统专项开发方案.md)：Agent Runtime、设备校准和调优闭环的专题设计。

运行参考入口：

1. [官方 Benchmark](torch_transformer_benchmark.py)
2. [Baseline Smoke Test](baseline_smoke_test.py)
3. [环境检查](environment_check.py)

## 六个开发步骤

数字表示建议的聚焦顺序，不是禁止 Agent 调整路线的阶段锁。实现者可以依据仓库事实、目标硬件、时间和已有能力合并、回访或跳过低收益工作。

1. [测量基础与实现接口](docs/02_开发路线/开发步骤/01_测量基础与实现接口.md)：官方快照、Runner Parity、结构化 Run、Solution 与 Candidate 装载接口、Correctness。
2. [性能分析与候选优化](docs/02_开发路线/开发步骤/02_性能分析与候选优化.md)：Profile、可证伪假设、隔离候选、首项端到端优化和机制证据。
3. [全量验证与优化收敛](docs/02_开发路线/开发步骤/03_全量验证与优化收敛.md)：全部官方 Workload、比较键、有限 Search、Fallback、资源记录和最终方案选择。
4. [发布证据与提交材料](docs/02_开发路线/开发步骤/04_发布证据与提交材料.md)：干净重放、最终证据、公开仓库、README、技术报告、Demo 和 Devpost。
5. [Agent 运行时与设备校准](docs/02_开发路线/开发步骤/05_Agent运行时与设备校准.md)：DeepSeek-only CLI、状态、打断、恢复、上下文压缩、工具系统和新设备校准。
6. [Agent 调优闭环与扩展](docs/02_开发路线/开发步骤/06_Agent调优闭环与扩展.md)：领域上下文、Skills、路由、Candidate/Run 反馈、有限搜索、效果评价和条件式扩展。

性能代码不依赖 Agent。`solution/transformer.py` 是唯一正式性能源码入口，`candidates/` 承载隔离实验，正确性和性能事实由同一个 Runner 产生。Agent 可以分析事实、准备 Candidate、调用工具和继续决策，但不直接宣布测量成立或静默替换正式 Solution。

## Agent 入口设计

Agent 面向用户的目标入口是项目根的简易启动脚本，内部入口是 Python 模块；模型配置独立于启动脚本，首版只支持 DeepSeek API，API Key 只从环境变量读取。启动、配置、离线状态查看、用户打断、可恢复 Session、上下文压缩与设备校准的接口语义统一收录在[第 5 步](docs/02_开发路线/开发步骤/05_Agent运行时与设备校准.md)，领域调优闭环见[第 6 步](docs/02_开发路线/开发步骤/06_Agent调优闭环与扩展.md)。README 只展示仓库中经过验证的可复制命令，避免把目标接口误写成现成能力。

## 参考入口文件

| 路径 | 稳定职责 |
|---|---|
| `torch_transformer_benchmark.py` | Bootstrap 官方 Benchmark；目标结构中的权威快照归 `official/` |
| `baseline_smoke_test.py` | 学习和诊断 Baseline；统一 Runner 可覆盖其长期测量职责 |
| `environment_check.py` | 参考环境检查；结构化环境事实由 Runner Probe 产生 |
| `activate_dev_env.ps1` | 作者开发环境辅助，不进入正式 Solution 身份 |
| `requirements.txt` | 公开运行所需的直接依赖 |
| `requirements-lock.txt` | 可复现运行采用的精确依赖快照 |

完整且唯一的目标文件树见 [开发步骤总览：完整目标文件树](docs/02_开发路线/开发步骤/00_开发步骤总览.md#完整目标文件树)。其他文档只解释关注路径，不维护竞争版本。

## 工程语言与展示约定

- 项目自有 Python、PowerShell、Triton、CUDA 和 C/C++ 源码中的标识符、Docstring、行内与块注释统一使用英文。
- 注释解释约束、原因和非显然的硬件行为，不逐行复述代码。
- 公开 CLI 的帮助、错误和关键状态默认使用英文并保持 UTF-8 可读；内部开发文档可以使用中文。
- 官方 Benchmark Snapshot 保持原始字节，不为统一风格修改上游注释或文本。
- 最终 README 的安装、运行、结果、限制和主要结果表采用英文或中英双语。
- Runner 生成的结构化结果是性能事实来源；截图、手抄表格和临时 Shell 数字只作诊断。

## 资料与可信度

信息发生冲突时，优先采用本届最新官方规则、官方 Benchmark 可执行行为和当前目标硬件实测。往届案例和跨赛事经验只用于提出假设或改善表达，不作为本届结构模板和获奖保证。

可公开核对的入口：

- [赛题中文整理与来源边界](docs/01_赛题规则/赛题中文翻译.md)
- [允许与不允许的改造范围](docs/01_赛题规则/允许与不允许的改造范围.md)
- [TikTok TechJam 2026 Overview](https://tiktoktechjam2026.devpost.com/)
- [TikTok TechJam 2026 Official Rules](https://tiktoktechjam2026.devpost.com/rules)

密码保护的官方材料、本地参考 PDF、第三方背景资料和仓库外部研究笔记不随公开仓库再分发；项目只保留许可允许的公开链接和自行整理的规则、证据边界与开发设计。

## 设计边界

- 官方 Benchmark 与正式 Solution 分离，优化源码不直接改写官方快照。
- 正式发布和提交可以脱离 Agent、Agent State、在线模型和运行时网络独立重放。
- 项目不建设 Dashboard、远程实验数据库、微服务、集群调度或通用 Coding Agent。
- Agent 不建设多模型路由、RAG、向量知识库、通用回撤、历史 Checkpoint、专业 Worker 或并发 GPU Job。
- 能力和结果只有在真实运行中产生可回链证据时才进入 README、报告、Demo 或 Devpost。
