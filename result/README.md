# 最终性能与报告证据

本目录只保存面向比赛交付的最终性能结果、报告所引用的独立评估证据及其说明，不保存搜索 Trial、候选淘汰记录或日常调优日志。最终性能入口位于 `scripts/run_final_performance.py`。

正式运行：

```powershell
.\.venv\Scripts\python.exe scripts\run_final_performance.py
```

低成本检查脚本是否可运行：

```powershell
.\.venv\Scripts\python.exe scripts\run_final_performance.py --preset smoke
```

脚本串行发起两个独立任务，不再使用旧的单任务合并路径：

1. `resident`：测量 Shapes 01–13；
2. `shape14`：单独测量 Shape 14。

两个任务分别生成 `result/.working/<run-id>/resident/summary.json` 和 `result/.working/<run-id>/shape14/summary.json`，再合并成最终报告。某一组失败不会删除或遮蔽另一组的结果，最终 JSON 和 Markdown 会分别展示 Resident 与 Shape 14 的状态。

每个 Shape 仍使用新的 Python 进程，两个任务与搜索、Profile 和其他 Benchmark 共享同一 GPU 排他锁。默认 `final` 预设对 Shapes 01–05、07–13 保持正式测量强度；Shape 06 使用 1 次正确性、2 次 Warmup、3 轮每轮 5 次正式计时。Shape 14 的 `smoke` 延迟明确标为低成本 Model-compute Estimate；`formal/final` 只运行一次完整逻辑 Batch，并让每个流式 Microbatch 使用不同输入。

Shapes 01–13 交替测量官方 Baseline 和当前部署方案，使用中位数延迟计算 Speedup。Shape 14 运行内存高效部署路径，不物化完整稠密 `S×S` Baseline，因此不计入 Speedup 几何平均值。

每次完整运行都会保留一份独立结果，完成时间同时写入 JSON/Markdown，并用 UTC 完成时间命名运行目录。目录名按字典序即可区分先后：

```text
result/
  20260830T135342.123456Z/
    final_performance.json   完整、机器可读的结果
    final_performance.md     评委可直接阅读的逐 Shape 表格
  20260830T141015.654321Z/
    final_performance.json
    final_performance.md
```

Shapes 01–13 展示 Baseline Median、部署方案 Median/P90、Speedup、Peak VRAM 和几何平均加速比。Shape 14 单独展示 Latency 类型、Latency、Peak VRAM 和正确性，不计入几何平均。搜索和迭代观测位于本地 `observations/`，不上传到 GitHub。

报告使用的机制族消融由 `scripts/run_component_ablation.py` 对 Shapes 01–13 单独生成，并保存在 `result/ablation/<run-id>/ablation.json`。它只比较当前部署和合法的单机制族回退，不修改部署注册表，也不计入最终几何平均加速比。展示指标为 `移除后加速比 / 完整部署加速比 = 完整部署延迟 / 移除后延迟`：低于 100% 表示移除该机制后性能下降。绘图用的精简表位于 `docs/technical_report/figures/source_data/component_ablation.csv`。
