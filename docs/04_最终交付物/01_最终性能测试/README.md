# 最终性能测试

本目录只保存面向比赛交付的最终性能测试入口与结果，不保存搜索 Trial、候选淘汰记录或日常调优日志。

正式运行：

```powershell
.\.venv\Scripts\python.exe "docs\04_最终交付物\01_最终性能测试\run_final_performance.py"
```

低成本检查脚本是否可运行：

```powershell
.\.venv\Scripts\python.exe "docs\04_最终交付物\01_最终性能测试\run_final_performance.py" --preset smoke
```

脚本按官方顺序串行运行全部 14 个 Shape；每个 Shape 使用新的 Python 进程，并与搜索、Profile 和其他 Benchmark 共享同一 GPU 排他锁。默认 `final` 预设对 Shapes 01–05、07–14 保持正式测量强度；Shape 06 使用 1 次正确性、2 次 Warmup、3 轮每轮 5 次正式计时，避免其超大 Batch 重复执行占用过长时间。普通 `formal` 预设不受影响。

Shapes 01–13 交替测量官方 Baseline 和当前部署方案，使用中位数延迟计算 Speedup。Shape 14 运行内存高效部署路径，不物化完整稠密 `S×S` Baseline，因此不计入 Speedup 几何平均值。

一次完整运行只生成两份最终文件：

```text
result/
  final_performance.json   完整、机器可读的结果
  final_performance.md     评委可直接阅读的逐 Shape 表格
```

最终展示保留设备环境、正确性、逐 Shape Baseline Median、部署方案 Median/P90、Speedup、Peak VRAM，以及 Shapes 01–13 的几何平均加速比。搜索和迭代观测仍位于仓库根目录的 `observations/`，不会进入本目录。
