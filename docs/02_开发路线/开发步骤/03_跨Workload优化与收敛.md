# 3. 跨 Workload 优化与收敛

> 本步骤负责 Shape 1–13 的 Paired 校准、几何平均和设备精确路由。Shape 14 复用同一 Workload、Solution 与 Registry，但由独立 `benchmark-streamed` 运行，不进入普通 Sweep 或路由晋升。

## 1. 收敛闭环

1. 对目标 GPU 执行一次 Probe。
2. 为每个 Shape 分析主要压力，排除硬件、Dtype、Head Dim 或资源不兼容的 Candidate。
3. 将 `eager-sdpa`、当前 Incumbent 和白盒排序后的适用 Challenger 组成有界 Smoke 计划。
4. 串行检查 Comparator 和实际执行路径，淘汰不正确或发生 Fallback 的候选。
5. 将所有通过 Smoke 的候选去重后交给 Formal 复测。
6. 由 Formal 端到端计时决定赢家，不用 Smoke 延迟提前裁掉正确候选。
7. 把通过正确性、实际执行和收益门禁的赢家写入当前 GPU Bundle。
8. 运行完整 1–13 Shape Dispatch Sweep，观察 Paired 结果、几何平均和路由回归。

这条链可以跟随当前瓶颈在步骤 2 与步骤 3 之间往返，不需要等待所有 Shape 共同完成才开始有价值的局部优化。

## 2. `calibrate` 与 `tune`

`calibrate` 负责自动候选规划和路由发布：

- `--plan-only`：生成白盒分析与候选顺序，不运行 Transformer Candidate；
- `--preset smoke`：检查白盒规划出的有界候选，不发布；
- `--preset formal`：在同一调用中完成 Smoke、全部通过候选的 Formal 复测与原子发布。

Smoke 只承担正确性和执行证据筛选。Formal 接收所有通过者，并以正式延迟结果选择可晋升赢家。每个 Fresh Worker 在正式模型 Warmup 前执行相同且不计时的短时 CUDA Conditioning，避免小 Shape 因候选先后顺序和 GPU Boost 状态发生策略反转；该步骤不进入 Transformer 延迟，也不改变官方计算。

`tune` 只负责用户显式给定的候选：

- 不内置另一个默认 Top-N；
- 不运行白盒粗排；
- 不发布路由，即使使用 Formal Preset；
- 用于比较一个明确机制的参数或备选实现。

## 3. 精确路由

精确路由分为硬件/运行栈身份与 Transformer Shape 两部分。

### 硬件与运行栈

- GPU 类型与名称；
- Compute Capability；
- 操作系统；
- PyTorch；
- CUDA Runtime；
- Driver；
- Matmul Precision 与 TF32 开关。

### Shape 与协议

- Batch Size；
- QKV Dim；
- Heads；
- Seq Len；
- Layers；
- Causal；
- FFN Dim；
- Dtype。

路由解析在模型配置或构建期完成。Forward 只消费一个不可变 Execution Plan，不 Probe、不读取历史结果、不临时 Benchmark。

精确键未命中或 Bundle 过期时，返回 `eager-sdpa`；专用路径运行条件不满足时会明确记录并进入不可部署的内部 `safe` Fallback。Shape 14 的 Batch Streaming 由独立 `benchmark-streamed` 组织，不通过 Transformer 路由键伪装成 Policy。

## 4. 晋升边界

一个新 Route 需要同时满足：

- 来自当前实现和当前官方/Workload Hash；
- 使用 Formal Protocol 在目标 GPU 上测量；
- Comparator 通过；
- Requested Policy、Selected Policy 与 Observed Execution 一致；
- Median/P90 为有限正数；
- 收益超过为测量噪声预留的保守门槛；
- 不用不充分数据覆盖一个有更强 Formal 依据的 Incumbent。

Smoke、Profiler、CPU 运行、Probe Anchor、理论排序和 `provisional` Correctness 都不直接晋升路由。

## 5. Verified Hardware Bundle

```text
verified_hardware/<device_id>/
├── README.md
├── profile.json
├── routes.json
├── manifest.json
└── run_verified.py

results/
├── final/
│   └── <hardware_id>.json
└── intermediate/
```

- `profile.json` 保存这张 GPU 的紧凑画像。
- `routes.json` 是当前设备的唯一部署路由表。
- `manifest.json` 只绑定 Workload、官方快照、Solution、Route 的 Hash，以及 Formal Protocol、Variant 和已正式验证/仍为 provisional 的 Case 分区。
- `run_verified.py` 是调用共享 Runner 的薄入口，不复制 Transformer 或校准逻辑。
- `results/final/<hardware_id>.json` 是唯一跟踪的最终性能文件；Shapes 1-13 保留 Paired Formal 和几何平均，Shape 14 保留独立 Target-only Provisional 结果。
- `results/intermediate/` 收纳全部本地中间测试并忽略上传。

Bundle 以完整事务发布。任一文件生成或校验失败时，保留原 Bundle，不留下半更新路由。

## 6. 跨硬件迁移

1. 读取官方 Shape 和当前 Solution。
2. 运行新设备 Probe。
3. 用白盒成本模型产生粗候选顺序。
4. 对 Shape 1–13 运行有界 Smoke 和动态 Formal；Shape 14 使用独立 Batch-streamed Target-only Benchmark，不混入普通设备校准。
5. 自动创建该设备的 Bundle、发布精确路由并更新该设备的单一最终性能文件。
6. 以后运行相同栈和 Shape 时直接命中；身份改变时回到 `eager-sdpa` 和新校准。

已验证 GPU 的路由只能作为相似机制的初始顺序参考，不能成为新 GPU 的胜者或 Speedup 结论。

## 7. 收敛判断

一轮优化可以在以下信号下收束或切换方向：

- 相关 Shape 的端到端收益已稳定；
- 完整 1–13 Shape Sweep 没有不可接受回归；Shape 14 的独立流式运行能完整结束且 provisional 边界清晰；
- 继续调参的收益已接近噪声；
- Profile 显示瓶颈已迁移到其他算子；
- 理论上限表明当前局部路线不再值得继续复杂化；
- 新方案的维护成本、数值风险或编译风险超过其实测价值。

## 8. 主要落点

| 文件 | 职责 |
| --- | --- |
| `runner/calibration.py` | 单一校准服务与结构化进度 |
| `runner/tuning.py` | 显式候选的串行测量 |
| `runner/route_promotion.py` | Formal 晋升门禁 |
| `runner/routing_contracts.py` | Hardware Identity 与 Route Key |
| `route_contracts.py` | Route v5、Manifest v4、Bundle 加载与严格校验 |
| `solution/dispatch.py` | Bundle 加载、精确匹配与 `eager-sdpa` Fallback |
| `runner/verified_hardware.py` | Bundle 身份检查、运行与发布编排 |
| `verified_hardware/<device_id>/` | 单设备 Profile、Route、Manifest 与精简证据 |
