# 2026-08-31 优化闭环

Resident 分支历史过度碎片化 | 四轮形成 1240 个 Study；956 个可判定分支仅 51 个达到按 cardinality 修正的 TPE 启动阈值，单 Study COMPLETE 中位数为 1 | 多数非平凡分支未进入 TPE 学习阶段 | 已改为“全结构单见证点 → survivor 补足首个 TPE 引导点 → 约 10% 最少采样分支纠偏 → survivor 继续 TPE”

Shape 06 搜索持续呈现高成本、低 Trial 吞吐 | 四轮共 713.9 秒、仅 10 个新 Trial；一次未进入 Formal、三次拒绝、无部署 | 广搜预算回报低 | 下轮只做热点结构内低维局部对照，再决定是否转向跨算子数据流方案

Shape 13 广搜连续无晋升 | 当前 evidence 四轮共 515.6 秒、40 个新 Trial且全部拒绝；旧 evidence 也连续四轮拒绝 | 已跨 evidence 重复出现，继续广搜边际收益低 | 下轮缩减广搜预算，用一次固定局部结构对照判断是否需要方法级候选

外部 GPU 负载会严重扭曲低延迟 Shape | 视频播放期间两次 Resident Final 约 18.1–18.3×，停止播放后两次稳定在 11.83–11.91×；差异主要来自低批次 Baseline 被拖慢约 2.5× | 高加速比是假象 | Final 测量前保持 GPU 独占；部署优劣只用同进程 AB/BA 配对决定

Shape 14 的 Reference 不适合作为性能候选 | 旧流程首轮把 730.9 秒耗在两个 Reference Screen 点；改造后两次独立 Smoke 均约 47 秒，各推进一个不同 Triton 分支点且无 Reference 调用 | 已改为 34 个高价值点的持久化无放回枚举，Reference 只作回退

Shape 14 microbatch=4 不适合 RTX 4080 | 最小协议正确性通过，但归一化延迟约 976 秒、峰值显存约 14.0 GB；相对当前 B1 约 20 秒差距近 49× | 收益方向明确为负且资源裕量过低 | 不加入搜索域，保持 microbatch 1/2
