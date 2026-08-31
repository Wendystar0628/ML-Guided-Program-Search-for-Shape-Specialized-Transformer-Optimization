# 2026-08-31 优化闭环

Resident 分支历史过度碎片化 | 四轮形成 1240 个 Study；956 个可判定分支仅 51 个达到按 cardinality 修正的 TPE 启动阈值，单 Study COMPLETE 中位数为 1 | 多数非平凡分支未进入 TPE 学习阶段 | 已改为“全结构单见证点 → survivor 补足首个 TPE 引导点 → 约 10% 最少采样分支纠偏 → survivor 继续 TPE”

Shape 06 的旧覆盖门禁阻断完整闭环 | 连续四轮每轮只能新增 4–5 个 Trial，并要求全 36 分支覆盖；固定为 10 个 mandatory 核心且稳定搜索域后，两次 300 秒运行跨轮补齐覆盖，随后 Formal 以 1.674× 晋升 | 预算问题而非空间无收益 | 保留 36 分支自由搜索；mandatory 只作单因子核心覆盖，warm-start 不再改变结构域

Shape 13 的历史阻塞来自结构域漂移与不完整覆盖 | 稳定域累计到 53 个 Trial 后进入 Formal，专用 Triton Shape13 路径以 1.054× 晋升；下一轮相对新 incumbent 为 0.969×并被拒绝 | 原“空间无收益”判断被新证据否定 | 保留专用路径与持久历史，不再把动态 warm-start 当作 required structure

外部 GPU 负载会严重扭曲低延迟 Shape | 视频播放期间两次 Resident Final 约 18.1–18.3×，停止播放后两次稳定在 11.83–11.91×；差异主要来自低批次 Baseline 被拖慢约 2.5× | 高加速比是假象 | Final 测量前保持 GPU 独占；部署优劣只用同进程 AB/BA 配对决定

Shape 14 的 Reference 不适合作为性能候选 | 旧流程首轮把 730.9 秒耗在两个 Reference Screen 点；改造后两次独立 Smoke 均约 47 秒，各推进一个不同 Triton 分支点且无 Reference 调用 | 已改为 34 个高价值点的持久化无放回枚举，Reference 只作回退

Shape 14 microbatch=4 不适合 RTX 4080 | 最小协议正确性通过，但归一化延迟约 976 秒、峰值显存约 14.0 GB；相对当前 B1 约 20 秒差距近 49× | 收益方向明确为负且资源裕量过低 | 不加入搜索域，保持 microbatch 1/2

Shape 14 部署不应由源码摘要决定是否命中 | 源码摘要变化后已有 Triton 部署未命中，Formal 回退到约 1013.7 秒的 Reference，整轮 36.8 分钟仍未结束；改为按硬件/运行时身份命中并由 PlanBuilder 重验后，完整闭环 995.6 秒，候选 0.9998×被拒绝 | 源码摘要适合隔离测量证据，不适合作为部署路由键 | 部署按 measurement identity 复用，当前实现下重新测量和验证

Resident 新结构种子仍有显著价值 | `structure_seed=1235` 一轮完成 993 个新 Screen Trial，Shape 10/13 分别以 1.051×/1.099× Formal 晋升；Shape 06 补跑后完成覆盖，但通用 challenger 仅为融合 incumbent 的 0.450× | 旧种子局部饱和不等于程序空间饱和；Shape 06 已转为 incumbent 家族内局部问题 | Resident 继续轮换结构种子；Shape 06 不再均匀广搜

Shape 06 局部 TPE 已进入本轮低收益区 | 同一融合家族 challenger 一度从 0.450×改善到 0.912×，下一轮回落到 0.432×且 duplicate proposal 从 3 增到 11；28 个点后仍未接近 2% 门槛 | 已部署 tile 组合保持明显领先 | 暂停 Shape 06 调参，除非加入新的数据流机制

Shape 14 的 launch/microbatch 空间已经穷尽 | 34/34 个高价值点完成；`64×64 / 4 warps / 3 stages / microbatch 2` 以 1.1646×晋升，余下最佳候选 Formal 为 0.9987× | 继续增加相邻 launch 组合的预期价值很低 | 后续只考虑新的算法或跨算子融合 primitive

Resident 新种子收益集中在专用执行机制 | `structure_seed=1236` 的 Shape 07 以 D8 Triton Attention + Native QKV + Direct BSD + Triton FFN 获得 1.281×，Shape 09 以 cuDNN SDPA + Triton 边界获得 1.051×；其余 11 个 Formal 拒绝 | 程序族切换仍有收益，相邻通用调度多数已平台 | 下一轮优先探索尚未获得专用数据流的 Shape，不重复追加已拒绝候选

Resident 连续结构种子仍能产生增量赢家，但 Shape 06 已不适合例行轮换 | `structure_seed=1237` 中 Shape 02/07 分别约以 1.03×/1.067×晋升；Shape 06 用 218 秒只生成 5 个 Trial，最好候选仅 0.871× | Shape 06 的单位 GPU 时间信息价值显著低于其余 Shape | 例行脚本默认串行搜索 01–05、07–13；只有出现新 Shape 06 数据流机制时才通过 `-IncludeShape06` 显式纳入

旧 Resident 结构族已进入低边际区 | `structure_seed=1238` 在 12 个常规 Shape 上完成一轮、没有新部署；绝大多数 Formal 候选低于 incumbent，Shape 03 也未达到 2% 晋升线 | 继续轮换同类结构种子的预期收益低 | 暂停旧结构族整轮广搜，转向有数据流依据的新 primitive

Shape 12 融合 FFN 边界获得可重复部署收益 | 新积木在一个 Triton kernel 内完成 W1、Exact GELU、FP16 边界、W2、Residual 和 LayerNorm；36 个 Trial 后 Formal 成对加速 1.0303×并自动部署 | 隐藏激活不再写回和读出全局显存，每层减少一个边界 kernel | 保持 D=F=128 的窄搜索域；先验证第二个代表 Shape，再决定是否扩展

相同 FFN 矩阵形态可迁移融合收益 | Shape 04 与 Shape 12 展平后均为 2048×128；只把资格条件改为这一数学合同后，36 个 Trial 得到 9 个一致配对，Formal 从 0.115712 ms 降到 0.107520 ms，1.0762×晋升 | 收益显著高于 3% 扩展门槛，证明边界融合并非 Shape12 偶然调参 | 保留单一融合结构；下一步只做一个更大行数的窄迁移探针

融合 FFN 在更大行数和 Dh8 Attention 下仍成立 | Shape 11 用保持现有 Dh8 路径的单因素见证点开始搜索，36 个 Trial 后融合程序族与同族 launch 共同把 Formal 从 0.297984 ms 降到 0.268288 ms，1.1107×晋升 | 2048 与 8192 行两个尺度均有稳定收益，融合积木具备继续迁移依据 | 只扩展到同为 D=F=128 的形态；每次先做一个低成本目标，不做全量重跑

融合积木促成跨程序族执行图替换 | Shape 01 的 36 个 Trial 将 compiled-forward/torch 边界 incumbent 替换为 CUDA Graph、Native QKV 与融合 FFN，Formal 从 0.236544 ms 降到 0.217088 ms，1.0948×晋升 | 8192 行收益不依赖 Dh8 Attention，说明核心收益来自融合数据流 | 继续按 token-row 尺度逐个开放，不对 Shape06 或 Shape14 做无依据迁移

同尺度融合收益开始接近晋升边界 | Shape 10 的 36 个 Trial 用融合边界把 Formal 从 0.203776 ms 降到 0.198656 ms，13 个配对的中位加速为 1.0259×并晋升 | 相同 8192 行在不同 Head 数下收益从约 2.6% 到 11.1%，完整 Forward 的非 FFN 占比决定上限 | 保留已证部署；后续迁移必须保持原 Attention 家族并继续逐 Shape 证伪

新边界积木应与各 Shape 的最强 Attention 家族组合 | Shape 09 保留 cuDNN SDPA，只替换 FFN 边界后，Formal 从 0.235520 ms 降到 0.222208 ms，9 个配对中位加速 1.0557×并晋升 | 强制统一 Efficient SDPA 会混入 Attention 退化；结构合成应保留独立成熟库优势 | 搜索器按 HeadDim 与硬件能力生成 cuDNN、Efficient 或 Dh8 的单一融合结构

小行数形态不适合当前融合 FFN 边界 | Shape 03 的融合候选由 Screen 约 0.079 ms 退化到 Enhanced 约 0.165 ms；最终由已有精确 GELU/调度结构以 1.025×晋升。Shape 02 的最佳 challenger Formal 仅为 incumbent 的 0.854× | 128/512 行时单层仅 8/32 个 CTA，融合后占用率损失高于中间张量流量收益 | 执行层保留通用 D=F=128 合同，搜索层只在实测有效的 2048/8192 行开放；小行数等待持久化或启动专用方法
