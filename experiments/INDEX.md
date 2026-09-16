# 实验总表

规则：每个实验一行，**只追加，不修改历史行**。
目录命名：`experiments/E<阶段>.<编号>[_<变体>]_<slug>/`

> 状态说明（2026-09-16 晚）：
> 1. 所有实验的 `mask*.tif` 等可再生大文件已清理（experiments 体积 101GB → 32MB），
>    留痕证据（config/命令/env/git commit/指标/官方日志/图/结论）完整保留；
> 2. 下方标注 ⚠️ 的条目属于**链路偏离期**的结果：当时候选边用几何门控（非论文式26）、
>    GNN 全权决策（非残差校正）。这些结果将作为**消融对照**在 Phase B 重跑；
> 3. 详细的错误复盘与后续计划见 [`docs/reports/day1_summary.md`](../docs/reports/day1_summary.md)。

| 实验 ID | 日期 | 阶段 | 目的（自变量） | git commit | 核心指标 | 主图 | 结论 | 目录 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E0.1 | 2026-09-15 | P0 | 仓库初始化与实验管理脚手架 | (本提交) | — | — | 骨架就绪 | `experiments/E0.1_repo_skeleton/` |
| E0.2 | 2026-09-15 | P0 | CTC 3D 数据下载校验与结构统计（Fluo-N3DH-CHO） | (见各目录 git_commit.txt) | 92 帧/序列，27+28 轨迹，10+10 次分裂，帧间位移 p95=14 体素 | `object_count_over_time`, `distributions` | 数据可用；CHO 作为快速验证集，CE 为主力 | `experiments/E0.2_data_stats/` |
| E0.3 | 2026-09-15 | P0 | 评测管线自检（官方二进制 + 本地 SEG） | 同上 | SEG 0.232874/0.443686、DET 0.688000、TRA 0.622980 全部精确匹配 | — | 评测可信：权威指标在云端算，本地 SEG 一致 | `experiments/E0.3_eval_selfcheck/` |
| E1.1 | 2026-09-15 | P1 | 匈牙利基线（GT 检测上界，CHO/01） | 见目录 | **DET 1.000000 / TRA 0.998555**（FN=FP=0） | `diagnostics`, `tracks_gt`, `tracks_pred` | 链路打通；损失全部来自未识别分裂边 | `experiments/E1.1_baseline_hungarian/` |
| E1.2 | 2026-09-15 | P1 | 贪心最近邻（CHO/01） | 见目录 | DET 1.000000 / TRA 0.998555 | `diagnostics` | 与匈牙利同分 → CHO 无区分度 | `experiments/E1.2_baseline_greedy/` |
| E1.2b | 2026-09-15 | P1 | 匈牙利+匀速先验（CHO/01） | 见目录 | DET 1.000000 / TRA 0.998555 | `diagnostics` | 同上；运动先验收益待 CE 验证 | `experiments/E1.2_baseline_hungarian_vel/` |
| E2.1 | 2026-09-15 | P2 | 纯 OT（η=0，平衡 Sinkhorn） | 见目录 | DET 1.000000 / TRA 0.998555 | `diagnostics` | OT 求解器+轨迹重建正确；CHO 无区分度 | `experiments/E2.1_ot_pure/` |
| E2.3a | 2026-09-15 | P2 | FGW 结构项 η=0.3（运行验证） | 见目录 | DET 1.000000 / TRA 0.998555 | `diagnostics` | FGW 实现正确（梯度经有限差分校验），η 效果待 CE | `experiments/E2.3a_fgw_eta0.3/` |
| E1.3 | 2026-09-15 | P1 | **CE seq01 匈牙利基线**（分割无关上界） | 见目录 | **DET 1.000000 / TRA 0.995594** | `CE01_comparison` | CE 上的 reference 数字；修掉幽灵轨迹 bug | `experiments/E1.3_CE01_hungarian/` |
| E2.1 | 2026-09-15 | P2 | CE seq01 纯 OT | 见目录 | DET 1.000000 / TRA 0.994361 | 同上 | IDsw 更少但碎片化 3.5×，净负 | `experiments/E2.1_CE01_ot_pure/` |
| E2.5 | 2026-09-16 | P2 | CE θ_Γ×div_ratio 扫描 | 见目录 | 本地指标（TRA 由 E2.5c 裁决） | — | θ_Γ 几乎无影响；div_ratio 是主杠杆 | `experiments/E2.5_CE01_theta_div_sweep/` |
| E2.5b/c | 2026-09-16 | P2 | CE 分裂判据精调（div_sum_min, ε） | 见目录 | DET 1.000000 / TRA 0.994617 | `CE01_comparison` | 碎片↔IDsw 权衡，超参收益有限 | `experiments/E2.5c_CE01_ot_eps0.3/` |
| E2.6 | 2026-09-16 | P2/P3 | **分裂事件 P/R 诊断**（定位 ED 多余边来源） | 见目录 | 匈牙利 P=1.000/R=0.112；OT P=0.446/R=0.475 | — | 手工判据精确率封顶 0.48 → 必须学习（P4） | `experiments/E2.6_CE01_division_diag/` |
| E4.1 | 2026-09-16 | P4 | 动态图 GNN 边分类训练（3 分类） | 见目录 | move F1 **0.9931** / division F1 **0.6215** | `training_history` | 图模型可学到"何时该分" | `experiments/E4.1_gnn_ce01/` |
| E4.2d | 2026-09-16 | P4 | **GNN 追踪官方评测（首次超过基线）** | 见目录 | **DET 1.000000 / TRA 0.996089**（IDsw 111，div P/R 0.854/0.802） | `training_history`, `diagnostics` | AOGM 1070.5 vs 基线 1206（-11%），ED -79%、EC -74%，**验证 §2.0.1** | `experiments/E4.2d_gnn_ce01_official/` |
| E4.3 | 2026-09-16 | P4 | GNN 决策阈值扫描（tau_move×tau_div） | 见目录 | tm0.5/td0.5 最优 TRA 0.996089 | — | 提高分裂阈值反而伤 TRA：瓶颈是召回不是精确率；本地指标与 TRA 不单调 | `experiments/E4.3_gnn_threshold_sweep/` |
| E4.4 | 2026-09-16 | P4 | 多步时间上下文消融（window=1） | 见目录 | TRA 0.996023（略低于 0.996089） | — | **负结果**：原始帧级上下文无收益，多步应落在 tracklet 层 | `experiments/E4.4b_gnn_ctx1_official/` |
