# 实验总表

规则：每个实验一行，**只追加，不修改历史行**。
目录命名：`experiments/E<阶段>.<编号>[_<变体>]_<slug>/`

| 实验 ID | 日期 | 阶段 | 目的（自变量） | git commit | 核心指标 | 主图 | 结论 | 目录 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E0.1 | 2026-09-15 | P0 | 仓库初始化与实验管理脚手架 | (本提交) | — | — | 骨架就绪 | `experiments/E0.1_repo_skeleton/` |
| E0.2 | 2026-09-15 | P0 | CTC 3D 数据下载校验与结构统计（Fluo-N3DH-CHO） | (见各目录 git_commit.txt) | 92 帧/序列，27+28 轨迹，10+10 次分裂，帧间位移 p95=14 体素 | `object_count_over_time`, `distributions` | 数据可用；CHO 作为快速验证集，CE 为主力 | `experiments/E0.2_data_stats/` |
| E0.3 | 2026-09-15 | P0 | 评测管线自检（官方二进制 + 本地 SEG） | 同上 | SEG 0.232874/0.443686、DET 0.688000、TRA 0.622980 全部精确匹配 | — | 评测可信：权威指标在云端算，本地 SEG 一致 | `experiments/E0.3_eval_selfcheck/` |
| E1.1 | 2026-09-15 | P1 | 匈牙利基线（GT 检测上界，CHO/01） | 见目录 | **DET 1.000000 / TRA 0.998555**（FN=FP=0） | `diagnostics`, `tracks_gt`, `tracks_pred` | 链路打通；损失全部来自未识别分裂边 | `experiments/E1.1_baseline_hungarian/` |
| E1.2 | 2026-09-15 | P1 | 贪心最近邻（CHO/01） | 见目录 | DET 1.000000 / TRA 0.998555 | `diagnostics` | 与匈牙利同分 → CHO 无区分度 | `experiments/E1.2_baseline_greedy/` |
| E1.2b | 2026-09-15 | P1 | 匈牙利+匀速先验（CHO/01） | 见目录 | DET 1.000000 / TRA 0.998555 | `diagnostics` | 同上；运动先验收益待 CE 验证 | `experiments/E1.2_baseline_hungarian_vel/` |
