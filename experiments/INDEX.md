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
| **A2** | 2026-09-19 | PhaseA | **相邻帧 OT 阶段接入（η/τ）+ 消融** | 见目录 | 先验 AUC: η=0 **0.9931** / η=0.5 0.9870；τ=1 0.9873 | — | 候选覆盖度与 η/τ 无关；η/τ 均略降先验判别力；ε 与手工阈值强耦合 | `experiments/A2_ot_stage_ablation/` |
| **A3** | 2026-09-19 | PhaseA | **运动先验（式20-22）两遍式接入** | 见目录 | 功能验证通过（`velocity_live=True`，23102 个检测有前驱） | — | 死特征已打通；性能消融留到 Phase B | `experiments/A3_motion_prior/` |
| **A4** | 2026-09-19 | PhaseA | **多尺度时间正则（式17-19）接入** | 见目录 | 接口链路打通（OT→多尺度→图构建）；修 3 个缺陷后精炼生效（计划改变 7/7） | — | 发现质量尺度错配/归一化跳过/无线搜索三个真 bug；下降效率待调优 | `experiments/A4_multiscale/` |
| **A5** | 2026-09-19 | PhaseA | **第二层 tracklet OT（§1.6）接入** | 见目录 | 接口通过（610→605 轨迹，拆断 0）；CTC 格式合法 | — | 第二层必须非平衡 OT；跨空洞合并受 CTC 格式约束，需 Phase C 补检测 | `experiments/A5_tracklet_ot/` |
| **A6** | 2026-09-19 | PhaseA | **统一 runner + 接口收口（Phase A 完成）** | 见目录 | 七阶段端到端跑通（9 帧 74 秒）；消融/配置接口齐备 | — | 修 OT 规则阈值语义错位；多尺度线搜索性能问题待 Phase B 优化 | `experiments/A6_pipeline_runner/` |
| **B0** | 2026-09-19 | PhaseB | **GNN 接入主链路（适配层）** | 见目录 | GNN 路径 510 轨迹 / OT 规则 634（16 帧，仅验证链路） | — | 单一配置源；消融开关可切换决策阶段 | `experiments/B0_gnn_adapter/` |
| **B1** | 2026-09-19 | PhaseB | **论文口径训练 GNN + 全链路评测（目前最好）** | 见目录 | **DET 1.000000 / TRA 0.996513**，AOGM 954.5（比基线低 21%） | — | 严格按论文口径即带来实质提升：ED 90→58、碎片 301→174 | `experiments/B1_gnn_paper_ce01/` |
| **B2** | 2026-09-19 | PhaseB | **消融矩阵首批（GNN / tracklet）** | 见目录 | OT规则 0.993135 / GNN 0.996513 / **GNN+tracklet 0.996622** | — | GNN 贡献 +0.0034（手工规则找不到好工作点）；tracklet 使碎片 174→79 | `experiments/B2_ablation_ce01/` |
| **B2-m** | 2026-09-19 | PhaseB | **消融方法论修正：噪声地板与方向** | 见目录 | 噪声地板 **0.0007**；式(26) −0.0012、top-k −0.0011 | — | ΔTRA<0.001 无法与噪声区分；模块默认关闭者应测"打开" | `experiments/B2_methodology/` |
| **B2-c** | 2026-09-19 | PhaseB | **消融矩阵完整表**（7 项，均重训 GNN） | 见目录 | 基准 0.996513；GNN −0.0034、式26 −0.0012、top-k −0.0011；FGW/非平衡/运动先验均**负收益** | — | 学习型决策最关键；论文三个扩展在本数据上无正收益（需参数扫描确认） | `experiments/B2c_ablation_matrix/` |
| **B2-d/e** | 2026-09-19 | PhaseB | **参数扫描：η / τ / α′** | 见目录 | FGW 损害与 η 无关；非平衡随 τ 单调收敛向平衡；运动先验非单调且两值均负 | — | 与 A2 先验 AUC 两条独立证据一致 → 朴素平衡 OT 已是最优先验 | `experiments/B2de_param_sweeps/` |
| **B3** | 2026-09-19 | PhaseB | **seq02 复验（双序列一致性）** | 见目录 | **seq02 TRA 0.996369**（基线 0.995052），AOGM 936（比基线低 27%），碎片 67 | — | 论文口径在双序列上一致胜出；修掉 top-k 越界 latent bug | `experiments/B3_seq02_validation/` |
| **C1** | 2026-09-19 | PhaseC | **nnU-Net 推理（CE 双序列 385 帧，真实检测来源）** | 见目录 | 385 帧 / 33MB；训练验证 Dice 0.9605；36 分钟（4090） | `instances_vs_markers` | 语义质量高，但拆实例后平均 50.4 实例/帧 vs 122.1 标记（0.41） | `experiments/C1_nnunet_infer/` |
| **C2** | 2026-09-19 | PhaseC | **语义掩码→实例拆分标定（间距×体积×h-maxima）** | 见目录 | 所有工作点实例/标记 0.29–0.86；召回≈1.0；exclusive 0.34–0.59 | `calibration_tradeoff` | 后处理无法解决欠分割；**F1 奖励欠分割、不可作选参判据**；发现"标定配置≠部署配置"（h_frac 未记录）并已修 | `experiments/C2_instance_split/` |
| **C5.0** | 2026-09-20 | PhaseC | **EDT 物理间距修复 + 实例拆分重新标定** | 见目录 | 实例/标记 **0.41 → 1.007**；检测天花板 U0 0.407→**0.905**、U1 0.257→**0.846**、U3 0.221→**0.723**；本地 IDsw 16507→**3671**、碎片 14463→**6431** | `spacing_recalibration` | 后处理从未"走到头"——是 **EDT 未按物理间距计算**（z 粗 11 倍）导致的人为欠分割；修复后无需重训即恢复实例数 | `experiments/C5.0_spacing_fix/` |
| **C5.0-f** | 2026-09-20 | PhaseC | **漏斗复测：修复前端后的剩余瓶颈** | 见目录 | move 边 S3 留存 **71.0%**（≈GT 上界 73%）；分裂事件级 S1 48.2% → S2 13.8% → **S3 2.2%** | — | 瓶颈已从"检测"转移到**候选生成+分裂判定**；Distance Head 优先级下调 | `experiments/C5.0_spacing_fix/` |
| **C5.0b** | 2026-09-20 | PhaseC | **修分裂标签污染（gt_parent 空表 → 假阳性边被标成分裂）** | 见目录 | 分裂候选标签 5866→**378**（真实 453）；分裂事件 S3 2.2%→**21.1%**；本地分裂精确率 0.011→**0.489**、碎片 6431→**3479** | — | 标签 bug 让分裂头在垃圾标签上训练；C4a"GNN 不如匈牙利"的结论需重测 | `experiments/C5.0b_label_fix/` |
| **C5.0c** | 2026-09-20 | PhaseC | **候选规则变体（cand_topk / theta_gamma）** | 见目录 | 分裂事件候选覆盖 48.2%→**78.0%**（topk 3→5）；θ_Γ 几乎无影响；本地指标互有胜负（碎片 3479→3021，IDsw 3832→3971） | — | 候选可扩、但 S1 78%→S2 35% 说明**决策阶段是下一瓶颈**；参数改动待官方指标裁决 | `experiments/C5.0c_cand_rule_sweep/` |
| **C4a** | 2026-09-20 | PhaseC | **预测检测建图 + 同源重训 GNN + 检测层天花板** | 见目录 | 天花板 U0 0.407 / U1 0.257 / U3 0.221；本地 IDsw 16507、碎片 14463 | — | 首次量化"追踪之前丢了多少"；结论受 C5.0/C5.0b 两个 bug 影响，已追加更正说明 | `experiments/C4_pred_graphs/`、`C4_pred_train/`、`C4_pred_hungarian/`、`C4_local_check/` |
| **C4A-o** | 2026-09-21 | PhaseC | **全链路官方评测（真实检测档，论文口径 topk=3）** | 见目录 | DET **0.939833** / SEG **0.673196** / **TRA 0.905706**；AOGM 25810（EA 7602、NS 2199、FP 2456 为主） | `official_tiers` | 真实检测档端到端基线落定；误差主因是缺失边/多余分裂/假阳性节点 | `experiments/C4A_official_gnn_topk3/` |
| **C4B-o** | 2026-09-21 | PhaseC | **全链路官方评测（真实检测档，cand_topk=5）** | 见目录 | DET/SEG 同上；**TRA 0.906542**（Δ +0.00084）；EA 7602→7448、EC 57→65 | `official_tiers` | top-k 扩容方向一致，但**幅度仅 1.2× 噪声地板，不足以改默认参数**，需多种子复验 | `experiments/C4B_official_gnn_topk5/` |
| **C4O-o** | 2026-09-21 | PhaseC | **Oracle 实例上界（GT 标记作种子，量化前端收益上限）** | 见目录 | **TRA 0.984784** / DET 0.991581 / SEG 0.688720；AOGM 4165（FP 2456→**9**、NS 2199→**211**） | `official_tiers` | 前端实例化上限 ≈ **+0.079 TRA**，是追踪侧改动的 10× 量级；**SEG 几乎不变（0.673→0.689）→ 瓶颈是实例归属而非掩码质量**；Distance Head 值得做 | `experiments/C4O_oracle_upper_bound/` |
| **C5.0d** | 2026-09-23 | PhaseC | **OT 代价/R_max 改用物理单位（µm）** | 见目录 | 官方 **TRA 0.904473 vs A 0.905706（−0.00123）**；AOGM：ED 29→11、EC 57→27 变好，但 **EA 7602→7859 变差** | — | **官方否定：不采纳**（收紧 z 门限删掉了真实关联；NS 不变印证"NS 由掩码决定"）；顺带修掉 `window>0` NameError | `experiments/C5.0d_phys_units/` |
| **C5.0e** | 2026-09-23 | PhaseC | **后处理上界诊断（超大实例再切 / 孤立短轨迹过滤）** | 见目录 | **规则 A（k=1.6）：官方 TRA 0.926890（+0.02118，30× 噪声地板）**，AOGM 25810→20011（**NS 2199→1265、EA 7602→6225**、FP 2456→3067）；**规则 B（L=2）：TRA 0.852466（−0.053）** | `official_tiers` | **A 采纳为当前最佳**（零重训、单相对参数，拿回 Oracle 上界的 27%）；**B 为负结果**：论文 §2.0.1"孤立即假阳性"正式证伪 | `experiments/C5.0e_postproc_rules/` |
| **C5.0f** | 2026-09-23 | PhaseC | **再切规则 k 扫描（seq01）+ 跨序列验证（seq02 留出集）** | 见目录 | seq01：k=1.3 **0.927615** / k=1.6 **0.926890** / k=2.0 0.918808（无再切 0.905706）；seq02：0.909340 → **0.926320（+0.01698）** | `resplit_sweep` | 最优平台 k≈1.3–1.6（差 < 噪声地板）；**留出集同向有效 → 非 seq01 过拟合**；NS 腰斩、FP 上升的机制在双序列一致 | `experiments/C5.0f_resplit_sweep/` |
| **协议审查** | 2026-09-23 | 决策 | **seq02 数字的评测协议与泄漏量化** | 见 `docs/decisions/0003` | fold0 train 含 **150/190 = 79% 的 seq02 帧**；GNN 在 seq02 上训练（in-sequence）；按帧分组的官方 AOGM：留出帧 95.3/帧 **优于** 见过帧 101.2/帧 | — | 该数字**不支持**跨序列/端到端泛化声明；只支持"再切参数可迁移"。要泛化证据须做跨序列 GNN（A）或跨序列分割重训（B） | `docs/decisions/0003-seq02-eval-protocol.md` |
| **P0-pipe** | 2026-09-23 | P0 | **`paperpipe/`：严格按 `ideas.pdf` 重建 pipeline**（符合原文的模块直接复用，偏离项重写；参数按量纲标定） | 见目录 | 新增 15/15 测试通过 + 原 89 项不回归；CE-01 帧 120–130：OT 规则路径 601 轨迹 / GNN 路径 614 轨迹，两条路径的 CTC 提交**格式校验全部通过**（空洞 22/48 帧全部补画、0 幽灵轨迹）；图数据集 10 张 / 6404 节点 / 1375 正样本；冒烟暴露并修掉 4 个问题（多尺度 NaN 溢出+16× 超时、tracklet 轨迹 id 为 0、桥接边过度触发 537→22、空洞无处补画） | — | 严格论文口径链路可用；**下一步：云端 nnU-Net 检测 → 重训 GNN → 官方指标**；2 处口径待用户拍板（空洞补画 / θ_Γ 标定方式，见 `paperpipe/FORMULA_MAP.md` §3） | `experiments/P0_paperpipe_build/` |
