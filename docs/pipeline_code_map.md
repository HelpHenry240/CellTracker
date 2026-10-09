# Pipeline 骨架代码标注：模块 ↔ 论文公式 ↔ 代码文件



---

## 0. 怎么读这份文档

每一行回答四个问题：

1. **论文出处**：对应 `ideas.pdf` 的哪一节、哪个公式；
2. **代码位置**：`文件::函数`（行号仅为当前基线的定位辅助，函数名才是稳定锚点）；
3. **作用**：这一步在整条链路里承担什么职责、输入输出是什么；
4. **配置 / 开关**：哪个配置字段控制它，哪个 `--ablate` 能关掉它。

---

## 1. 端到端骨架（一张图）

```
输入： CTC 序列（原图 + _GT/TRA + _GT/SEG）
       │
       ├─ S0  数据层 / 检测来源
       │      data/ctc.py              解析 CTC 目录、TRA 标记 → 轨迹血缘
       │      data/build_dataset.py    CTC → 内部 h5（逐帧 label/centroid/volume/强度）
       │      detect/instances.py      语义掩码 → 实例（EDT + h-maxima + watershed）
       │      scripts/predict_to_h5.py nnU-Net 掩码 → 预测检测 h5（带 gt_label）
       │                                     ↓ Detections
       ├─ S1  §1.2 测度与帧内图        cost/features.py
       │                                     ↓ μ^t, D^t
       ├─ S2  §1.3 相邻帧 OT           pipeline/ot_stage.py（+ ot/sinkhorn.py, ot/fgw.py）
       │                                     ↓ {Γ_t}, {C_t}
       ├─ S3  §1.5 运动先验（两遍式）   pipeline/motion.py + pipeline/runner.py
       ├─ S4  §1.4 多尺度时间正则       pipeline/multiscale_stage.py（+ ot/multiscale.py）
       │                                     ↓ 精炼后的 {Γ_t}
       ├─ S5  §2.0.1 时间展开图         graph/build.py::build_pair_graph
       │                                     ↓ 节点/边特征 + 3 分类标签
       ├─ S6  §2.0.1 决策               gnn/（论文口径）
       │        或 §1.6 OT 规则         pipeline/runner.py::ot_rule_reconstruct（消融对照）
       │                                     ↓ TrackResult（第一层）
       ├─ S7  §1.6 第二层 tracklet OT   pipeline/tracklet_stage.py
       │                                     ↓ 合并后的 TrackResult
       ├─ S8  §1.6 轨迹重建 + CTC 校验   track/base.py::finalize_tracks
       │                                     ↓ 合法提交（mask + res_track）
       └─ S9  评测                      eval/ctc_io.py, eval/local_metrics.py,
                                        scripts/cloud_eval.py（官方二进制，云端）
```

主控函数只有一个：**`src/celltracker/pipeline/runner.py::run_pipeline`**（`runner.py:133`）。
它按上图的顺序串起七个阶段，每个阶段都可以经 `--ablate` 关闭或 `--set` 改参。

---

## 2. 一页速查总表

| 阶段 | 论文 | 公式 | 作用 | 代码位置 | 配置段 | 消融开关 |
| --- | --- | --- | --- | --- | --- | --- |
| S0 数据解析 | — | — | CTC 目录 → 逐帧检测与 GT 血缘 | `data/ctc.py::CTCSequence`、`read_man_track`、`object_table_from_labels` | — | — |
| S0 内部 h5 | — | — | 序列 → 统一 h5 中间表示 | `data/build_dataset.py::build` | — | — |
| S0 实例拆分 | §2.0.1（前端） | — | 语义掩码 → 单细胞实例 | `detect/instances.py::split_instances` | `InstanceSplitConfig` | — |
| S0 预测检测 h5 | §2.0.1（前端） | — | 实例 → 与 GT 同构的检测 h5 | `scripts/predict_to_h5.py` | CLI | — |
| S1 经验测度 | §1.2 | 式(1)(2)(3) | 每个细胞的 OT 质量 `a_i` | `cost/features.py::masses` | `measure.mass_mode` | — |
| S1 帧内 kNN 图 | §1.2 | 式(4)(5)(6)(7) | 提供结构项用的 `D^t`、边权 `W` | `cost/features.py::gaussian_knn_graph` | `measure.knn_k/sigma_x` | — |
| S2 特征代价 | §1.3 | 式(8) | `C_feat`：位移 + 尺寸差 + `R_max` 门限 | `cost/features.py::build_cost` | `ot.alpha/beta/sigma_s/r_max` | — |
| S2 运动先验代价 | §1.5 | 式(22) | 追加匀速外推残差项 | `cost/features.py::build_cost`（`alpha_pred>0`） | `ot.alpha_pred` | `motion` |
| S2 FGW 结构项 | §1.3 | 式(9) | 惩罚局部结构扭曲 | `ot/fgw.py::fused_gw`、`structural_term/grad` | `ot.eta` | `fgw` |
| S2 熵正则 OT | §1.3 | 式(11)(12) | Sinkhorn 求解、数值稳定 | `ot/sinkhorn.py::sinkhorn_log` | `ot.eps/eps_rel` | — |
| S2 非平衡 OT | §1.3 | 式(13)(14) | KL 松弛，允许出生/死亡/分裂 | `ot/sinkhorn.py::sinkhorn_log`（`tau_a/b`） | `ot.tau_a/tau_b` | `unbalanced` |
| S2 OT 阶段封装 | §1.3 | 式(8)(9)(11)(12)(14)(22) | 统一求解 + 落盘复用 | `pipeline/ot_stage.py::compute_pairwise_plan` | `OTConfig` | — |
| S3 两遍式估速 | §1.5 | 式(20)(21) | 由第一遍硬关联估速并外推 | `pipeline/motion.py::estimate_velocity` | `ot.alpha_pred>0` | `motion` |
| S3 速度回写 | §1.5 | 式(20)(21) | 速度写回检测表（打通死特征） | `pipeline/motion.py::attach_velocity` | — | — |
| S4 跳帧直接 OT | §1.4 | 式(19) | `D_{t,t+k}`：跨 k 帧的直接耦合 | `ot/multiscale.py::direct_jump_coupling` | `multiscale.ks` | — |
| S4 时间正则 | §1.4 | 式(17)(18)(19) | 多步复合与直接耦合的不一致度 | `ot/multiscale.py::temporal_regularizer`、`pipeline/multiscale_stage.py::refine_couplings` | `multiscale.lambda_temp/n_rounds` | `multiscale` |
| S5 节点特征 | §2.0.1 | 式(25) | `[x, s, f, t̃]`（实际 7 维） | `graph/build.py::_node_features` | `graph.*` | — |
| S5 候选边 | §2.0.1 | 式(26) | `{Γ≥θ_Γ} ∩ {C≤θ_C}` + top-k 保底 🔧 | `graph/build.py::build_pair_graph`（候选筛选段） | `graph.cand_from_ot/theta_gamma/theta_c/cand_topk` | `ot_cand`、`cand_topk` |
| S5 边特征 | §2.0.1 | 式(27) | 10 维边特征（成本/质量/位移/尺寸/先验/argmax） | `graph/build.py::_edge_feature_matrix` | — | — |
| S5 帧内边 | §2.0.1 | 式(28) | 帧内 kNN 上下文 | `graph/build.py`（`intra_edges`） | `graph.intra_knn` | — |
| S5 边标签 | §2.0.1 | 式(29) | 移动/分裂监督（三分类扩展） | `graph/build.py::build_pair_graph`（标签段） | — | — |
| S6 GNN 消息传递 | §2.0.1 | 式(30)(31)(32) | 边–节点交替更新 | `gnn/model.py::EdgeGNN.forward` | `ModelConfig.hidden/layers` | — |
| S6 边分类头 | §2.0.1 | 式(33) | 候选边类别概率 | `gnn/model.py::EdgeGNN`（`edge_head`） | — | — |
| S6 边损失 | §2.0.1 | 式(34) | 边级交叉熵（三分类） | `gnn/train.py::train` | — | — |
| S6 OT 一致性正则 | §2.0.1 | 式(35) | `L_OT-reg = Σ ŷ_e·C_e` | `gnn/train.py::train`（`lambda_ot`） | `TrainConfig.lambda_ot` | — |
| S6 GNN 推理 | §2.0.1 | 式(30)–(33) | 内存内构图 → 概率 → 轨迹 | `gnn/infer.py::predict_from_couplings`、`reconstruct_tracks` | `ReconstructConfig` | `gnn` |
| S6 适配层 | §2.0.1 | — | 把 GNN 插回主链路（单一配置源） | `gnn/adapter.py::make_gnn_runner`、`graph_cfg_from_pipeline` | — | — |
| S6 OT 规则重建 | §1.6 | 式(23)(24) | 无 GNN 时的对照路径 | `pipeline/runner.py::ot_rule_reconstruct` | `reconstruct.div_ratio` | `gnn` |
| S7 超级节点 | §1.6 | — | tracklet 汇总特征（时长/首末/速度） | `track/tracklets.py::tracklet_stats` | — | — |
| S7 第二层 OT | §1.6 | 式(14)(20-22)（tracklet 级） | tracklet 级关联与合并 | `pipeline/tracklet_stage.py::link_tracklets` | `TrackletConfig` | `tracklet` |
| S8 轨迹重建 | §1.6 | 式(23)(24) | 软计划 → 硬关联 → 轨迹 + 谱系 | `track/base.py::run_tracking`、`pipeline/runner.py::ot_rule_reconstruct`、`gnn/infer.py::reconstruct_tracks` | `reconstruct.*` | — |
| S8 格式校验 | — | — | CTC 合法性（幽灵轨迹/断裂/父子）🔧 | `track/base.py::finalize_tracks` | — | — |
| S9 提交写盘 | — | — | 结果 → CTC 提交格式 | `eval/ctc_io.py::ResultWriter`、`write_result` | — | — |
| S9 本地指标 | — | — | SEG + 诊断量（IDsw/碎片/分裂 P-R） | `eval/local_metrics.py::seg_measure`、`StreamingDiagnostics` | — | — |
| S9 官方指标 | — | — | 云端官方 DET/SEG/TRA | `scripts/cloud_eval.py` | CLI `--official` | — |

---

## 3. 逐阶段详解

### S0 数据层与检测来源（链路的最前端，决定一切上游口径）

| 模块 | 代码位置 | 作用与要点 |
| --- | --- | --- |
| CTC 目录解析 | `data/ctc.py::CTCSequence`（`ctc.py:75`） | 发现 `t*.tif` / `man_track*.tif` / `man_seg*.tif`，支持 3D 多页 TIFF 与逐切片金标准（`man_seg_T_Z.tif`） |
| 轨迹表读写 | `data/ctc.py::Track`、`read_man_track`（`ctc.py:27,44`） | `L B E P` 四元组 = 标签/起始帧/结束帧/父标签，是边标签与分裂判据的唯一来源 |
| 实例统计 | `data/ctc.py::object_table_from_labels`（`ctc.py:205`） | 标签体数据 → 质心/体积/包围盒/强度统计 |
| 内部 h5 | `data/build_dataset.py::build`（`build_dataset.py:34`） | 逐帧流式写入 h5；预测检测 h5 与之**同构**，故下游零改动 |
| 数据画像 | `data/summarize.py::summarize` | 细胞数随时间、尺寸/位移分布、分裂时间线（用于标定 `R_max`、`σ_s`） |
| 实例拆分 | `detect/instances.py::split_instances`（`instances.py:52`） | EDT（**必须传物理间距**）→ 高斯平滑（单位 µm）→ h-maxima 种子 → watershed；`seeds=` 时跳过找峰，供 Oracle 实验 |
| 拆分质量评估 | `detect/instances.py::detection_recall_vs_markers`（`instances.py:128`） | 用 GT 标记量化召回/精确/**exclusive**（独立性）；⚠️ F1 在欠分割时单调升高，**不可作选参判据** |
| 预测检测 h5 | `scripts/predict_to_h5.py` | nnU-Net 掩码 → 实例 → h5，并写入 `gt_label`（每个检测对应的 GT 轨迹 id）→ 满足 R11「训练/测试检测来源同分布」 |
| 拆分标定 | `scripts/calibrate_instance_split.py` | 用 GT 标记扫 `h_frac × min_volume × gaussian_sigma`，选「实例/标记 ≈ 1.0 且 exclusive 最高」的工作点 |

> **两条容易踩的口径**（都已在 C0/C5.0 事故中吃过亏）
>
> 1. `_GT/TRA` 是**等体积标记点**而非真实细胞核（marker:核 ≈ 1:23），所以质量与尺寸在 marker 上被解耦；
> 2. EDT 不传 `sampling` 会把 z 方向的 1.0 µm 当成 1 个体素，造成人为的 4–6 倍欠分割。

### S1 §1.2 帧内表示：经验测度 + 邻域图

| 公式 | 代码位置 | 作用 |
| --- | --- | --- |
| 式(1) `a_i = s_i / Σ_k s_k` | `cost/features.py::masses`（`features.py:55`） | 生成 OT 的边际质量向量；`mass_mode="uniform"` 时取 `1/n_t`（依据 `docs/decisions/0001`），`"volume"` 保留作对照 |
| 式(2)(3) 特征 `f=[x, s]` | `track/base.py::Detections`（`base.py:12`） | 逐帧检测容器；`centroid()` / `volume()` / `gt_label()` 是下游统一取值接口 |
| 式(4)(5)(6)(7) 帧内 kNN 图 | `cost/features.py::gaussian_knn_graph`（`features.py:64`） | 返回 `(D, W)`：`D` 是 kNN 截断的欧氏距离矩阵（非邻接处置 0，等价于「无结构项贡献」），`W` 是空间 + 特征相似度边权 |

### S2 §1.3 相邻帧最优传输（链路的核心求解器）

| 公式 | 代码位置 | 作用 |
| --- | --- | --- |
| 式(8) `C_feat = α‖x_i−x_j‖² + β((s_i−s_j)/σ_s)²`（+`R_max` 门限） | `cost/features.py::build_cost`（`features.py:29`） | 构造代价矩阵；超出 `R_max` 的配对置 `+inf`（**不可逆筛选**，必须保证候选覆盖） |
| 式(22) 运动先验项 | `cost/features.py::build_cost`（`alpha_pred>0` 分支） | 追加 `α′‖x̂^{t+1}_i − x^{t+1}_j‖²` |
| 式(9) FGW 结构项 | `ot/fgw.py::structural_term` / `structural_grad` / `fgw_objective` / `fused_gw`（`fgw.py:22,30,39,45`） | 条件梯度（Frank–Wolfe）+ Sinkhorn 线性 oracle；结构项用矩阵形式 `rᵀ(D∘D)r + cᵀ(D′∘D′)c − 2⟨P, DPD′⟩` 避免四重和 |
| 式(11) 熵正则 | `ot/sinkhorn.py::sinkhorn_log`（`sinkhorn.py:34`） | log-domain 迭代，允许很小的 ε |
| 式(12) 平衡约束 | 同上（`tau_a=tau_b=None`） | 硬边际 `P1=a, Pᵀ1=b`；额外用**边际违反量**判据 `tol_marg` 收敛（只判对偶变量会把边际误差留在 1e-5） |
| 式(13)(14) 非平衡 OT | 同上（`tau_a/tau_b` 给定） | `λ = τ/(τ+ε)`；允许出生/死亡/分裂 |
| 阶段封装 + 落盘 | `pipeline/ot_stage.py::compute_pairwise_plan`（`ot_stage.py:59`）、`CouplingArtifacts`（`ot_stage.py:36`） | 统一求解并返回 `plan/cost/mass/eps_eff/d_cur`；`rnorm` 属性 = 行归一化传输质量（式(26) 判据与式(27) 特征都用它） |

> ⚠️ **ε 必须按代价尺度标定**：`C` 是平方距离（CE 上中位数量级 ~4×10²），直接取 `ε=1` 会让计划退化成硬分配。
> 代码用 `eps_rel=0.1`（即 `ε = 0.1 × median(C)`）自适应，属 🔧 工程补充。

### S3 §1.5 运动先验（两遍式，不是在线估速）

| 公式 | 代码位置 | 作用 |
| --- | --- | --- |
| 式(20) `v = x^t − x^{t−1}_{p(i)}` | `pipeline/motion.py::estimate_velocity`（`motion.py:35`） | 由**上一轮追踪**的硬关联估速，同时返回 `valid` 区分「静止」与「无前驱」 |
| 式(21) `x̂ = x + v` | `pipeline/runner.py`（`runner.py:175` 使用 `pred_xy`） | 外推预测位置，喂给 `compute_pairwise_plan(pred_xy=...)` |
| 式(22) 带先验的代价 | `cost/features.py::build_cost` + `pipeline/runner.py`（`runner.py:163`） | 第 1 遍 `α′=0` 跑粗追踪 → 估速回写 → 第 2 遍带 `α′` 重跑 |
| 死特征检查 | `pipeline/motion.py::velocity_feature_is_live`（`motion.py:75`） | 验收断言：速度必须真的非零。历史上 GNN 图里的 `src_vel` 恒为 0，式(22) 形同虚设 |
| 独立两遍式驱动 | `pipeline/motion.py::run_two_pass`（`motion.py:88`） | 供不受 pipeline runner 约束的场景（测试、独立实验） |

### S4 §1.4 多尺度时间一致性与全局目标

| 公式 | 代码位置 | 作用 |
| --- | --- | --- |
| 式(17) 全局目标（含 `λ_temp` 正则） | `pipeline/multiscale_stage.py::refine_couplings`（`multiscale_stage.py:48`） | 交替优化：先逐对独立求解，再按时间正则梯度重解 |
| 式(18) 两步复合一致性 | 同上（`_cond` + 梯度段） | `Γ_tΓ_{t+1}` 与 `D_{t,t+2}` 的 Frobenius 差；**必须先做行归一化**，否则两边质量尺度不同（`(1/n)²` vs `1/n`） |
| 式(19) 多尺度 `R^(k)`，`k∈{2,3,5}` | `ot/multiscale.py::temporal_regularizer`（`multiscale.py:54`）、`direct_jump_coupling`（`multiscale.py:31`） | 跳帧直接 OT（`R_max` 按 gap 线性放宽）+ 多步复合的不一致度 |
| 精炼入口 | `pipeline/multiscale_stage.py::refine_couplings` | `enabled=False` 时是**完全恒等映射**（消融对照干净）；含 5 次半砍步长的回溯线搜索，保证正则单调不增 |

> ⚠️ 该模块目前只能标「接口可用」：三个缺陷（质量尺度错配 / 梯度归一化被静默跳过 / 无线搜索过冲）已修，
> 但**下降效率仍待优化**，`λ_temp` 建议区间仅 `[0.01, 0.2]`（详见 `AGENTS.md` C9）。

### S5 §2.0.1 时间展开图构建

| 公式 | 代码位置 | 作用 |
| --- | --- | --- |
| 式(25) 节点特征 `h⁰=[x, s, f, t̃]` | `graph/build.py::_node_features`（`build.py:56`） | 实际维度 **7** = 归一化坐标(3) + `log1p(vol)`(1) + 强度均值(1) + 强度标准差(1) + 归一化时间(1)；`f` 目前是**手工强度统计**，尚未接 nnU-Net encoder 特征 |
| 式(26) 候选边 | `graph/build.py::build_pair_graph`（`build.py:159-186`） | `{(i→j) : Γ_ij ≥ θ_Γ 且 C_ij ≤ θ_C}`；OT 在这里是「**决定候选边的人**」，GNN 只在候选集内判定 |
| 式(26) 的 top-k 保底 🔧 | 同上（`cand_topk` 段） | 每行额外保留传输质量前 k 个目标，避免真实分裂的第二子目标被不可逆滤掉；`k_eff = min(k, n_dst)` 防越界（seq02 早期帧只有 2 个目标时曾触发 latent bug） |
| 式(27) 边特征 | `graph/build.py::_edge_feature_matrix`（`build.py:70`） | 10 维：`0-2=位移 / 3=位移模长 / 4=尺寸比 / 5=cost / 6=运动先验残差 / 7=log质量 / 8=rnorm / 9=is_argmax`；再加 1 位 `intra` 标志 → 模型侧 11 维 |
| 式(28) 帧内边 | `graph/build.py`（`intra_edges` 段，`build.py:227`） | 每帧 kNN 邻接，提供局部拓扑上下文 |
| 式(29) 边标签 | `graph/build.py`（标签段，`build.py:196-211`） | 三分类 `0=无关联 / 1=移动 / 2=分裂`；**任一端 `gt_label==0`（假阳性）一律保持 NONE**，这是 C5.0b 标签污染 bug 的修复点 |
| 多步时间上下文 | `graph/build.py`（`window` 段，`build.py:238-270`） | `window>0` 时纳入前后各若干帧的边（不参与损失，只提供多步信息） |
| 图数据集落盘 | `graph/build.py::build_dataset`（`build.py:299`）、`pipeline/runner.py::_dump_graphs`（`runner.py:246`） | 前者从 h5 独立建图；后者把**本次运行实际用的耦合**导出，保证「训练用的图」与「评测用的耦合」同源 |
| 早期独立实现（保留对照） | `track/ot_tracker.py::run_tracking_ot`（`ot_tracker.py:52`） | 早于 pipeline 的独立 OT 追踪器，用于 P2 与消融对照，**不再走主链路** |

### S6 §2.0.1 决策：GNN 残差校正（论文口径）或 OT 规则（消融对照）

| 公式 | 代码位置 | 作用 |
| --- | --- | --- |
| 式(30) 边更新 `e'=φ_e(e,h_u,h_v)` | `gnn/model.py::EdgeGNN.forward`（`model.py:59`） | 边 MLP + 残差连接 |
| 式(31) 消息聚合 `m_v=Σ e'` | 同上（`agg.index_add_`） | 目标节点聚合入边 |
| 式(32) 节点更新 `h'=φ_h(h,m_v)` | 同上 | 节点 MLP + 残差连接 |
| 式(33) 边分类头 `ŷ=σ(ψ(e^L))` | `gnn/model.py::EdgeGNN`（`edge_head`） | 输出 3 类 logits；推理时 `softmax` → `prob[:,1]`=移动、`prob[:,2]`=分裂 |
| 式(34) 边级交叉熵 | `gnn/train.py::train`（`train.py:95`） | 原文二分类；我们扩展为**三分类**以显式区分移动/分裂（对应 AOGM 的 EC），属有依据的偏离 |
| 式(35) `L_OT-reg=Σ ŷ_e·C_e` | `gnn/train.py::train`（`train.py:105-113`） | 取 `cand_feat[:,5]`（cost），按平均代价归一；鼓励高置信边同时低 OT 代价 |
| 类别不平衡处理 | `gnn/train.py::_class_weights`（`train.py:42`） | 按类频倒数加权交叉熵（分裂样本极少） |
| 训练评估 | `gnn/train.py::evaluate`（`train.py:140`） | `move_f1` / `div_f1` / `exist_f1` / `sem_acc` |
| 训练数据封装 | `gnn/data.py::PairDataset`、`collate`（`data.py:15,52`） | 图数据集 → 批次（节点/边偏移）；训练入口 `scripts/run_gnn.py train`，本地 CPU 约 88 秒 / 60 epoch |
| 内存推理 | `gnn/infer.py::predict_from_couplings`（`infer.py:81`） | 直接用上游耦合构图推理，不经过磁盘往返 |
| 轨迹重建 | `gnn/infer.py::reconstruct_tracks`（`infer.py:152`） | 分裂优先（全局降序、冲突时在未占用目标里选 `max_children` 个）、再处理移动、其余为新生；父轨迹分裂后必须终止 |
| 适配层 | `gnn/adapter.py::make_gnn_runner`（`adapter.py:43`）、`graph_cfg_from_pipeline`（`adapter.py:25`） | 生成 `(dets, couplings, cfg) -> (TrackResult, graph_dir)` 可调用对象，配置从 pipeline 单一来源映射 |
| OT 规则重建（对照） | `pipeline/runner.py::ot_rule_reconstruct`（`runner.py:56`） | 式(23)(24)：`argmax_j Γ_ij` + `θ_Γ/θ_C` 阈值；分裂判据 =「一行中 ≥2 个质量占比 ≥ `div_ratio` 的目标」 |
| 消融融合公式 🔧 | `gnn/fusion.py::fused_existence` / `move_division_scores`（`fusion.py:28,41`） | `A=(1−λ)·rnorm + λ·P_model`。**原文没有这个公式**，已降级为 `--fusion` 消融开关，默认关闭 |

### S7 §1.6 第二层 tracklet OT（时间上的 coarse-graining）

| 公式 / 机制 | 代码位置 | 作用 |
| --- | --- | --- |
| 超级节点特征 | `track/tracklets.py::tracklet_stats`（`tracklets.py:23`） | 每个 tracklet 的：出现帧、首末位置、平均速度、时长 |
| tracklet 级测度 | `pipeline/tracklet_stage.py::link_tracklets`（`tracklet_stage.py:64`） | 质量 ∝ 时长（越长越可信，在 OT 里话语权越大） |
| tracklet 级代价（式20-22 思想） | 同上（`tracklet_stage.py:68-98`） | `‖末(A)−首(B)‖ + αv·‖匀速外推(A)−首(B)‖`，门限 `R_max × gap`；**不可行配对必须置 `np.inf`**（用大常数会让平衡 OT 为凑质量把不可能边算进解） |
| 第二层 OT | 同上（`tracklet_stage.py:100-109`） | **非平衡** OT（`tau=cfg.tau`）：找不到后继 → 质量流失（轨迹终止），正是式(14) 的语义 |
| 合并与规范化 | 同上（`tracklet_stage.py:111-181`） | 行内 argmax + `θ_link` 质量门限 → 链式合并 → `finalize_tracks` |

> ⚠️ **CTC 格式约束**：轨迹必须在 `[begin, end]` 内**每帧都出现**，因此 `gap>1` 的跨空洞合并天然非法
> （实测 19 个连接里 16 个被拆回）。要真正兑现论文的遮挡恢复，必须先做「空洞帧补检测」（C7）。

### S8 §1.6 轨迹重建与 CTC 格式校验（送官方评测的最后一道闸）

| 公式 / 机制 | 代码位置 | 作用 |
| --- | --- | --- |
| 式(23) `j* = argmax_j Γ_ij` | `track/ot_tracker.py::run_tracking_ot`、`pipeline/runner.py::ot_rule_reconstruct`、`gnn/infer.py::reconstruct_tracks` | 软计划 → 硬后继 |
| 式(24) `Γ≥θ_Γ 且 C≤θ_C` | 同上 | 避免噪声质量产生伪匹配 |
| 出生/死亡判定 | 同上（行和 / 列和不足 → 轨迹终止 / 起点） | 对应非平衡 OT 的「质量流失」 |
| 分裂判定 | `pipeline/runner.py::ot_rule_reconstruct`（`div_ratio`）、`gnn/infer.py::reconstruct_tracks` | 原文用「`s_j1+s_j2 ≈ s_i` 体积守恒」——**在 CTC marker 上不成立**，改用行内质量分配判据（决策记录 0001） |
| 格式校验 🔧 | `track/base.py::finalize_tracks`（`base.py:250`） | ①丢弃幽灵轨迹；②起止帧以实际赋值为准；③打断不连续段；④子轨迹起点必须 = 父终点+1；⑤一父最多 2 子 |
| 经典基线（对照） | `track/base.py::run_tracking`（`base.py:151`）、`_match_hungarian`（`base.py:142`） | 贪心 / 匈牙利 + 可选匀速先验 + 二分裂判定 |
| 结果画回体数据 | `track/base.py::paint_result`（`base.py:235`） | 用 LUT 做标签映射，输出 `uint16` 避免 3D 大体积的 int64 内存开销 |

### S9 评测层

| 模块 | 代码位置 | 作用 |
| --- | --- | --- |
| 提交写盘 | `eval/ctc_io.py::ResultWriter`、`write_result`（`ctc_io.py:23,44`） | 逐帧流式写 `mask*.tif` + `res_track.txt`（关键：禁止整卷载入） |
| 本地 SEG | `eval/local_metrics.py::seg_measure`（`local_metrics.py:66`） | 与官方**逐位一致**（E0.3 自检：0.232874 / 0.443686 精确匹配）；支持 3D 逐切片金标准 |
| 追踪诊断量 | `eval/local_metrics.py::StreamingDiagnostics`（`local_metrics.py:201`）、`tracking_diagnostics`（`:138`） | FN/FP、ID switch、碎片化、分裂 P/R；仅用于**方向判断**（R4） |
| 官方指标 | `scripts/cloud_eval.py` | 上传结果 → 云端官方二进制跑 DET/SEG/TRA → 回传指标与原始日志；默认清理临时目录 |
| AOGM 分解 | `scripts/analyze_tra_log.py` | 解析 `TRA_log.txt` → `NS/FN/FP/ED/EA/EC` 六项（权重 5/10/1/1/1.5/1） |
| 误差归属工具 | `scripts/detection_ceiling.py`（U0–U3 检测层天花板）、`scripts/pipeline_funnel.py`（S0–S3 全链路漏斗） | **本项目最重要的两个诊断工具**：三个隐性 bug（间距、标签污染、标定口径）都是它们逼出来的 |
| 汇总与图 | `scripts/make_comparison.py`、`scripts/plot_official_tiers.py`、`scripts/plot_c1c2_evidence.py`、`viz/ot_plots.py` | 跨实验对比表 + 论文级图（耦合矩阵热图、质量流） |

---

## 4. 配置 ←→ 公式 ←→ 开关 对照

配置定义全部在 `src/celltracker/pipeline/config.py`，默认值镜像 `configs/pipeline_default.yaml`。

| 配置字段 | 论文 | 默认 | 含义 / 取值建议 |
| --- | --- | --- | --- |
| `measure.mass_mode` | 式(1) | `uniform` | `uniform`（依据决策 0001）或 `volume`（接真实分割时对照） |
| `measure.knn_k` | 式(5) | `6` | 帧内 kNN 度数 |
| `measure.sigma_x/sigma_f` | 式(6) | `None` | `None` = 按中位距离自适应 |
| `ot.alpha` | 式(8) | `1.0` | α：位移代价权重 |
| `ot.beta` / `ot.sigma_s` | 式(8) | `0.0` / `1.0` | β：尺寸变化代价（默认关闭） |
| `ot.r_max` | 式(8) | `30.0` | `R_max`：候选位移上限（体素），超出置 `+inf` |
| `ot.alpha_pred` | 式(22) | `0.0` | α′>0 触发**两遍式**运动先验 |
| `ot.eta` | 式(9) | `0.0` | η：FGW 结构项权重（0 = 纯特征 OT） |
| `ot.eps` / `ot.eps_rel` | 式(11) | `1.0` / `0.1` | 后者优先：`ε = 0.1 × median(C)`，避免量纲失配 |
| `ot.tau_a` / `ot.tau_b` | 式(14) | `None` | `None` = 平衡 OT（硬边缘）；给数值 = 非平衡 |
| `multiscale.enabled` | 式(17-19) | `false` | 多尺度精炼开关 |
| `multiscale.ks` / `lambda_temp` / `n_rounds` | 式(19) | `(2,3,5)` / `0.1` / `2` | 时间尺度集合、正则权重（建议 `[0.01, 0.2]`）、交替优化轮数 |
| `tracklet.enabled` | §1.6 | `false` | 第二层 tracklet OT |
| `tracklet.max_gap` / `theta_link` / `tau` | §1.6 | `3` / `0.2` / `0.5` | 最大间隔 / 接受门限 / 第二层 KL（非平衡） |
| `graph.cand_from_ot` | 式(26) | `true` | `false` = 退化为纯 `R_max` 几何门控（消融对照） |
| `graph.theta_gamma` / `theta_c` | 式(26) | `0.02` / `None` | 质量下限 / 代价上限（`None` → `r_max²`） |
| `graph.cand_topk` 🔧 | 式(26) 补充 | `3` | 每行质量前 k 保底 |
| `graph.window` | §2.0.1 多步 | `0` | 时间上下文窗口（0 = 仅相邻两帧） |
| `graph.intra_knn` | 式(28) | `4` | 帧内 kNN 边 |
| `reconstruct.tau_move/tau_div` | 式(33) | `0.5` / `0.5` | GNN 概率阈值 |
| `reconstruct.max_children` | §1.6 | `2` | 一父最多子数 |
| `reconstruct.div_ratio` | §1.6 | `0.2` | **仅 OT 规则路径使用**；与 ε 强耦合（GNN 路径不用） |
| `detection_source` | — | `gt_tra` | `gt_tra`（分割无关上界）或 `nnunet`（Phase C 真实前端） |
| `seed` | — | `20260919` | 随机种子（写进实验留痕） |

### 消融开关总表（`config.py::ABLATIONS`，`config.py:142`）

| 开关 | 改什么 | 论文口径默认 | 正确测法（R12） |
| --- | --- | --- | --- |
| `fgw` | `ot.eta → 0` | **默认关闭（η=0）** | 应测「打开 η>0」，而不是「关掉」 |
| `unbalanced` | `ot.tau_a/tau_b → None` | **默认关闭（平衡）** | 应测「打开非平衡」 |
| `motion` | `ot.alpha_pred → 0` | **默认关闭** | 应测「打开 α′>0」 |
| `multiscale` | `multiscale.enabled → False` | **默认关闭** | 应测「打开」 |
| `tracklet` | `tracklet.enabled → False` | **默认关闭** | 应测「打开」 |
| `ot_cand` | `graph.cand_from_ot → False` | **默认开启** | 测「关掉」 |
| `cand_topk` | `graph.cand_topk → 0` | **默认开启（k=3）** | 测「关掉」 |
| `gnn` | 走 OT 规则重建 | **默认开启** | 测「关掉」 |

> 每项消融都必须**独立重训 GNN**：本 pipeline 里几乎每个模块都会改变 GNN 的输入分布
> （候选边集合或边特征），沿用原模型测到的是「分布漂移」而不是模块贡献
> （`scripts/run_ablation_matrix.py` 已把「重建图 → 重训 → 官方评测」自动化）。

---

## 5. 论文之外的工程补充与已知偏离（写论文必须标注）

| 类型 | 内容 | 代码位置 | 依据 |
| --- | --- | --- | --- |
| 🔧 工程补充 | 式(26) 的 **top-k 保底** | `graph/build.py`（`cand_topk`） | 少数真实分裂边的 OT 传输质量为 0，会被式(26) 不可逆滤掉 |
| 🔧 工程补充 | ε 按 `0.1 × median(C)` **自适应** | `pipeline/ot_stage.py`、`cost/features.py` | ε=1 配平方距离代价会退化成硬分配 |
| 🔧 工程补充 | CTC 格式校验器 `finalize_tracks` | `track/base.py:250` | 官方 TRAMeasure 会拒评非法轨迹 |
| ⚠️ 有依据偏离 | 边判定由**二分类扩展为三分类**（无关联/移动/分裂） | `graph/build.py::LABEL_NONE/MOVE/DIV`、`gnn/model.py` | 原文式(29)(34) 是二分类；三分类才能显式区分 AOGM 的 EC |
| ⚠️ 有依据偏离 | 质量取**均匀**而非「∝ 体积」 | `cost/features.py::masses` | 决策记录 0001：CTC marker 是等体积标记点（父子体积比恒 2.00） |
| ⚠️ 有依据偏离 | 分裂判据改用**行内质量分配** | `pipeline/runner.py::ot_rule_reconstruct` | 原文 §1.6 的体积守恒判据在 marker 上不成立 |
| ⚠️ 待清理 | 自创 λ 融合公式 `A=(1−λ)Γ+λ·P_model` | `gnn/fusion.py`、`gnn/infer.py`（`fusion=True`） | 原文没有该公式，已降级为消融开关，默认关闭 |
| ⚠️ 未实现 | §2.0.1 的跨帧桥接边 `(i,t−1)→(k,t+1)` | — | 论文的漏检/遮挡桥接机制，当前无实现（C7） |
| ⚠️ 未实现 | §2.0.1 的四类分割误差鲁棒性实验 | — | FN/FP/过分割/欠分割（C6） |
| ⚠️ 未实现 | 式(25) 的 `f` 接 nnU-Net encoder 特征 | `graph/build.py::_node_features`（当前为手工强度统计） | 论文明确扩展点，属 Phase D 备选 |

---

## 6. 入口脚本速查

| 脚本 | 用途 | 关键参数 |
| --- | --- | --- |
| `scripts/run_pipeline.py` | **统一 pipeline 入口**（论文顺序 + 消融开关） | `--h5 --exp-id --config --set --ablate --frames --dump-graphs --gt-h5` |
| `scripts/eval_pipeline.py` | Phase B 主力：跑链路 + 本地指标 + 可选官方指标 | `--ckpt --official --gt-h5 --cloud-gt-root` |
| `scripts/run_gnn.py` | GNN 一条龙：`build` / `train` / `infer` | `--graphs --epochs --ckpt --tau-move --tau-div` |
| `scripts/run_ablation_matrix.py` | 消融矩阵：重建图 → 重训 → 官方评测 → 成表 | `--ablations --experiments --official --skip-existing` |
| `scripts/ablate_ot_prior.py` | A2：η/τ/α′ 对 **OT 先验判别力**（AUC / argmax 命中率）的影响 | `--configs --tau --alpha-pred --frames` |
| `scripts/detection_ceiling.py` | **检测层天花板 U0–U3**：追踪之前丢了多少 | `--pred-h5 --gt-h5 --r-max` |
| `scripts/pipeline_funnel.py` | **全链路误差漏斗 S0–S3**：误差在哪一环进入 | `--graphs --ckpt --lam --fusion` |
| `scripts/predict_to_h5.py` | C3：nnU-Net 掩码 → 预测检测 h5（可选 oracle 种子） | `--pred-dir --h-frac --spacing-zyx --oracle-markers` |
| `scripts/calibrate_instance_split.py` | C2：用 GT 标记标定实例拆分参数 | `--spacing-zyx --gaussian-sigma` |
| `scripts/check_candidate_recall.py` | 式(26) 的候选筛选是否损伤召回 | `--theta-gamma --cand-topk` |
| `scripts/run_baseline.py` / `scripts/run_ot.py` | P1 经典基线 / P2 纯 OT 追踪 | `--method --eps --eta --tau` |
| `scripts/run_sweep.py` / `scripts/sweep_gnn_thresholds.py` | 参数网格扫描（表格 + 热图） | `--x --y` |
| `scripts/analyze_tra_log.py` | 官方 TRA 日志 → AOGM 六项分解 | 日志路径 |
| `scripts/cloud_eval.py` / `scripts/cloud_run.py` / `scripts/cloud_run.sh` | 云端官方评测 / 远程执行 / 文件传输 | `--res-dir --dataset --seq` |
| `scripts/nnunet/build_dataset.py` | CTC → nnU-Net 数据集（训练标签用**银标准 `_ST/SEG`**） | `--src --out` |
| `scripts/nnunet/make_infer_input.py` | 待测帧 → `*_0000.nii.gz`（间距必须与训练一致） | `--src --out` |
| `scripts/nnunet/cloud_train.sh` | 云端：解压 → 转换 → plan/preprocess → 训练 | `stage = all / data / preprocess / train` |
| `src/celltracker/experiment/runner.py::Experiment` | 实验八件套留痕（config/命令/env/git/指标/日志/图/结论） | — |

---

## 7. 单测覆盖对照（82 项全绿）

| 测试文件 | 覆盖的模块 | 守护的关键行为 |
| --- | --- | --- |
| `tests/test_ctc.py` | `data/ctc.py` | CTC 命名约定、3D 逐切片金标准、`L B E P` 解析 |
| `tests/test_ot.py` | `ot/sinkhorn.py` | 平衡/非平衡解，与 LP 对照 |
| `tests/test_fgw.py` | `ot/fgw.py` | 结构项与梯度（有限差分校验）、目标单调下降 |
| `tests/test_multiscale.py` | `ot/multiscale.py` | 式(19) 正则定义 |
| `tests/test_multiscale_stage.py` | `pipeline/multiscale_stage.py` | `enabled=False` 恒等映射、精炼不增正则 |
| `tests/test_motion.py` | `pipeline/motion.py` | 式(20)(21) 速度正确性、`valid` 掩码、**死特征检查**、两遍式等价性 |
| `tests/test_instances.py` | `detect/instances.py` | 物理间距 EDT、seeds 路径、标签面积判据 |
| `tests/test_tracklets.py` | `track/tracklets.py` | 超级节点汇总特征 |
| `tests/test_tracklet_stage.py` | `pipeline/tracklet_stage.py` | 非平衡第二层、链式合并、格式合法 |
| `tests/test_tracking_consistency.py` | `track/base.py` | **幽灵轨迹回归**、父子关系、`finalize_tracks` |
| `tests/test_pipeline_config.py` | `pipeline/config.py` | 消融开关语义、未知开关报错、yaml 往返 |
| `tests/test_pipeline_runner.py` | `pipeline/runner.py` | 七阶段串联、`--ablate` 切换、`CouplingArtifacts` 一致性 |
| `tests/test_gnn_adapter.py` | `gnn/adapter.py` | 配置单一来源、GNN 可插入 runner |
| `tests/test_decision_and_ablation.py` | `gnn/infer.py::reconstruct_tracks` | 分裂/移动冲突消解、父轨迹终止 |
| `tests/test_fusion.py` | `gnn/fusion.py` | 融合分数的「存在性 × 语义」分解 |
| `tests/test_detection_ceiling.py` | `scripts/detection_ceiling.py` | U0–U3 口径（U2 已改为「位移在 `R_max` 内」） |

---

## 8. 已知代码 / 文档不一致（待清理）

| # | 问题 | 影响 | 建议 |
| --- | --- | --- | --- |
| 1 | `graph/build.py::GraphConfig.node_feat_dim=8` / `edge_feat_dim=10` 与实际不符（实际节点 7 维、边 11 维） | 这两个字段**未被任何代码读取**，但会误导读者，且与 `ModelConfig.node_dim=7/edge_dim=11` 矛盾 | 删除这两个字段，或改为从 `ModelConfig` 派生 |
| 2 | `pipeline/multiscale_stage.py` 返回的 `info` 里 `lambda_temp` 键重复出现两次 | 无功能影响 | 清理 |
| 3 | `ot/multiscale.py::temporal_regularizer` 与 `multiscale_stage.py::_reg_value` 是同一公式的两处实现 | 存在「一处改了另一处忘了」的口径漂移风险（历史上已因重复实现吃过亏） | 让阶段实现调用基元 |
| 4 | 121 个实验目录中有 78 个缺 `notes.md`（含 B2 系列、E4.5–E4.10、C4O 等） | 留痕不齐（E9 要求八件套） | 按 C0 的方式补目录与 INDEX 行 |
| 5 | 本地 `detection_precision` 在「结果对象少于 GT 对象」时会 >1 | 度量不可用（曾算出 2.415） | 修正定义或直接弃用该量 |

---

## 附：一句话回顾每个 stage 的「为什么」

1. **S0–S1 把图像变成测度**：细胞群体 → 带质量的点集 + 帧内几何结构，是后面一切公式的输入。
2. **S2 用 OT 把「点对点匹配」松弛成「质量流」**：软耦合同时给出主路径与备选路径，并天然表达出生/死亡/分裂。
3. **S3–S4 把「逐帧独立」提升到「时间上一致」**：运动先验给惯性，多尺度正则约束多步路径的统计行为。
4. **S5–S6 把「局部最优」交给「全局上下文」修正**：OT 只负责筛候选与提供特征，GNN 用多步时空上下文做残差校正——
   实验证明这是**唯一真正必需**的模块（关掉 −0.0034，4.9× 噪声地板）。
5. **S7–S8 把「软计划」变成「合法轨迹」**：tracklet 级 coarse-graining 管长程与碎片化，格式校验器保证送得出官方评测。
6. **S9 用误差归属而不是调参来定位问题**：检测层天花板、全链路漏斗、Oracle 上界——三个工具逼出了三个隐性 bug，
   并把端到端瓶颈定量归属到「前端实例归属」（上限 +0.079 TRA）而不是追踪算法本身。
