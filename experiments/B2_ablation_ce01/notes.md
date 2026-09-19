# B2 消融矩阵（CE seq01，官方指标）—— 首批结果

## 方法学说明（重要）

这个 pipeline 里**几乎每个模块都会改变 GNN 的输入**（候选边集合或边特征），
所以严格消融需要**为每个配置重建图并重训 GNN**（约 2 分钟建图 + 1.5 分钟训练），
否则测到的是"分布漂移"而不是模块贡献。本记录区分两类：

- **纯后处理消融**（无需重训）：`gnn` 开关、`tracklet` 开关 → 本轮完成；
- **需要重训的消融**：`fgw`(η)、`unbalanced`(τ)、`motion`(α′)、`multiscale`、
  `ot_cand`(式26)、`cand_topk` → 下一步批量做。

## 首批结果（CTC 官方指标，195 帧全量）

| # | 配置 | 轨迹数 | ID switch | 碎片化 | 分裂 P | 分裂 R | **TRA** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | §1.6 OT 规则（GNN 关闭，阈值调到最优） | 388 | 517 | 293 | — | 0.140 | 0.993135 |
| 2 | OT + GNN（B1 基准） | 755 | 139 | 174 | 0.854 | 0.802 | 0.996513 |
| 3 | **OT + GNN + 第二层 tracklet OT** | 533 | 266 | **79** | 0.865 | 0.405 | **0.996622** |

### 逐项贡献

| 模块 | 贡献（ΔTRA） | 说明 |
| --- | --- | --- |
| **GNN 决策** | **+0.0034** | OT 规则最好只能到 0.993135；GNN 到 0.996513 |
| **第二层 tracklet OT** | **+0.0001** | 碎片化 174 → 79（腰斩），但 ID switch 139 → 266 |

## 关于"OT 规则"这一行的公平性（重要发现）

手工规则**找不到好工作点**。对其重建阈值 `div_ratio` 做扫描（70 帧子集）：

| div_ratio | 轨迹数 | IDsw | 碎片化 | 分裂召回 |
| --- | --- | --- | --- | --- |
| 0.3 | 1610 | 250 | 1266 | 0.332 |
| 0.5 | 388 | 517 | 293 | 0.140 |
| 0.7 | 388 | 517 | 293 | 0.140 |

**要么过分裂（0.3），要么完全测不出分裂（≥0.5）**——这是"手工阈值与 ε 强耦合"
（A2 结论）的直接后果，也说明**学习型决策不是锦上添花，而是必需的**。
因此表里的第 1 行取的是扫描中最优的 0.5（对 OT 规则最有利），
若取默认 0.2 则只有 0.979362 —— 那样对 OT 规则不公平。

## 观察：tracklet 阶段的取舍

tracklet 让碎片化腰斩（174 → 79）但 ID switch 翻倍（139 → 266），
TRA 只微增（+0.0001）。机制上可解释：

- 当前只接受 `gap==1` 的合并（CTC 格式约束，见 A5），能合并的多是"同一 GT 轨迹被切成两段"；
- 但也可能把两条本应独立的轨迹连起来 → ID switch 上升；
- 两者在 AOGM 里代价相近（ED/EA 1.0–1.5 vs EC 1.0），所以净效果很小。

**Phase B 调优方向**：`theta_link` 与第二层 OT 的 `tau` 需要扫描；
若开启空洞补检测（Phase C），遮挡恢复的收益才可能体现。

## 下一步

1. 批量做"需要重训"的六项消融（每项：改配置 → 重建图 → 重训 GNN → 官方评测）；
2. 调 `theta_link` / 第二层 `tau`；
3. 在 seq02 上复验当前最好配置。

## 复现

```bash
# 1) OT 规则（GNN 关闭，阈值最优）
python scripts/eval_pipeline.py --h5 data/interim/Fluo-N3DH-CE_01.h5 \
    --dataset Fluo-N3DH-CE --seq 01 --exp-id B2_otrule_best \
    --set reconstruct.div_ratio=0.5 --official

# 2) GNN 基准
python scripts/eval_pipeline.py ... --exp-id B1_eval_ce01_gnn \
    --ckpt experiments/B1_gnn_paper_ce01/artifacts/model/best.pt --official

# 3) GNN + tracklet（当前最好）
python scripts/eval_pipeline.py ... --exp-id B2_gnn_tracklet \
    --ckpt experiments/B1_gnn_paper_ce01/artifacts/model/best.pt \
    --set tracklet.enabled=true tracklet.max_gap=3 --official
```
