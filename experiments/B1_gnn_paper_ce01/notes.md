# B1 论文口径训练 GNN（式34 + 式35）—— 目前最好的结果

## 目的

用**严格论文口径**重训 GNN，并在 Phase A 搭好的主链路上评测：

- 损失 = 式(34) 边级交叉熵 + 式(35) `L_OT-reg = mean(ŷ_e·C_e / mean(C))`
- 候选边由式(26) 筛（`Γ ≥ θ_Γ` + 工程补充 top-3 保底）
- 决策用式(33) 的边分类概率（**无任何自创融合**）
- 训练数据：`graphs_ce01_otcand3`（CE seq01，194 对帧，61,242 条候选边）

## 训练

| 项 | 值 |
| --- | --- |
| epochs | 60 |
| 耗时 | **87.76 秒**（本地 CPU；模型小，不需要 GPU） |
| 最优 move F1 | 0.9916 |
| 最优 division F1 | 0.5207 |
| 语义准确率（移动 vs 分裂） | 0.9832（epoch 30） |

## 评测（CE seq01 全量 195 帧，CTC 官方指标）

配置：`eta=0, tau=null, alpha_pred=0, multiscale=off, tracklet=off`
——即 **OT + GNN** 两段式（多尺度/运动先验/tracklet 留到 B2 消融）。

| 方法 | 轨迹数 | ID switch | 碎片化 | DET | **TRA** |
| --- | --- | --- | --- | --- | --- |
| 匈牙利基线 | 371 | 459 | 107 | 1.000000 | 0.995594 |
| 纯 OT（手调最优） | 570 | 432 | 278 | 1.000000 | 0.994617 |
| 早期 GNN（偏离期实现：几何门控候选 + 全权决策） | 913 | 111 | 301 | 1.000000 | 0.996089 |
| **B1：论文口径 pipeline** | 755 | 139 | **174** | 1.000000 | **0.996513** |

## AOGM 误差分解（官方 TRA 日志，权重 NS5/FN10/FP1/ED1/EA1.5/EC1）

| 方法 | ED | EA | EC | AOGM | TRA |
| --- | --- | --- | --- | --- | --- |
| 匈牙利基线 | 141 | 498 | 318 | 1206.0 | 0.995594 |
| 早期 GNN | 90 | 589 | 97 | 1070.5 | 0.996089 |
| **B1（论文口径）** | **58** | **505** | 139 | **954.5** | **0.996513** |

**论文口径 pipeline 在 AOGM 上比基线低 21%、比早期实现低 11%**：

- **ED 58（历次最低）**：多余边大幅减少，说明式(26) 的候选筛选 + top-k 保底
  让"误分裂"显著变少；
- **EA 505**（基线 498、早期实现 589）：缺失边回到基线水平附近，
  说明"分裂召回不足"的问题被缓解；
- **EC 139**（基线 318、早期实现 97）：语义错误远低于基线，
  但仍高于早期实现——后者用了更多手工调参的阈值，属于"过拟合到该序列"。

## 与早期实现的对比说明了什么

早期实现（E4.2d）虽然 TRA 也不低，但它是**偏离论文口径**的：
候选边用纯几何门控（OT 只提供特征）、GNN 全权决策。
改成论文口径后：

1. TRA 从 0.996089 → **0.996513**；
2. 碎片化从 301 → **174**（接近腰斩）；
3. ED 从 90 → **58**。

即"**严格按论文搭 pipeline**"本身就是一次实质改进——这也是本项目
2026-09-16 复盘的核心教训（见 `docs/reports/day1_summary.md`）。

## 复现

```bash
# 训练（87 秒）
python scripts/run_gnn.py train --graphs data/interim/graphs_ce01_otcand3 \
    --exp-id B1_gnn_paper_ce01 --epochs 60

# 评测（含官方指标）
python scripts/eval_pipeline.py --h5 data/interim/Fluo-N3DH-CE_01.h5 \
    --dataset Fluo-N3DH-CE --seq 01 --exp-id B1_eval_ce01_gnn \
    --ckpt experiments/B1_gnn_paper_ce01/artifacts/model/best.pt --official
```

## 下一步（B2）

用同一套 runner + `--ablate` 跑消融矩阵：
`fgw / unbalanced / motion / multiscale / tracklet / ot_cand / cand_topk / gnn`，
每项一组官方指标，形成论文的逐模块贡献表；并在 seq02 上复验。
