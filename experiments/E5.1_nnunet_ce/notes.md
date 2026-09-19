# E5.1 nnU-Net 3D 分割前端（Fluo-N3DH-CE）—— **训练完成**

## 训练完成（2026-09-19 20:12）

| 项 | 值 |
| --- | --- |
| 完成 epoch | **1000 / 1000**（按 nnU-Net 默认协议） |
| **最优 EMA pseudo Dice** | **0.9605**（验证集 77 例） |
| 训练总时长 | 分两段：09-16 20:44–23:31（epoch 1–338）+ 09-19 14:31–20:12（续训 339–1000） |
| 每 epoch 耗时 | 26.5 s（RTX 4090，显存 6.7/24 GB） |
| 产物 | `checkpoint_best.pth` / `checkpoint_latest.pth` / **`checkpoint_final.pth`** |

### Dice 轨迹（说明"续训"的价值）

| 时间 | epoch | 最优 EMA Dice |
| --- | --- | --- |
| 09-16 23:31（当初误判为平台期而停止） | 338 | 0.9530 |
| 09-19 15:51 | 463 | 0.9531 |
| 09-19 16:50 | 586 | 0.9550 |
| 09-19 20:12 | 1000 | **0.9605** |

从"平台期"继续训练后又提升 **+0.0075**，说明当初的停止判断偏早
（余弦学习率衰减的后段仍有实质收益）。

### 备份

| 位置 | 内容 |
| --- | --- |
| 本地 `data/checkpoints/nnunet_ce_fold0/` | `checkpoint_best_ep338.pth`、`checkpoint_final_ep1000.pth` |
| 云端 `/root/autodl-tmp/nnunet/nnUNet_results/.../fold_0/` | `checkpoint_best.pth`、`checkpoint_latest.pth`、`checkpoint_final.pth` |
| 训练曲线 | `experiments/E5.1_nnunet_ce/figures/progress_final.png`、`progress_ep1000.png` |

**注意**：checkpoint 约 348 MB/个，不入库（走本地/云端备份）；
GNN 的 413 KB checkpoint 则已入库。

### 续训命令（若日后需要）

```bash
# 不要改 --num_epochs，否则余弦学习率调度会与 checkpoint 内的总 epoch 不一致
nnUNetv2_train 501 3d_fullres 0 --c
```

## 以下为早期记录（保留）

## 目的

为 Phase 5 的"全链路（分割 → 检测 → OT → 跟踪）"与 §2.0.1 的鲁棒性论断提供分割前端，
并产出可用于官方 SEG 指标的分割结果。

## 关键设计决策

1. **训练标签用银标准 `_ST/SEG`**（195 + 190 帧全卷分割）。
   CTC 的金标准 `_GT/SEG` 在 CE 上每个序列只有 **5 个切片**，无法训练；
   官方文档明确允许用银标准做算法微调。
2. **实例标签二值化**为语义分割（细胞=1）。nnU-Net 做语义分割，
   实例拆分留到推理后处理（连通域 / 分水岭）。
3. **显式写入体素间距** `(0.09, 0.09, 1.0) µm`。若用裸 TIFF（间距默认 1,1,1），
   nnU-Net 的重采样决策会完全错误 —— 这是 CTC 3D 数据接入 nnU-Net 最常见的坑。
4. 两类序列合并训练（385 例），5 折划分，本次训练 fold 0（308 训练 / 77 验证）。

## 训练配置（nnU-Net 自动规划）

| 项 | 值 |
| --- | --- |
| config | 3d_fullres |
| patch size | 16 × 256 × 384 |
| batch size | 2 |
| spacing | [1.0, 0.09, 0.09]（保持原始各向异性，未强制各向同性重采样） |
| 归一化 | ZScoreNormalization |
| 每 epoch 耗时 | **26.7 s**（RTX 4090，GPU 利用率 96%，显存 6.6/24 GB） |
| 预计 1000 epochs | 约 8.3 小时 |

## 训练最终结果（2026-09-16 23:31，按平台期判据停止）

| 项 | 值 |
| --- | --- |
| 停止时 epoch | **338 / 1000** |
| 最优 EMA pseudo Dice（验证集 77 例） | **0.9530** |
| 已用时间 | 2 小时 48 分（约 30 s/epoch） |
| 若跑满 1000 epoch | 还需约 5.5 小时（余弦调度剩余部分） |

Dice 轨迹：epoch 21 → 0.9289，epoch 150 → 0.9496，epoch 240 → 0.9519，
epoch 338 → 0.9530。**最近 100 个 epoch 仅提升 0.0011（每 50 epoch < 0.001）**，
触发"平台期即停"的判据（见 `docs/` 计划中的策略）。

证据与备份：
- `figures/progress_final.png`（完整训练曲线）、`logs/training_log.txt`
- 最优权重已备份到本地 `data/checkpoints/nnunet_ce_fold0/checkpoint_best_ep338.pth`
  （348MB；含网络权重、优化器状态、grad scaler、epoch 计数）
- 云上保留 `checkpoint_best.pth` / `checkpoint_latest.pth`，
  如需继续训练：`nnUNetv2_train 501 3d_fullres 0 --c`
  （**续训不要改 `--num_epochs`**，否则余弦学习率曲线会错位）

## 观察（训练早期，epoch 21）

- EMA pseudo Dice 已达 **0.9289**，且仍在上升（epoch 13 时 0.9238）。
- 说明该任务对 nnU-Net 而言相对容易（银标准标签 + 胚胎核形态一致），
  收敛可能明显早于 1000 epochs。
- 证据：`figures/progress_snapshot.png`、`logs/training_log.txt`。

## 与主线的关系

分割前端**不在追踪方法（OT + GNN）的关键路径上**：Phase 1–4 的结论全部建立在
"GT 检测"的与分割无关设定上。但 SEG 指标、全链路实验与鲁棒性论断需要它，
且它可与本地开发并行，不占用本地时间。

## 下一步

1. 监控训练曲线，记录 250 / 500 / 1000 epoch 的验证 Dice 快照。
2. 用 `nnUNetv2_predict` 对 CE 推理，接实例拆分后处理。
3. 与 `_GT/SEG` 的 5 个官方参考切片比对，得到 SEG 指标（E5.2）。
