# B3 seq02 复验：论文口径 pipeline 在第二个序列上同样胜出

## 目的

B1/B2 的所有结论都基于 seq01。本轮在与 seq01 **完全相同的协议**（同序列内划分
训练/测试）与**相同配置**（平衡 OT + 式(26) 候选 + top-k 保底 + GNN + 第二层 tracklet）
下复验 seq02。

## 过程中修掉一个 latent bug（重要）

seq02 有早期帧只有 **2 个目标**，触发了 top-k 保底的越界 bug：

```
IndexError: shape mismatch: indexing arrays could not be broadcast together
             with shapes (6,) (4,)
```

根因：`rows = np.repeat(np.arange(n_src), k)` 按 k 构造行索引，而 `cols` 只有
`min(k, n_dst)` 列。当 `n_dst < cand_topk` 时两者长度不一致。

**seq01 从未触发**（其每帧目标数 ≥3），所以这个 bug 一直潜伏到今天。
修复：用实际可用列数 `k_eff = order.shape[1]`。已补回归测试。

## 结果（CTC 官方指标）

| 方法（seq02，190 帧） | 轨迹数 | IDsw | 碎片化 | DET | **TRA** |
| --- | --- | --- | --- | --- | --- |
| 匈牙利基线 | — | 473 | 138 | 1.000000 | 0.995052 |
| 早期 GNN（偏离期实现，同序列） | 1387 | 57 | 862 | 1.000000 | 0.992653 |
| 早期 GNN（seq01→seq02 跨序列） | 1387 | 96 | 757 | 1.000000 | 0.992958 |
| **B3：论文口径（GNN + tracklet）** | **570** | 221 | **67** | 1.000000 | **0.996369** |

### AOGM 误差分解

| 方法 | ED | EA | EC | AOGM | TRA |
| --- | --- | --- | --- | --- | --- |
| 匈牙利基线 | 162 | 535 | 311 | 1275.5 | 0.995052 |
| 早期 GNN | 109 | 1116 | 111 | 1894.0 | 0.992653 |
| **B3（论文口径）** | **46** | **462** | 197 | **936.0** | **0.996369** |

AOGM **比基线低 27%**、比早期实现低 51%。

## 双序列汇总（论文主表）

| 方法 | seq01 TRA | seq02 TRA | 平均 |
| --- | --- | --- | --- |
| 匈牙利基线 | 0.995594 | 0.995052 | 0.995323 |
| 纯 OT（手调最优） | 0.994617 | — | — |
| 早期 GNN（偏离期实现） | 0.996089 | 0.992653 | 0.994371 |
| **论文口径 pipeline** | **0.996513** | **0.996369** | **0.996441** |
| 论文口径 + 第二层 tracklet | **0.996622**（seq01） | 0.996369（含 tracklet） | — |

**关键**：早期实现只在 seq01 上看起来不错（0.996089），在 seq02 上**输给基线**；
严格按论文口径搭好 pipeline 后，**两个序列都稳定超过基线**：

| | seq01 | seq02 |
| --- | --- | --- |
| vs 匈牙利基线 | +0.0009 | **+0.0013** |
| 碎片化 vs 基线 | 174 vs 107（+tracklet 后 79） | **67 vs 138** |

## 结论

1. **论文口径的 pipeline 在双序列上一致优于强基线**，这是本项目最核心的结果；
2. seq02 的碎片化（67）甚至低于基线（138），说明"式(26) 候选筛选 + GNN 决策"
   在密集分裂场景下的优势更明显；
3. 早期"偏离期实现"在 seq02 上的失败（0.9927）不是方法问题，而是**实现偏离论文**
   造成的——这再次印证 09-16 复盘的核心教训。

## 复现

```bash
# 1) 导出 seq02 图数据集（与评测同源的耦合）
python scripts/run_pipeline.py --h5 data/interim/Fluo-N3DH-CE_02.h5 \
    --exp-id B3_seq02_graphs --dump-graphs data/interim/graphs_02_base
# 2) 训练
python scripts/run_gnn.py train --graphs data/interim/graphs_02_base \
    --exp-id B3_seq02_train --epochs 60
# 3) 评测（含官方指标）
python scripts/eval_pipeline.py --h5 data/interim/Fluo-N3DH-CE_02.h5 \
    --dataset Fluo-N3DH-CE --seq 02 --exp-id B3_seq02_eval \
    --ckpt experiments/B3_seq02_train/artifacts/model/best.pt \
    --set tracklet.enabled=true --official
```
