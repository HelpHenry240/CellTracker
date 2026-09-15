# E2.1 纯 OT 追踪（η=0，平衡熵正则）

## 目的

验证 P2 求解器与"由传输计划重建轨迹"的完整实现：`C_feat` + 熵正则 Sinkhorn +
θ_Γ/θ_C 阈值 + 分裂判定。

## 做了什么

- `src/celltracker/cost/features.py`：式 (8)(22) 代价矩阵（位置² + 尺寸² + 匀速先验），
  `R_max` 门限；式 (1) 质量向量；式 (4)–(7) 帧内 kNN 图。
- `src/celltracker/ot/sinkhorn.py`：log-domain Sinkhorn，支持平衡与**非平衡**
  （KL 松弛，式 14），`tau→∞` 时退化为平衡 OT。
- `src/celltracker/track/ot_tracker.py`：§1.6 的轨迹重建（行内 argmax + 阈值 +
  出生/死亡 + 二分裂）。
- 单测：边沿约束、τ→∞ 退化、`eps→0` 逼近 POT 精确 OT、inf 代价屏蔽、
  非平衡质量折中。

## 结果（Fluo-N3DH-CHO seq 01）

配置：`eps=1.0, eta=0, R_max=30, 质量=uniform, θ_Γ=0.2`

| 指标 | 值 |
| --- | --- |
| 官方 DET | 1.000000 |
| 官方 TRA | 0.998555 |
| 预测轨迹数 | 27（GT 27） |
| 本地 FN/FP/IDsw/碎片 | 0 / 0 / 0 / 0 |

## 结论

- OT 求解器与轨迹重建实现正确（与匈牙利基线在 CHO 上同分，符合"CHO 无区分度"的预期）。
- 所有数值单测通过（16 项），其中 FGW 结构项梯度通过有限差分校验。
- 真正的判别实验需要 CE。

## 下一步

CE 数据就绪后：E1.3（基线在 CE 上）与 E2.2/E2.3（平衡 vs 非平衡、η 扫描）。
