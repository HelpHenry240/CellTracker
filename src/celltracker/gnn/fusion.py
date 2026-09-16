"""OT 先验与 GNN 输出的融合（§2.0.1 "残差校正"）——训练与推理共用同一公式。

设计要点：
  * **存在性**：`A = (1-λ)·Γ_先验 + λ·P_model(有关联)`
    - λ=0 时 A 就是 OT 的行归一化传输质量（§1.6 的软匹配先验）；
    - λ=1 时完全由模型决定；
    - 中间值即"模型对 OT 先验的残差修正"。
    训练时用 A 与 GT 计算 BCE，因此模型学的正是"把 OT 判错的边修回来"，
    而不是从零学一个独立的分类器。

  * **语义**（移动 vs 分裂）：传输计划 Γ 只表达"有多少质量流向对方"，
    不携带"移动/分裂"的语义，因此语义由模型的条件概率 `p_div/(p_move+p_div)`
    决定；这一项在训练与推理中含义一致。

把公式集中在 `src/celltracker/gnn/fusion.py`，训练与推理都从这里取，
从代码结构上杜绝"训练/推理口径不一致"。
"""

from __future__ import annotations

import numpy as np

__all__ = ["fused_existence", "conditional_division_prob", "move_division_scores"]

EPS = 1e-12


def fused_existence(rnorm, p_related, lam: float):
    """融合后的"该边存在"分数 A = (1-λ)·rnorm + λ·p_related。"""
    return (1.0 - lam) * rnorm + lam * p_related


def conditional_division_prob(p_move, p_div):
    """条件语义概率 P(分裂 | 有关联)；np 或 torch 张量均可。"""
    tot = p_move + p_div
    if hasattr(tot, "clamp"):        # torch
        return p_div / tot.clamp(min=EPS)
    return p_div / np.maximum(tot, EPS)


def move_division_scores(rnorm, p_move, p_div, lam: float):
    """把"存在性 × 语义"分解成移动/分裂两类概率。

    `P(div) = A · P(div | related)`，`P(move) = A · (1 − P(div | related))`。

    为什么必须乘 A：`P(div|related)` 只在"已经有关联"时才有意义。对无关边
    （A≈0、p_move+p_div 极小），比值是噪声——实测 `none` 边的条件概率均值高达
    0.92，若不乘 A 就会有大量无关边混进分裂候选，污染 top-2 排序。
    """
    a = fused_existence(rnorm, p_move + p_div, lam)
    sem = conditional_division_prob(p_move, p_div)
    return a * (1.0 - sem), a * sem
