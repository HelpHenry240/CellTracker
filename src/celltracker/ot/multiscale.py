"""多尺度时间一致性的**基元**（ideas.pdf 式 17–19）。

基础目标（逐对独立求解）::

    sum_t [ L(P_t) + eps Ent(P_t) + tau_a KL(P_t 1 | a_t) + tau_b KL(P_t^T 1 | a_{t+1}) ]

加上多尺度时间正则::

    R^(k)(P_t..P_{t+k-1}) = || D_{t,t+k} - P_t P_{t+1} ... P_{t+k-1} ||_F^2

其中 `D_{t,t+k}` 是帧 t 与 t+k 之间**直接**求解（可在空间上粗化/下采样）的 OT 耦合。
本模块只提供两个基元：
  - `direct_jump_coupling`：跳帧直接 OT（式19 的 `D_{t,t+k}`）
  - `temporal_regularizer`：多步复合与直接耦合的不一致度（式19）

**阶段实现**（交替优化、接入 pipeline 的 CouplingArtifacts 接口）在
`pipeline/multiscale_stage.py`；此处不再保留第二份精炼实现，
避免"一个模块两套代码"导致口径漂移。
"""

from __future__ import annotations

import numpy as np

from ..cost.features import CostConfig, build_cost, masses
from .sinkhorn import sinkhorn_log

__all__ = ["direct_jump_coupling", "temporal_regularizer"]


def direct_jump_coupling(src_xy: np.ndarray, dst_xy: np.ndarray, gap: int,
                         ot_cfg, eps: float | None = None,
                         tau_a: float | None = None,
                         tau_b: float | None = None) -> np.ndarray:
    """式(19) 的 `D_{t,t+k}`：帧 t 与 t+gap 之间的**直接** OT 耦合。

    R_max 按 gap 线性放宽（`r_max × gap`），与式(19)"跨 k 帧的直接耦合"一致。
    统一走 `pipeline.ot_stage.compute_pairwise_plan`，保证与相邻帧 OT 同口径。
    """
    from ..pipeline.config import MeasureConfig, OTConfig
    from ..pipeline.ot_stage import compute_pairwise_plan

    jump_cfg = OTConfig(alpha=ot_cfg.alpha, beta=ot_cfg.beta,
                        sigma_s=ot_cfg.sigma_s, r_max=ot_cfg.r_max * gap,
                        alpha_pred=0.0, eta=0.0,
                        eps=(eps if eps is not None else ot_cfg.eps),
                        eps_rel=(None if eps is not None else ot_cfg.eps_rel),
                        tau_a=tau_a, tau_b=tau_b)
    art = compute_pairwise_plan(src_xy, dst_xy, jump_cfg,
                                measure=MeasureConfig(mass_mode="uniform"))
    return art.plan


def temporal_regularizer(P_list: list[np.ndarray], t: int, k: int,
                         D_direct: np.ndarray) -> float:
    """式 (19)：`R^(k) = ||D_{t,t+k} - P_t···P_{t+k-1}||_F^2`。"""
    if k < 1:
        raise ValueError("k 必须 >= 1")
    prod = P_list[t]
    for i in range(1, k):
        prod = prod @ P_list[t + i]
    return float(np.sum((D_direct - prod) ** 2))

