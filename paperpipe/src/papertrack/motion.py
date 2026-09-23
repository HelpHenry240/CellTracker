"""§1.5 运动先验与速度约束（ideas.pdf 式 20–22）——两遍式。

原文（R1 抄录）
--------------
式(20)  v^t_i ≈ x^t_i − x^{t−1}_{p(i)}
        "假设已经通过**上一轮追踪**在帧 t−1 与帧 t 之间建立了硬关联 p(i)"
式(21)  x̂^{t+1}_i = x^t_i + v^t_i
式(22)  C̃_ij = α‖x_i^t−x_j^{t+1}‖² + α′‖x̂_i^{t+1}−x_j^{t+1}‖² + β((s_i−s_j)/σ_s)²

复用说明
--------
式(20) 的实现（"用追踪结果的硬关联估速"）与原文完全一致，
**直接复用** `celltracker.pipeline.motion.{estimate_velocity, attach_velocity,
velocity_feature_is_live}`（原仓库少数严格按论文口径重建的模块）。
本文件只补两遍式的**驱动顺序**（第 1 遍 α′=0 → 估速 → 第 2 遍 α′>0）。
"""

from __future__ import annotations

import numpy as np

from celltracker.pipeline.motion import (attach_velocity,  # noqa: F401  式(20) 复用
                                        estimate_velocity,
                                        velocity_feature_is_live)
from celltracker.track.base import Detections, TrackResult

__all__ = ["estimate_velocity", "attach_velocity", "velocity_feature_is_live",
           "velocity_report"]


def velocity_report(dets: Detections) -> dict:
    """验收检查（防止"速度特征恒为 0"的死特征，AGENTS.md 要求）。"""
    mags = []
    for t in dets.t_range:
        v = dets.frames[t].get("velocity")
        if v is not None and np.asarray(v).size:
            mags.append(np.linalg.norm(np.asarray(v, dtype=float), axis=1))
    if not mags:
        return {"live": False, "n": 0}
    cat = np.concatenate(mags)
    nz = cat[cat > 0]
    return {"live": bool(nz.size > 0), "n": int(cat.size),
            "n_nonzero": int(nz.size),
            "p50_um": float(np.median(nz)) if nz.size else 0.0,
            "p99_um": float(np.percentile(nz, 99)) if nz.size else 0.0}


def estimate_velocity_from_result(dets: Detections, result: TrackResult):
    """式(20)：由"上一轮追踪"的硬关联 p(i) 估计速度。"""
    return estimate_velocity(dets, result)
