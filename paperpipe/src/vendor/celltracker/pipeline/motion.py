"""§1.5 运动先验（式 20-22）——两遍式实现。

原文：
  式(20)  v^t_i ≈ x^t_i − x^{t−1}_{p(i)}      （p(i) 为上一轮追踪建立的硬关联）
  式(21)  x̂^{t+1}_i = x^t_i + v^t_i            （匀速外推的预测位置）
  式(22)  C̃_ij = α‖x^t_i−x^{t+1}_j‖² + α′‖x̂^{t+1}_i−x^{t+1}_j‖² + β((s_i−s_j)/σ_s)²

关键点：速度**不是**凭空来的，而是用"上一轮追踪"的关联估计出来的
（原文："假设已经通过上一轮追踪在帧 t−1 与 t 之间建立了硬关联 p(i)"）。
因此需要两遍：

  第 1 遍：α′=0 跑一次追踪  →  得到硬关联 p(i)
  第 2 遍：由 p(i) 估速并外推 → 带 α′ 重跑

这与 `ot_tracker` 里"边追踪边在线估速"不同：后者在轨迹尚未稳定时估速，
本节实现的是论文口径的两遍式。
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from ..track.base import Detections, TrackResult

if TYPE_CHECKING:  # pragma: no cover
    from ..track.ot_tracker import OTTrackConfig

__all__ = ["estimate_velocity", "attach_velocity", "velocity_feature_is_live",
           "run_two_pass"]


def estimate_velocity(dets: Detections, result: TrackResult, ignore_background: bool = True
                      ) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """式(20)：用追踪结果里的硬关联估计每个检测的速度。

    编号 0 表示被排除的检测，不构成式(20) 中的硬关联。ignore_background=False
    仅用于复现旧 ideas-v2 实验的背景匹配错误，不能作为正常估速设置。

    返回 `(velocity, valid)`：
      - `velocity[t]` (n_t, d)：第 t 帧每个检测的速度；无前驱时为 0
      - `valid[t]`    (n_t,)  ：该检测是否真的有前驱（用于区分"静止"与"无前驱"）
    """
    velocity: dict[int, np.ndarray] = {}
    valid: dict[int, np.ndarray] = {}
    ts = dets.t_range
    for pos, t in enumerate(ts):
        xy = dets.centroid(t)
        vel = np.zeros_like(xy)
        ok = np.zeros(xy.shape[0], dtype=bool)
        if pos > 0 and (t - 1) in result.assignment and t in result.assignment:
            prev_xy = dets.centroid(t - 1)
            prev_ids = result.assignment[t - 1]
            lookup = {int(tid): prev_xy[k] for k, tid in enumerate(prev_ids)
                      if not ignore_background or int(tid) > 0}
            for k, tid in enumerate(result.assignment[t]):
                if ignore_background and int(tid) <= 0:
                    continue
                p = lookup.get(int(tid))
                if p is not None:
                    vel[k] = xy[k] - p
                    ok[k] = True
        velocity[t] = vel
        valid[t] = ok
    return velocity, valid


def attach_velocity(dets: Detections, velocity: dict[int, np.ndarray],
                    valid: dict[int, np.ndarray] | None = None) -> Detections:
    """把速度写进检测表（供 `graph/build.py` 的边特征与 `cost.build_cost` 使用）。"""
    for t, vel in velocity.items():
        if t in dets.frames:
            dets.frames[t]["velocity"] = np.asarray(vel, dtype=float)
            if valid is not None and t in valid:
                dets.frames[t]["velocity_valid"] = np.asarray(valid[t], dtype=bool)
    return dets


def velocity_feature_is_live(dets: Detections) -> bool:
    """验收检查（AGENTS.md 要求）：速度特征必须真的非零，避免"死特征"。

    历史上 GNN 图里的 `src_vel` 恒为 0（因为从没人往里写），
    导致式(22) 的运动先验项形同虚设。
    """
    for t in dets.t_range:
        vel = dets.frames[t].get("velocity")
        if vel is not None and np.any(np.abs(np.asarray(vel)) > 0):
            return True
    return False


def run_two_pass(dets: Detections, cfg: "OTTrackConfig",
                 alpha_pred: float, runner=None) -> tuple[TrackResult, dict]:
    """论文口径的两遍式追踪。

    参数
    ----
    cfg        : OTTrackConfig（第 2 遍会使用其 eta/tau/eps 等设置）
    alpha_pred : 式(22) 的 α′；0 表示不做第 2 遍
    runner     : 可注入的追踪函数（默认 `ot_tracker.run_tracking_ot`），便于测试
    """
    if runner is None:
        from ..track.ot_tracker import run_tracking_ot as runner  # type: ignore

    # ---- 第 1 遍：不带运动先验、也不在线估速（保证是"上一轮追踪"的结果）----
    cfg1 = replace(cfg, use_velocity=False)
    result1 = runner(dets, cfg1)

    if alpha_pred <= 0:
        return result1, {"passes": 1, "alpha_pred": 0.0,
                         "velocity_live": False}

    # ---- 式(20)(21)：由第 1 遍的硬关联估速并外推 ----
    velocity, valid = estimate_velocity(dets, result1)
    attach_velocity(dets, velocity, valid)

    # ---- 第 2 遍：带运动先验 α′ ----
    cfg2 = replace(cfg, use_velocity=False)   # 用预计算速度，不用在线估速
    cost = replace(cfg2.cost, alpha_pred=alpha_pred)
    cfg2 = replace(cfg2, cost=cost)
    result2 = runner(dets, cfg2, pred_from_detections=True)
    return result2, {"passes": 2, "alpha_pred": alpha_pred,
                     "velocity_live": velocity_feature_is_live(dets),
                     "n_with_predecessor": int(sum(int(v.sum()) for v in valid.values())),
                     "pass1_tracks": result1.n_tracks(),
                     "pass2_tracks": result2.n_tracks()}
