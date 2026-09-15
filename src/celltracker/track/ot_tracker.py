"""基于（非平衡）最优传输的追踪器 + 由传输计划重建轨迹（ideas.pdf §1.3/§1.6）。

每对相邻帧上：
  1. 构造 `C_feat`（式 8/22，含 R_max 门限与匀速先验）
  2. 构造帧内 kNN 结构图 `D, D'`（式 4–7）
  3. 解 FGW（式 9）或纯 OT（η=0），熵正则 + 可选 KL 松弛（式 12/14）
  4. 轨迹重建：行内 argmax + θ_Γ/θ_C 阈值；行和/列和不足 → 死亡/出生；
     一行对多个目标显著传输且体积守恒 → 二分裂
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..cost.features import CostConfig, build_cost, gaussian_knn_graph, masses
from ..data.ctc import Track
from ..ot.fgw import fused_gw
from ..ot.sinkhorn import sinkhorn_log
from .base import Detections, TrackResult

__all__ = ["OTTrackConfig", "run_tracking_ot"]


@dataclass
class OTTrackConfig:
    cost: CostConfig = field(default_factory=CostConfig)
    eta: float = 0.3                 # 结构项权重（式 9）
    eps: float = 1.0                 # 熵正则（式 11）
    tau_a: float | None = None       # KL 松弛；None = 平衡 OT（式 14）
    tau_b: float | None = None
    knn_k: int = 6                   # 帧内 kNN 图的 k
    theta_gamma: float = 0.2         # θ_Γ：接受关联所需的最小归一化传输质量（式 24）
    theta_c: float | None = None     # θ_C：最大允许位移；None 时用 cost.r_max
    n_outer: int = 40                # FGW 条件梯度外层迭代
    use_velocity: bool = False
    velocity_weight: float = 1.0
    div_ratio: float = 0.25          # 分裂判定：一行中显著目标的最小质量占比
    div_max_targets: int = 2         # 二分裂
    mass_mode: str = "uniform"       # "uniform" | "volume"


def _row_normalize(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rs = P.sum(axis=1, keepdims=True)
    return P / np.maximum(rs, 1e-300), rs[:, 0]


def run_tracking_ot(dets: Detections, cfg: OTTrackConfig | None = None) -> TrackResult:
    cfg = cfg or OTTrackConfig()
    ts = dets.t_range
    res = TrackResult(meta={"config": {**cfg.__dict__, "cost": cfg.cost.__dict__}})
    if not ts:
        return res

    next_id = 1
    assignment: dict[int, np.ndarray] = {}
    velocity: dict[int, np.ndarray] = {}
    last_c: dict[int, np.ndarray] = {}

    first = ts[0]
    ids0 = np.arange(next_id, next_id + dets.n(first))
    next_id += dets.n(first)
    assignment[first] = ids0
    for k, tid in enumerate(ids0):
        last_c[int(tid)] = dets.centroid(first)[k]

    theta_c = cfg.theta_c if cfg.theta_c is not None else cfg.cost.r_max

    for t_prev, t in zip(ts[:-1], ts[1:]):
        src_xy, dst_xy = dets.centroid(t_prev), dets.centroid(t)
        src_vol, dst_vol = dets.volume(t_prev), dets.volume(t)
        src_ids = assignment[t_prev]
        n_src, n_dst = src_xy.shape[0], dst_xy.shape[0]

        pred_xy = None
        if cfg.use_velocity:
            v = np.array([velocity.get(int(i), np.zeros_like(dst_xy[0])) for i in src_ids])
            pred_xy = src_xy + v
        cost_cfg = cfg.cost
        if pred_xy is not None:
            cost_cfg = CostConfig(**{**cfg.cost.__dict__,
                                     "alpha_pred": cfg.velocity_weight,
                                     "alpha": 1.0})
        C, info = build_cost(src_xy, dst_xy, src_vol, dst_vol, pred_xy, cost_cfg)

        a = masses(src_vol, n_src, cfg.mass_mode)
        b = masses(dst_vol, n_dst, cfg.mass_mode)

        if cfg.eta > 0:
            D, _ = gaussian_knn_graph(src_xy, k=cfg.knn_k)
            Dp, _ = gaussian_knn_graph(dst_xy, k=cfg.knn_k)
            P, _ = fused_gw(C, a, b, D, Dp, eta=cfg.eta, eps=cfg.eps,
                            tau_a=cfg.tau_a, tau_b=cfg.tau_b, n_outer=cfg.n_outer)
        else:
            P = sinkhorn_log(C, a, b, eps=cfg.eps, tau_a=cfg.tau_a, tau_b=cfg.tau_b)

        Rn, row_sum = _row_normalize(P)
        col_sum = P.sum(axis=0)

        out_ids = np.zeros(n_dst, dtype=np.int64)
        used_dst: set[int] = set()

        # --- 分裂判定（先做，避免被 argmax 抢占） ---
        for i in range(n_src):
            if n_dst < 2:
                break
            cols = np.where(Rn[i] >= cfg.div_ratio)[0]
            cols = cols[np.argsort(-Rn[i, cols])]
            cols = [int(j) for j in cols if j not in used_dst][: cfg.div_max_targets]
            if len(cols) < 2:
                continue
            parent_tid = int(src_ids[i])
            for k, j in enumerate(cols):
                out_ids[j] = next_id + k
                last_c[next_id + k] = dst_xy[j]
                velocity[next_id + k] = dst_xy[j] - src_xy[i]
                res.tracks[next_id + k] = Track(next_id + k, t, t, parent_tid)
                used_dst.add(j)
            next_id += len(cols)

        # --- 单目标关联：行内 argmax + θ_Γ / θ_C ---
        for i in range(n_src):
            if row_sum[i] <= 0:
                continue
            j = int(np.argmax(P[i]))
            if j in used_dst:
                continue
            if Rn[i, j] < cfg.theta_gamma or info["d_cur"][i, j] > theta_c:
                continue
            tid = int(src_ids[i])
            out_ids[j] = tid
            used_dst.add(j)
            c_new = dst_xy[j]
            if tid in last_c:
                velocity[tid] = c_new - last_c[tid]
            last_c[tid] = c_new

        # --- 其余目标：出生 ---
        for j in range(n_dst):
            if j in used_dst:
                continue
            out_ids[j] = next_id
            last_c[next_id] = dst_xy[j]
            res.tracks[next_id] = Track(next_id, t, t, 0)
            next_id += 1

        assignment[t] = out_ids

    for t in ts:
        for tid in np.unique(assignment[t]):
            tid = int(tid)
            tr = res.tracks.get(tid)
            if tr is None:
                res.tracks[tid] = Track(tid, t, t, 0)
            else:
                res.tracks[tid] = Track(tid, min(tr.begin, t), max(tr.end, t), tr.parent)
    res.assignment = assignment
    return res
