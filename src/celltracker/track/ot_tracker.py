"""基于（非平衡）最优传输的追踪器 + 由传输计划重建轨迹（ideas.pdf §1.3/§1.6）。

每对相邻帧上：
  1. 构造 `C_feat`（式 8/22，含 R_max 门限与匀速先验）
  2. 构造帧内 kNN 结构图 `D, D'`（式 4–7）
  3. 解 FGW（式 9）或纯 OT（η=0），熵正则 + 可选 KL 松弛（式 12/14）
  4. 轨迹重建：行内 argmax + θ_Γ/θ_C 阈值；行和/列和不足 → 死亡/出生；
     一行对多个目标显著传输且体积守恒 → 二分裂
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from ..cost.features import CostConfig, build_cost, gaussian_knn_graph, masses
from ..data.ctc import Track
from ..ot.fgw import fused_gw
from ..ot.sinkhorn import sinkhorn_log
from .base import Detections, TrackResult, finalize_tracks

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
    eps_rel: float | None = 0.1      # ε 按代价尺度自适应（见 pipeline/ot_stage.py）
    use_velocity: bool = False
    velocity_weight: float = 1.0
    div_ratio: float = 0.25          # 分裂判定：一行中显著目标的最小质量占比
    div_max_targets: int = 2         # 二分裂
    div_sum_min: float = 0.6         # 两个子目标质量之和的下限（抑制误分裂）
    mass_mode: str = "uniform"       # "uniform" | "volume"


def _row_normalize(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rs = P.sum(axis=1, keepdims=True)
    return P / np.maximum(rs, 1e-300), rs[:, 0]


def run_tracking_ot(dets: Detections, cfg: OTTrackConfig | None = None) -> TrackResult:
    debug = bool(os.environ.get("OT_TRACK_DEBUG"))

    def _put(arr: np.ndarray, j: int, val: int, tag: str) -> None:
        if debug and arr[j] != 0 and arr[j] != val:
            print(f"  [dbg] overwrite out_ids[{j}]: {arr[j]} -> {val} ({tag})")
        arr[j] = val

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

        # 运动先验（式20-22）：用上一帧建立的关联估计速度并外推位置
        pred_xy = None
        if cfg.use_velocity:
            v = np.array([velocity.get(int(i), np.zeros_like(dst_xy[0])) for i in src_ids])
            pred_xy = src_xy + v

        # 统一走 pipeline 的 OT 阶段（与图构建共用同一实现，避免"一个模块两套代码"）
        from ..pipeline.config import MeasureConfig, OTConfig
        from ..pipeline.ot_stage import compute_pairwise_plan
        ot_cfg = OTConfig(
            alpha=1.0, beta=cfg.cost.beta, sigma_s=cfg.cost.sigma_s,
            r_max=cfg.cost.r_max,
            alpha_pred=cfg.velocity_weight if pred_xy is not None else 0.0,
            eta=cfg.eta, eps=cfg.eps,
            eps_rel=getattr(cfg, "eps_rel", None),
            tau_a=cfg.tau_a, tau_b=cfg.tau_b)
        art = compute_pairwise_plan(
            src_xy, dst_xy, ot_cfg, src_vol, dst_vol, pred_xy=pred_xy,
            measure=MeasureConfig(mass_mode=cfg.mass_mode, knn_k=cfg.knn_k))
        P, C = art.plan, art.cost
        info = {"d_cur": art.d_cur if art.d_cur is not None else
                np.linalg.norm(src_xy[:, None, :] - dst_xy[None, :, :], axis=-1)}

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
            # 质量守恒式检查（ideas.pdf §1.6）：两个子目标应共同承担父细胞的绝大部分质量。
            # 注意：CTC 的 GT 标记是等体积小块，故不能用"体积守恒"，须用行内质量分配。
            if float(Rn[i, cols].sum()) < cfg.div_sum_min:
                continue
            parent_tid = int(src_ids[i])
            for k, j in enumerate(cols):
                _put(out_ids, j, next_id + k, f"div src={i}")
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
            _put(out_ids, j, tid, f"match src={i}")
            used_dst.add(j)
            c_new = dst_xy[j]
            if tid in last_c:
                velocity[tid] = c_new - last_c[tid]
            last_c[tid] = c_new

        # --- 其余目标：出生 ---
        for j in range(n_dst):
            if j in used_dst:
                continue
            _put(out_ids, j, next_id, "birth")
            last_c[next_id] = dst_xy[j]
            res.tracks[next_id] = Track(next_id, t, t, 0)
            next_id += 1

        assignment[t] = out_ids

    assignment, tracks, info = finalize_tracks(assignment, res.tracks)
    res.assignment = assignment
    res.tracks = tracks
    res.meta.update(info)
    return res
