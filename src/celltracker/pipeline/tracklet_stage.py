"""§1.6 短轨迹片段与第二层 OT（tracklet 级 coarse-graining）。

原文（§1.6 末段）：

  1. 在较短滑动窗口（如 3–5 帧）内用 OT 框架求解并**只保留高置信关联**，形成局部
     tracklet；
  2. 把每个 tracklet 视作**超级节点**，为其定义汇总特征（位置均值轨迹、平均速度、
     整体尺寸变化等）；
  3. "在更长时间范围内构建 tracklet 级别的**测度与代价**，再通过第二层 **OT 或
     最小代价流**将这些 tracklet 串联起来"。

本模块实现第 3 步（第 1 步由相邻帧 OT + 重建阈值完成）：

  * **测度**：tracklet 作为超级节点，质量取**与其时长成正比**（时长越长越可信，
    在 OT 里应有更大话语权）；
  * **代价**：`‖末位置(A) − 首位置(B)‖ + αv·‖匀速外推(A) − 首位置(B)‖`，
    门限为 `R_max × gap`、`gap ≤ max_gap`（式20-22 的思想用在 tracklet 级）；
  * **第二层 OT**：对"各 tracklet 的末"与"各 tracklet 的首"求熵正则 OT；
  * **决策与合并**：每行 argmax，行归一化质量 ≥ θ_link 才接受；随后链式合并，
    并交给 `finalize_tracks` 做 CTC 格式规范化。

`cfg.enabled=False` 时是完全恒等映射（消融对照干净）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..data.ctc import Track
from ..ot.sinkhorn import sinkhorn_log
from ..track.base import Detections, TrackResult, finalize_tracks
from ..track.tracklets import tracklet_stats
from .config import TrackletConfig

__all__ = ["TrackletResult", "link_tracklets"]


@dataclass
class TrackletResult:
    """第二层关联结果。"""

    result: TrackResult
    table: dict[int, dict] = field(default_factory=dict)   # tracklet id → 汇总特征
    coupling: np.ndarray | None = None                     # 第二层 OT 的 Γ
    info: dict = field(default_factory=dict)


def link_tracklets(dets: Detections, result: TrackResult,
                   cfg: TrackletConfig) -> TrackletResult:
    """按 §1.6 做第二层 tracklet 关联。"""
    if not cfg.enabled:
        return TrackletResult(result=result, table={}, coupling=None,
                              info={"enabled": False})

    stats = tracklet_stats(result, dets)
    ids = sorted(stats)
    if len(ids) < 2:
        return TrackletResult(result=result, table=stats, coupling=None,
                              info={"enabled": True, "n_tracklets": len(ids),
                                    "n_links": 0})

    # ---- 超级节点：时长作为质量（越长越可信）----
    durations = np.array([len(stats[i]["frames"]) for i in ids], dtype=float)
    mass = durations / durations.sum()

    # ---- 代价：末→首 的位移 + 匀速外推偏差（式20-22 的 tracklet 级版本）----
    n = len(ids)
    # 注意：不可行的配对必须置 **np.inf**（而不是一个大常数）。
    # 若用一个大的有限值，平衡 OT 的硬边缘约束会为了"凑够质量"而把这些
    # 不可能边也算进解里（实测：无后继的 tracklet 行 argmax 指向了无关轨迹），
    # 因为它的行和被迫等于 a_i。
    cost = np.full((n, n), np.inf)
    for ai, a in enumerate(ids):
        sa = stats[a]
        for bi, b in enumerate(ids):
            if a == b:
                continue
            sb = stats[b]
            gap = sb["begin"] - sa["end"]
            if gap < 1 or gap > cfg.max_gap:
                continue
            # CTC 格式：轨迹必须在 [begin, end] 内每帧出现。gap>1 的合并会留下
            # 帧空洞，提交时会被拆回去（实测 19 个连接里 16 个被拆断）。
            # 除非启用"空洞补检测"（Phase C，需要真实掩码来插值）。
            if gap > 1 and not cfg.allow_gap_filling:
                continue
            # 分裂子轨迹默认不并回父轨迹（分裂边应当保留）
            tr_b = result.tracks.get(b)
            if not cfg.merge_division_children and tr_b is not None and tr_b.parent:
                continue
            pred = sa["end_xy"] + sa["velocity"] * gap
            d1 = float(np.linalg.norm(sb["start_xy"] - sa["end_xy"]))
            if d1 > cfg.r_max * gap:
                continue
            d2 = float(np.linalg.norm(sb["start_xy"] - pred))
            cost[ai, bi] = d1 + cfg.velocity_weight * d2

    # ---- 第二层 OT（**非平衡**：tracklet 可以在此终止/起始）----
    # 语义：某 tracklet 找不到后继，应当让它的质量"流失"（对应轨迹结束），
    # 而不是被强行匹配。这正是式(14) 的 KL 松弛所表达的情形。
    finite = np.isfinite(cost)
    if not finite.any():
        return TrackletResult(result=result, table=stats, coupling=None,
                              info={"enabled": True, "n_tracklets": n, "n_links": 0})
    eps = 0.1 * float(np.median(cost[finite]))
    coupling = sinkhorn_log(cost, mass, mass, eps=max(eps, 1e-6),
                            tau_a=cfg.tau, tau_b=cfg.tau)

    # ---- 决策：行内 argmax + 质量阈值 ----
    rnorm = coupling / (coupling.sum(axis=1, keepdims=True) + 1e-12)
    succ: dict[int, int] = {}
    used_target: set[int] = set()
    for ai, a in enumerate(ids):
        row = cost[ai]
        if not np.isfinite(row).any():
            continue
        bi = int(np.argmax(coupling[ai]))
        if not np.isfinite(row[bi]):
            continue
        if rnorm[ai, bi] < cfg.theta_link:
            continue
        b = ids[bi]
        if b in used_target:
            continue
        succ[a] = b
        used_target.add(b)

    # ---- 链式合并 ----
    has_pred = set(succ.values())
    merged = {t: arr.copy() for t, arr in result.assignment.items()}
    new_tracks: dict = {}
    visited: set[int] = set()
    n_chains = 0
    for a in ids:
        if a in has_pred or a in visited:
            continue
        chain = [a]
        cur = a
        while cur in succ and succ[cur] not in chain:
            cur = succ[cur]
            chain.append(cur)
        if len(chain) < 2:
            continue
        n_chains += 1
        visited.update(chain)
        keep = chain[0]
        frames = [stats[c]["begin"] for c in chain] + [stats[c]["end"] for c in chain]
        prev = result.tracks.get(keep)
        new_tracks[keep] = Track(keep, min(frames), max(frames),
                                 prev.parent if prev is not None else 0)
        for c in chain[1:]:
            for t in merged:
                arr = merged[t]
                arr[arr == c] = keep

    # ---- 重建轨迹表并做 CTC 格式规范化 ----
    out_tracks = {}
    for tid, tr in result.tracks.items():
        if tid in visited and tid not in new_tracks:
            continue
        out_tracks[tid] = new_tracks.get(tid, tr)
    for t, arr in merged.items():
        for tid in np.unique(arr):
            tid = int(tid)
            tr = out_tracks.get(tid)
            if tr is None:
                out_tracks[tid] = Track(tid, t, t, 0)
            else:
                out_tracks[tid] = Track(tid, min(tr.begin, t), max(tr.end, t), tr.parent)

    merged, out_tracks, fix_info = finalize_tracks(merged, out_tracks)
    new_result = TrackResult(assignment=merged, tracks=out_tracks,
                             meta={**result.meta, "tracklet_stage": fix_info})
    return TrackletResult(
        result=new_result, table=stats, coupling=coupling,
        info={"enabled": True, "n_tracklets": n, "n_links": len(succ),
              "n_chains": n_chains, "tracks_before": result.n_tracks(),
              "tracks_after": new_result.n_tracks(),
              "theta_link": cfg.theta_link, **fix_info})
