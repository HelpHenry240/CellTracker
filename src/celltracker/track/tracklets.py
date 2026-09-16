"""二层 tracklet 关联（ideas.pdf §1.6 "短轨迹片段与第二层 OT"）。

动机（来自官方 TRA 日志的误差分解）：我们的 OT 追踪器在 CE 上的最大失分项是
**ED（多余边，需删除）** —— 把一条真实轨迹切成多段后，片段之间会多出不存在的边。
匈牙利基线正是靠"少切"（碎片化 107 vs 我们的 335+）拿到更低的 AOGM。

做法：
  1. 从第一层结果取出所有 tracklet（及其首/末位置与时间）；
  2. 对"前一 tracklet 的末"与"后一 tracklet 的首"构造代价（位移 + 匀速外推偏差），
     用 `R_max · gap` 门限屏蔽不可能连接，并要求 gap ≤ max_gap；
  3. 用全局指派（匈牙利）求一对一匹配，禁止跨越分裂边（子轨迹不并入父）；
  4. 按匹配结果合并（保留较早的 id），并处理链式合并。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..data.ctc import Track
from .base import Detections, TrackResult

__all__ = ["TrackletMergeConfig", "merge_tracklets"]


@dataclass
class TrackletMergeConfig:
    max_gap: int = 3             # 允许合并的最大时间间隔（帧）
    r_max: float = 30.0          # 单帧位移门限；实际门限 = r_max * gap
    velocity_weight: float = 1.0  # 匀速外推项权重
    merge_division_children: bool = False  # 是否允许把分裂子轨迹并回父轨迹


def _tracklet_stats(result: TrackResult, dets: Detections) -> dict[int, dict]:
    stats: dict[int, dict] = {}
    for t in sorted(result.assignment):
        ids = result.assignment[t]
        xy = dets.centroid(t)
        for k, tid in enumerate(ids):
            tid = int(tid)
            s = stats.setdefault(tid, {"frames": [], "cents": []})
            s["frames"].append(t)
            s["cents"].append(xy[k])
    for tid, s in stats.items():
        s["begin"] = min(s["frames"])
        s["end"] = max(s["frames"])
        s["start_xy"] = s["cents"][0]
        s["end_xy"] = s["cents"][-1]
        if len(s["cents"]) >= 2:
            s["velocity"] = s["cents"][-1] - s["cents"][-2]
        else:
            s["velocity"] = np.zeros_like(s["cents"][0])
    return stats


def merge_tracklets(dets: Detections, result: TrackResult,
                    cfg: TrackletMergeConfig | None = None) -> TrackResult:
    """把第一层结果中"本应相连"的碎片轨迹合并。"""
    from scipy.optimize import linear_sum_assignment

    cfg = cfg or TrackletMergeConfig()
    stats = _tracklet_stats(result, dets)
    ids = sorted(stats)
    if len(ids) < 2:
        return result

    n = len(ids)
    idx = {tid: i for i, tid in enumerate(ids)}
    BIG = 1e9
    cost = np.full((n, n), BIG)

    for a in ids:
        sa = stats[a]
        for b in ids:
            if a == b:
                continue
            sb = stats[b]
            gap = sb["begin"] - sa["end"]
            if gap < 1 or gap > cfg.max_gap:
                continue
            # 子轨迹默认不并回父轨迹（分裂边应当保留）
            parent_b = result.tracks.get(b)
            if not cfg.merge_division_children and parent_b is not None and parent_b.parent:
                continue
            pred = sa["end_xy"] + sa["velocity"] * gap
            d1 = float(np.linalg.norm(sb["start_xy"] - sa["end_xy"]))
            d2 = float(np.linalg.norm(sb["start_xy"] - pred))
            if d1 > cfg.r_max * gap:
                continue
            cost[idx[a], idx[b]] = d1 + cfg.velocity_weight * d2

    rows, cols = linear_sum_assignment(cost)
    # 匈牙利在 BIG 上也会给出指派，这里显式过滤
    pairs = [(ids[i], ids[j]) for i, j in zip(rows, cols) if cost[i, j] < BIG]
    if not pairs:
        return result

    # 链式合并：用"接续后继"表把片段串起来
    succ = {a: b for a, b in pairs}
    has_pred = {b for _, b in pairs}

    merged_assignment: dict[int, np.ndarray] = {}
    for t, ids_t in result.assignment.items():
        merged_assignment[t] = ids_t.copy()
    new_tracks: dict[int, Track] = {}

    visited: set[int] = set()
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
        visited.update(chain)
        keep = chain[0]
        frames = [stats[c]["begin"] for c in chain] + [stats[c]["end"] for c in chain]
        new_tracks[keep] = Track(keep, min(frames), max(frames), 0)
        for c in chain[1:]:
            for t in result.assignment:
                arr = merged_assignment[t]
                arr[arr == c] = keep

    # 重建轨迹表（未合并的保持原样）
    out_tracks: dict[int, Track] = {}
    for tid, tr in result.tracks.items():
        if tid in visited and tid not in new_tracks:
            continue
        out_tracks[tid] = new_tracks.get(tid, tr)
    for t, arr in merged_assignment.items():
        for tid in np.unique(arr):
            tid = int(tid)
            tr = out_tracks.get(tid)
            if tr is None:
                out_tracks[tid] = Track(tid, t, t, 0)
            else:
                out_tracks[tid] = Track(tid, min(tr.begin, t), max(tr.end, t), tr.parent)

    return TrackResult(assignment=merged_assignment, tracks=out_tracks,
                       meta={**result.meta, "merged_pairs": len(pairs),
                             "merge_cfg": cfg.__dict__})
