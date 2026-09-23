"""轨迹容器的规范化：CTC 合法性、空洞登记（§1.6 / §2.0.1 的落地点）。

为什么不能直接复用原仓库的 `finalize_tracks`
-------------------------------------------
`celltracker.track.base.finalize_tracks` 的口径是**"把不连续轨迹拆开"**
（它的 docstring 也是这么写的）——这正好把论文 §2.0.1 要求的
"轨迹在有空洞时**不被截断**"（原话："此时虽然帧 t 的 mask 仍为空，
但整体轨迹不会被截断，TRA 等轨迹指标得到保护"）拆掉了。

因此本包分两种口径：

* `hole_policy="fill"`（默认，论文意图）：保留跨空洞的同一条轨迹，
  把空洞帧登记在 `holes` 里，由 exporter 在**输出 CTC 结果**时补画该帧的掩码
  （CTC 官方格式要求轨迹在 [begin,end] 内每帧都出现，见 FORMULA_MAP.md 的 E-2）；
* `hole_policy="split"`：直接复用原仓库 `finalize_tracks`（CTC 合法但截断轨迹），
  作为对照口径。

两者都执行 CTC 的两条硬性父子约束（原仓库已实现的口径）：
  (a) 子轨迹起点 = 父轨迹终点 + 1；(b) 一个父轨迹最多 2 个子节点。
"""

from __future__ import annotations

import numpy as np

from celltracker.data.ctc import Track
from celltracker.track.base import finalize_tracks   # noqa: F401  split 口径复用

__all__ = ["frames_of", "holes_of", "normalize_tracks"]


def frames_of(assignment: dict[int, np.ndarray]) -> dict[int, list[int]]:
    """每个轨迹 id 实际出现的帧列表（升序）。"""
    out: dict[int, list[int]] = {}
    for t in sorted(assignment):
        for tid in np.unique(assignment[t]):
            out.setdefault(int(tid), []).append(int(t))
    return {k: sorted(v) for k, v in out.items()}


def holes_of(assignment: dict[int, np.ndarray]) -> dict[int, list[int]]:
    """每个轨迹在其 [begin,end] 内**缺失**的帧（即需要补画的空洞帧）。"""
    holes: dict[int, list[int]] = {}
    for tid, fr in frames_of(assignment).items():
        miss = [f for f in range(fr[0], fr[-1] + 1) if f not in set(fr)]
        if miss:
            holes[tid] = miss
    return holes


def _fix_parents(tracks: dict[int, Track]) -> dict:
    """规范化父子关系（官方 TRAMeasure 硬性要求，与原仓库同口径）。"""
    by_parent: dict[int, list[int]] = {}
    for tid, tr in tracks.items():
        if tr.parent:
            by_parent.setdefault(int(tr.parent), []).append(tid)
    fixed = dropped = 0
    for parent, kids in by_parent.items():
        ptr = tracks.get(parent)
        keep = set(sorted(kids, key=lambda c: (tracks[c].begin, c))[:2])
        if len(kids) > 2:
            dropped += len(kids) - 2
        for c in kids:
            ctr = tracks[c]
            if c not in keep or ptr is None or ctr.begin != ptr.end + 1:
                tracks[c] = Track(c, ctr.begin, ctr.end, 0)
                fixed += 1
    return {"parent_fixed": fixed, "extra_children_removed": dropped}


def normalize_tracks(assignment: dict[int, np.ndarray],
                     tracks: dict[int, Track],
                     hole_policy: str = "fill"
                     ) -> tuple[dict[int, np.ndarray], dict[int, Track], dict]:
    """规范化轨迹表；返回 `(assignment, tracks, info)`，并登记空洞。"""
    if hole_policy == "split":
        a, t, info = finalize_tracks(assignment, tracks)
        info["hole_policy"] = "split"
        info["holes_total"] = 0
        return a, t, info
    if hole_policy != "fill":
        raise ValueError(f"未知 hole_policy: {hole_policy!r}")

    present = frames_of(assignment)
    new_tracks: dict[int, Track] = {}
    dropped = 0
    for tid, fr in present.items():
        tr = tracks.get(tid)
        parent = int(tr.parent) if tr is not None else 0
        new_tracks[tid] = Track(tid, fr[0], fr[-1], parent)
    for tid in tracks:
        if tid not in present:
            dropped += 1
    info = {"hole_policy": "fill", "ghost_dropped": dropped, **_fix_parents(new_tracks)}
    holes = holes_of(assignment)
    info["hole_tracks"] = len(holes)
    info["holes_total"] = int(sum(len(v) for v in holes.values()))
    info["holes"] = {int(k): [int(x) for x in v] for k, v in sorted(holes.items())}
    return dict(assignment), new_tracks, info
