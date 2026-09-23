"""CTC 提交格式校验（AGENTS.md E2：送官方评测前必须本地校验）。

校验项（官方 TRAMeasure 的硬性要求）：
  1. `res_track.txt` 里每条轨迹，在 `[begin, end]` 内**每帧都必须出现**
     （这正是"跨空洞合并"与 CTC 格式冲突的地方，见 `pipeline.export_ctc`）；
  2. 无幽灵轨迹（轨迹表里有、掩码里从未出现）；
  3. 父轨迹存在、子轨迹起点 = 父终点 + 1、父轨迹最多 2 个子节点；
  4. 每帧掩码里的标签都在轨迹表里登记。
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import tifffile

from celltracker.data.ctc import read_man_track

__all__ = ["validate_ctc_dir"]


def validate_ctc_dir(res_dir: str | Path, num_digits: int = 3) -> dict:
    res_dir = Path(res_dir)
    tracks = read_man_track(res_dir / "res_track.txt")
    pat = re.compile(rf"mask(\d{{{num_digits}}})\.tif$", re.IGNORECASE)
    frames: dict[int, set] = {}
    for p in sorted(res_dir.iterdir()):
        m = pat.match(p.name)
        if not m:
            continue
        lab = np.asarray(tifffile.imread(p))
        frames[int(m.group(1))] = set(np.unique(lab).tolist()) - {0}

    errors: list[str] = []
    present: dict[int, set] = {lab: set() for lab in tracks}
    unknown = set()
    for t, labels in frames.items():
        for lab in labels:
            if lab in present:
                present[lab].add(t)
            else:
                unknown.add(lab)
    if unknown:
        errors.append(f"{len(unknown)} 个标签出现在掩码但不在 res_track.txt")

    ghosts = holes = 0
    for lab, tr in tracks.items():
        got = present[lab]
        if not got:
            ghosts += 1
            continue
        expect = set(range(tr.begin, tr.end + 1))
        miss = expect - got
        if miss:
            holes += 1
            if holes <= 5:
                errors.append(f"轨迹 {lab} 在 [B,E] 内有 {len(miss)} 帧缺失")
        extra = got - expect
        if extra:
            errors.append(f"轨迹 {lab} 出现在 [B,E] 之外的 {sorted(extra)[:5]}")
    if ghosts:
        errors.append(f"{ghosts} 条幽灵轨迹（res_track.txt 有、掩码里没有）")

    by_parent: dict[int, list[int]] = {}
    for lab, tr in tracks.items():
        if tr.parent:
            by_parent.setdefault(tr.parent, []).append(lab)
    for parent, kids in by_parent.items():
        if parent not in tracks:
            errors.append(f"父轨迹 {parent} 不存在")
            continue
        if len(kids) > 2:
            errors.append(f"父轨迹 {parent} 有 {len(kids)} 个子节点（>2）")
        for c in kids:
            if tracks[c].begin != tracks[parent].end + 1:
                errors.append(f"子轨迹 {c} 起点 {tracks[c].begin} ≠ "
                              f"父 {parent} 终点+1 {tracks[parent].end + 1}")
    return {"ok": not errors, "errors": errors[:20], "n_errors": len(errors),
            "n_tracks": len(tracks), "n_frames": len(frames),
            "n_ghost_tracks": ghosts, "n_tracks_with_holes": holes}
