"""CTC 提交结果的读写。

提交约定::

    <seq>_RES/mask000.tif ...   每帧标签体数据（2D 为单页 TIFF，3D 为多页 TIFF 栈）
    <seq>_RES/res_track.txt     每行: L B E P

`mask` 中的像素值是轨迹标签（ground truth 的 TRA 掩码同理），因此结果既编码
分割又编码追踪，`res_track.txt` 描述每条轨迹的时间跨度与父节点。
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import tifffile

from ..data.ctc import Track


def write_result(
    labels_per_frame: dict[int, np.ndarray],
    tracks: dict[int, Track],
    out_dir: str | Path,
    num_digits: int = 3,
) -> Path:
    """写出 CTC 结果目录（mask###.tif + res_track.txt）。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for t in sorted(labels_per_frame):
        lab = labels_per_frame[t]
        dtype = np.uint16 if lab.max(initial=0) < 65535 else np.uint32
        tifffile.imwrite(out_dir / f"mask{t:0{num_digits}d}.tif",
                         lab.astype(dtype, copy=False))
    with (out_dir / "res_track.txt").open("w") as fh:
        for label in sorted(tracks):
            tr = tracks[label]
            fh.write(f"{tr.label} {tr.begin} {tr.end} {tr.parent}\n")
    return out_dir


def read_result(res_dir: str | Path, num_digits: int = 3, frames: list[int] | None = None
                ) -> tuple[dict[int, np.ndarray], dict[int, Track]]:
    """读取 CTC 结果目录。"""
    from ..data.ctc import read_man_track

    res_dir = Path(res_dir)
    tracks = read_man_track(res_dir / "res_track.txt")
    pat = re.compile(rf"mask(\d{{{num_digits}}})\.tif$", re.IGNORECASE)
    labels: dict[int, np.ndarray] = {}
    for p in sorted(res_dir.iterdir()):
        m = pat.match(p.name)
        if not m:
            continue
        t = int(m.group(1))
        if frames is not None and t not in frames:
            continue
        labels[t] = tifffile.imread(p)
    return labels, tracks


def tracks_from_labels(labels_per_frame: dict[int, np.ndarray],
                       parents: dict[int, int] | None = None) -> dict[int, Track]:
    """从逐帧标签体数据推导 res_track.txt 所需的轨迹表。

    每个标签的起止帧由其在各帧中的出现位置决定；`parents` 给出分裂父标签。
    """
    present: dict[int, list[int]] = {}
    for t in sorted(labels_per_frame):
        uniq = np.unique(labels_per_frame[t])
        for lab in uniq:
            if lab == 0:
                continue
            present.setdefault(int(lab), []).append(t)
    parents = parents or {}
    tracks: dict[int, Track] = {}
    for lab, ts in present.items():
        tracks[lab] = Track(label=lab, begin=min(ts), end=max(ts),
                            parent=int(parents.get(lab, 0)))
    return tracks
