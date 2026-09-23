"""tracklet 汇总基元（§1.6 的"超级节点"特征）。

本模块只提供 `tracklet_stats`：把第一层追踪结果汇总成 tracklet 表
（出现帧、首末位置、平均速度、时长等），供第二层关联阶段使用。

**第二层 OT 的实现已迁移到 `pipeline/tracklet_stage.py::link_tracklets`**，
这样"tracklet 级测度 + 代价 + OT 求解 + 合并"只有一处实现，避免口径漂移。
（历史上这里有一版基于匈牙利指派的简化实现，已被论文口径的第二层 OT 取代。）
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..data.ctc import Track
from .base import Detections, TrackResult

__all__ = ["tracklet_stats"]


def tracklet_stats(result: TrackResult, dets: Detections) -> dict[int, dict]:
    """把追踪结果汇总成 tracklet 表（§1.6 的"超级节点"特征）。

    每个 tracklet 给出：出现帧列表、首末位置、平均速度、时长等。
    这是第二层 OT 阶段的输入接口（见 `pipeline/tracklet_stage.py`）。
    """
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

