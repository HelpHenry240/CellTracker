"""§1.5 长程一致性：运动先验（式20–22）与二层 tracklet OT（§1.5 末段/§1.6）。

* `motion`   式(20)(21)(22) 的两遍式运动先验（估速实现复用 vendor 的 `celltracker.pipeline.motion`）
* `tracklet` 短时窗局部 tracklet → 超级节点 → 第二层**非平衡** OT
"""

from .motion import (attach_velocity, estimate_velocity, velocity_report)
from .tracklet import TrackletResult, cut_pieces, link_tracklets

__all__ = ["estimate_velocity", "attach_velocity", "velocity_report",
           "link_tracklets", "cut_pieces", "TrackletResult"]
