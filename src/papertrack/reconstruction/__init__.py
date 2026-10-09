"""§1.6 + §2.0.1 轨迹重建与 CTC 导出。

* `rules`     式(23)(24) 的 OT 规则重建 + 生死/分裂判据；式(33) 的 GNN 边决策重建
* `tracks`    轨迹规范化（CTC 合法性、空洞登记）
* `exporter`  流式写 CTC 提交 + 空洞帧补画（论文"不截断" vs CTC"必须连续"的取舍）
"""

from .exporter import export_ctc
from .rules import reconstruct_from_edges, reconstruct_from_ot, volume_consistent
from .tracks import frames_of, holes_of, normalize_tracks

__all__ = ["reconstruct_from_ot", "reconstruct_from_edges", "volume_consistent",
           "normalize_tracks", "holes_of", "frames_of", "export_ctc"]
