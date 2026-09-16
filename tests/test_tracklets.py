"""二层 tracklet 合并的单元测试。"""

from __future__ import annotations

import numpy as np

from celltracker.data.ctc import Track
from celltracker.track import Detections, TrackResult, paint_result
from celltracker.track.tracklets import TrackletMergeConfig, merge_tracklets


def _dets():
    """4 帧、单个细胞沿 x 匀速前进，但第一层把它切成了 2 段。"""
    frames = {}
    for t in range(4):
        frames[t] = {"label": np.array([t + 1]), "centroid": np.array([[float(t * 2), 0.0, 0.0]]),
                     "volume": np.array([100.0])}
    return Detections(frames)


def _fragmented_result():
    """t=0,1 用 id=1；t=2,3 用 id=2（人为切断）。"""
    assignment = {0: np.array([1]), 1: np.array([1]),
                  2: np.array([2]), 3: np.array([2])}
    tracks = {1: Track(1, 0, 1, 0), 2: Track(2, 2, 3, 0)}
    return TrackResult(assignment=assignment, tracks=tracks)


def test_merge_joins_fragments():
    dets = _dets()
    res = merge_tracklets(dets, _fragmented_result(),
                          TrackletMergeConfig(max_gap=2, r_max=10.0))
    assert res.n_tracks() == 1
    assert set(np.unique(res.assignment[0]).tolist()) == {1}
    assert set(np.unique(res.assignment[3]).tolist()) == {1}
    tr = res.tracks[1]
    assert (tr.begin, tr.end) == (0, 3)


def test_no_merge_beyond_max_gap_or_rmax():
    dets = _dets()
    # gap = 1 允许，但 r_max 太小 -> 不应合并
    res = merge_tracklets(dets, _fragmented_result(),
                          TrackletMergeConfig(max_gap=2, r_max=0.1))
    assert res.n_tracks() == 2


def test_division_child_not_merged_back():
    dets = _dets()
    res0 = _fragmented_result()
    res0.tracks[2] = Track(2, 2, 3, 1)  # 标成 1 的子轨迹（分裂）
    res = merge_tracklets(dets, res0, TrackletMergeConfig(max_gap=2, r_max=10.0))
    assert res.n_tracks() == 2


def test_merge_is_consistent_for_painting():
    """合并后仍满足"轨迹在起止帧内连续出现"（CTC 提交要求）。"""
    dets = _dets()
    res = merge_tracklets(dets, _fragmented_result(), TrackletMergeConfig(max_gap=2))
    present: dict[int, set[int]] = {}
    for t, arr in res.assignment.items():
        for tid in np.unique(arr):
            present.setdefault(int(tid), set()).add(t)
    for tid, tr in res.tracks.items():
        missing = [t for t in range(tr.begin, tr.end + 1) if t not in present.get(tid, set())]
        assert not missing, f"轨迹 {tid} 缺失 {missing}"
    # 画掩码不应报错
    vol = np.zeros((2, 4, 4), dtype=np.uint16)
    vol[0, 0, 0] = 1
    out = paint_result(vol, np.array([1]), res.assignment[0])
    assert out.shape == vol.shape
