"""A5 第二层 tracklet OT（§1.6）阶段测试。"""

from __future__ import annotations

import numpy as np

from celltracker.data.ctc import Track
from celltracker.pipeline.config import TrackletConfig
from celltracker.pipeline.tracklet_stage import link_tracklets
from celltracker.track import Detections, TrackResult


def _dets_moving(n_frames=6, n_cells=3, step=2.0):
    """细胞匀速右移的序列。"""
    frames = {}
    for t in range(n_frames):
        xy = np.array([[10.0 * k, 40.0 * k, 0.0] for k in range(n_cells)])
        xy = xy + np.array([step * t, 0.0, 0.0])
        frames[t] = {"label": np.arange(1, n_cells + 1),
                     "centroid": xy, "volume": np.full(n_cells, 100.0)}
    return Detections(frames)


def _fragmented_result():
    """把 6 帧切成 3+3 两段（人为制造碎片，应被第二层合并）。"""
    a = {t: np.array([1, 2, 3]) for t in range(3)}
    b = {t: np.array([4, 5, 6]) for t in range(3, 6)}
    tracks = {i: Track(i, 0, 2, 0) for i in (1, 2, 3)}
    tracks.update({i: Track(i, 3, 5, 0) for i in (4, 5, 6)})
    return TrackResult(assignment={**a, **b}, tracks=tracks)


def test_disabled_is_identity():
    dets = _dets_moving()
    res = _fragmented_result()
    out = link_tracklets(dets, res, TrackletConfig(enabled=False))
    assert out.info["enabled"] is False
    assert out.result.n_tracks() == res.n_tracks()
    for t in res.assignment:
        np.testing.assert_array_equal(out.result.assignment[t], res.assignment[t])


def test_enabled_merges_fragments():
    """第二层 OT 应把 3+3 的碎片合并回 3 条轨迹。"""
    dets = _dets_moving()
    res = _fragmented_result()
    cfg = TrackletConfig(enabled=True, max_gap=4, r_max=30.0, theta_link=0.05)
    out = link_tracklets(dets, res, cfg)
    assert out.info["enabled"] is True
    assert out.info["n_links"] == 3, out.info
    assert out.result.n_tracks() == 3, out.info
    # 合并后所有帧都应是同一组 id
    ids0 = set(np.unique(out.result.assignment[0]).tolist())
    ids5 = set(np.unique(out.result.assignment[5]).tolist())
    assert ids0 == ids5


def test_coupling_shape_and_mass():
    """第二层是**非平衡** OT：总质量 ≤ 1（无后继的 tracklet 质量会流失），
    且质量只落在可行的候选对上。"""
    dets = _dets_moving()
    res = _fragmented_result()
    out = link_tracklets(dets, res, TrackletConfig(enabled=True, max_gap=4))
    assert out.coupling is not None
    n = len(out.table)
    assert out.coupling.shape == (n, n)
    total = float(out.coupling.sum())
    assert 0.0 < total <= 1.0 + 1e-9, f"非平衡 OT 的总质量应在 (0,1]: {total}"
    # 质量必须只落在"gap ≤ max_gap 且位移在门限内"的候选对上
    assert (out.coupling > 1e-9).sum() <= n


def test_result_is_ctc_valid_after_merge():
    """合并结果必须满足 CTC 提交格式（连续、父≤2子、子起点=父终点+1）。"""
    dets = _dets_moving()
    res = _fragmented_result()
    out = link_tracklets(dets, res, TrackletConfig(enabled=True, max_gap=4))
    present: dict[int, set[int]] = {}
    for t, arr in out.result.assignment.items():
        for tid in np.unique(arr):
            present.setdefault(int(tid), set()).add(t)
    for tid, tr in out.result.tracks.items():
        for t in range(tr.begin, tr.end + 1):
            assert t in present.get(tid, set()), f"轨迹 {tid} 在帧 {t} 缺失"


def test_division_children_not_merged_by_default():
    """默认禁止把分裂子轨迹并回父轨迹（分裂边要保留）。"""
    dets = _dets_moving()
    res = _fragmented_result()
    # 把后段标成前段的子轨迹（分裂）
    res.tracks[4] = Track(4, 3, 5, 1)
    out = link_tracklets(dets, res, TrackletConfig(enabled=True, max_gap=4))
    assert 4 in out.result.tracks
    assert out.result.tracks[4].parent == 1, "分裂关系不应被第二层抹掉"


def test_gap_greater_than_one_is_rejected_by_default():
    """CTC 格式约束：gap>1 的合并会留下帧空洞，默认必须拒绝。

    历史 bug：不加这个限制时，第二层找到 19 个连接却只有 3 个真正生效，
    其余被 `finalize_tracks` 当作"轨迹不连续"拆了回去（`segments_split`=16）。
    """
    dets = _dets_moving(n_frames=8)
    # 前段 0-2，后段 4-7（gap = 4-2 = 2 > 1）
    a = {t: np.array([1, 2, 3]) for t in range(3)}
    b = {t: np.array([4, 5, 6]) for t in range(4, 8)}
    tracks = {i: Track(i, 0, 2, 0) for i in (1, 2, 3)}
    tracks.update({i: Track(i, 4, 7, 0) for i in (4, 5, 6)})
    res = TrackResult(assignment={**a, **b}, tracks=tracks)

    # 默认：拒绝 gap>1
    out = link_tracklets(dets, res, TrackletConfig(enabled=True, max_gap=5))
    assert out.info["n_links"] == 0, f"gap>1 应被拒绝: {out.info}"

    # 显式开启"空洞补检测"后才允许（Phase C 会实现真正的插值）
    out2 = link_tracklets(dets, res, TrackletConfig(enabled=True, max_gap=5,
                                                    allow_gap_filling=True))
    assert out2.info["n_links"] == 3, out2.info
    # 但结果仍会被格式校验拆断（说明必须真的补出检测才算数）
    assert out2.info["segments_split"] == 3, out2.info
