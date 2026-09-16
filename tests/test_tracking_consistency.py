"""追踪结果的一致性约束（CTC 提交要求：轨迹在起止帧区间内必须连续出现）。"""

from __future__ import annotations

import numpy as np

from celltracker.data.ctc import Track
from celltracker.track import Detections, LinkerConfig, run_tracking
from celltracker.track.ot_tracker import OTTrackConfig, run_tracking_ot


def _synthetic():
    """构造一个"一分为二"的场景：t=0 一个细胞，t=1 变成两个子细胞。"""
    frames = {
        0: {"label": np.array([1]), "centroid": np.array([[0.0, 0.0, 0.0]]),
            "volume": np.array([100.0])},
        1: {"label": np.array([1, 2]), "centroid": np.array([[0.0, 0.0, 0.0],
                                                             [5.0, 0.0, 0.0]]),
            "volume": np.array([55.0, 55.0])},
        2: {"label": np.array([1, 2]), "centroid": np.array([[1.0, 0.0, 0.0],
                                                             [6.0, 0.0, 0.0]]),
            "volume": np.array([57.0, 56.0])},
    }
    return Detections(frames)


def _check_contiguous(result, n_frames=3):
    """每条记录必须在 [begin, end] 内每帧都出现在赋值里。"""
    present: dict[int, set[int]] = {}
    for t, a in result.assignment.items():
        for tid in np.unique(a):
            present.setdefault(int(tid), set()).add(t)
    for tid, tr in result.tracks.items():
        assert tr.begin <= tr.end
        frames = present.get(tid, set())
        missing = [t for t in range(tr.begin, tr.end + 1) if t not in frames]
        assert not missing, f"轨迹 {tid} 在 {missing} 缺失（记录 {tr}）"


def test_hungarian_division_ids_consistent():
    """匈牙利基线的 id 分配必须自洽（历史上曾产生"幽灵 id"）。"""
    res = run_tracking(_synthetic(), LinkerConfig(method="hungarian", max_dist=30.0,
                                                  detect_division=True,
                                                  division_max_dist=30.0))
    _check_contiguous(res)


def test_hungarian_detects_division_when_parent_unmatched():
    """父细胞在下一帧"无匹配"且附近有两个新生目标时，应识别为二分裂。

    注意：匈牙利基线只用"父无匹配"规则，因此这里构造父细胞远离任一子细胞的情形。
    """
    frames = {
        0: {"label": np.array([1]), "centroid": np.array([[0.0, 0.0, 0.0]]),
            "volume": np.array([100.0])},
        1: {"label": np.array([1, 2]),
            "centroid": np.array([[40.0, 0.0, 0.0], [45.0, 0.0, 0.0]]),
            "volume": np.array([55.0, 55.0])},
    }
    res = run_tracking(Detections(frames),
                       LinkerConfig(method="hungarian", max_dist=30.0,
                                    detect_division=True, division_max_dist=60.0))
    _check_contiguous(res)
    parents = [tr.parent for tr in res.tracks.values() if tr.parent]
    assert len(parents) == 2 and len(set(parents)) == 1


def test_ot_tracking_ids_consistent():
    res = run_tracking_ot(_synthetic(), OTTrackConfig(eps=1.0, eta=0.0, cost=_cost()))
    _check_contiguous(res)
    # OT 追踪器按 ideas.pdf §1.6 的"一行对多个目标显著传输"判定分裂
    parents = [tr.parent for tr in res.tracks.values() if tr.parent]
    assert len(parents) == 2 and len(set(parents)) == 1


def _cost():
    from celltracker.cost.features import CostConfig
    return CostConfig(alpha=1.0, r_max=30.0)


def test_no_duplicate_ids_across_frames_within_track():
    """同一 id 不应在两个不同的时间区间里被复用（复用会破坏 CTC 提交格式）。"""
    res = run_tracking(_synthetic(), LinkerConfig(method="greedy", max_dist=30.0))
    seen: dict[int, set[int]] = {}
    for t, a in res.assignment.items():
        for tid in np.unique(a):
            seen.setdefault(int(tid), set()).add(t)
    for tid, fr in seen.items():
        assert fr == set(range(min(fr), max(fr) + 1)), f"id {tid} 出现帧不连续: {fr}"


def test_finalize_tracks_drops_ghost_and_splits_gap():
    """结果校验器：幽灵轨迹必须被丢弃，出现帧不连续的 id 必须被拆开。"""
    from celltracker.data.ctc import Track
    from celltracker.track.base import finalize_tracks

    assignment = {
        0: np.array([1, 2]),
        1: np.array([1, 2]),
        2: np.array([1, 3]),   # id=2 在帧 2 消失
        3: np.array([2, 3]),   # id=2 又出现 -> 不连续，必须拆
    }
    tracks = {1: Track(1, 0, 3, 0), 2: Track(2, 0, 3, 0),
              99: Track(99, 1, 1, 0)}   # 99 是幽灵轨迹
    new_asg, new_tracks, info = finalize_tracks(assignment, tracks)

    assert 99 not in new_tracks
    assert info["ghost_dropped"] == 1
    assert info["segments_split"] == 1

    present: dict[int, set[int]] = {}
    for t, arr in new_asg.items():
        for tid in np.unique(arr):
            present.setdefault(int(tid), set()).add(t)
    for tid, tr in new_tracks.items():
        missing = [t for t in range(tr.begin, tr.end + 1) if t not in present.get(tid, set())]
        assert not missing, f"轨迹 {tid} 在 {missing} 缺失"


def test_finalize_tracks_normalizes_parent_links():
    """父子关系规范化：子起点必须=父终点+1，父最多 2 个子。"""
    from celltracker.data.ctc import Track
    from celltracker.track.base import finalize_tracks

    assignment = {
        0: np.array([1, 2]),
        1: np.array([1, 2]),
        2: np.array([1, 2]),      # 父 1 到帧 2 结束
        3: np.array([2, 2]),      # 帧 3 没有任何"子"（制造时间空洞）
        4: np.array([3, 4]),      # 两个"子"却声明 parent=1（起点 4 ≠ 父终点+1 = 3）
        5: np.array([3, 4]),
    }
    tracks = {1: Track(1, 0, 2, 0), 2: Track(2, 0, 5, 0),
              3: Track(3, 4, 5, 1), 4: Track(4, 4, 5, 1)}
    _, new_tracks, info = finalize_tracks(assignment, tracks)
    assert info["parent_fixed"] >= 1
    for tid, tr in new_tracks.items():
        if tr.parent:
            p = new_tracks[tr.parent]
            assert tr.begin == p.end + 1, f"子 {tid} 起点 {tr.begin} ≠ 父 {tr.parent} 终点+1 {p.end}"


def test_submission_is_ctc_valid():
    """端到端：追踪结果必须通过 CTC 提交的全部格式约束。"""
    import numpy as np

    from celltracker.data.ctc import Track
    from celltracker.track.base import finalize_tracks

    rng = np.random.default_rng(0)
    assignment = {t: rng.integers(1, 5, size=3) for t in range(6)}
    tracks = {i: Track(i, 0, 5, 0) for i in range(1, 5)}
    asg, trs, info = finalize_tracks(assignment, tracks)

    present: dict[int, set[int]] = {}
    for t, arr in asg.items():
        for tid in np.unique(arr):
            present.setdefault(int(tid), set()).add(t)
    kids: dict[int, list[int]] = {}
    for tid, tr in trs.items():
        for t in range(tr.begin, tr.end + 1):
            assert t in present.get(tid, set()), f"轨迹 {tid} 在帧 {t} 缺失"
        if tr.parent:
            kids.setdefault(tr.parent, []).append(tid)
    for parent, ks in kids.items():
        assert len(ks) <= 2, f"父 {parent} 有 {len(ks)} 个子"
        for c in ks:
            assert trs[c].begin == trs[parent].end + 1


def test_division_pr_metric_selfcheck():
    """分裂 P/R 诊断的正确性：完美预测应为 (1,1)，漏判/误判应被正确计数。"""
    import numpy as np

    from celltracker.eval.local_metrics import StreamingDiagnostics

    # GT: 轨迹 1 在帧 0-1 存在，帧 2 分裂为 3 和 4
    gt_tracks = np.array([(1, 0, 1, 0), (3, 2, 3, 1), (4, 2, 3, 1)],
                         dtype=[("label", "i4"), ("begin", "i4"),
                                ("end", "i4"), ("parent", "i4")])

    def run(pred_parent_of_child, pred_ids):
        diag = StreamingDiagnostics(gt_tracks=gt_tracks)
        # 每帧的 GT→预测 映射
        for t, m in pred_ids.items():
            diag.mapping[t] = m
            diag.n_frames += 1
        from celltracker.data.ctc import Track
        pred_tracks = {tid: Track(tid, *span, pred_parent_of_child.get(tid, 0))
                       for tid, span in [(10, (0, 1)), (11, (2, 3)), (12, (2, 3))]}
        return diag.division_pr(pred_tracks)

    # 情形 A：完美预测（父 10 结束，子 11/12 开始，且父≠子）
    perfect = {0: {1: 10}, 1: {1: 10}, 2: {3: 11, 4: 12}, 3: {3: 11, 4: 12}}
    r = run({11: 10, 12: 10}, perfect)
    assert r["division_gt"] == 1 and r["division_pred"] == 1
    assert r["division_recall"] == 1.0 and r["division_precision"] == 1.0

    # 情形 B：漏判（预测把父一路连到子 11，子 12 另起一条无父轨迹）
    missed = {0: {1: 10}, 1: {1: 10}, 2: {3: 10, 4: 12}, 3: {3: 10, 4: 12}}
    r = run({}, missed)
    assert r["division_pred"] == 0
    assert r["division_recall"] == 0.0

    # 情形 C：误判（在非分裂处硬造一次分裂）
    diag = StreamingDiagnostics(gt_tracks=np.array([(1, 0, 3, 0)],
                                                   dtype=[("label", "i4"), ("begin", "i4"),
                                                          ("end", "i4"), ("parent", "i4")]))
    for t, m in {0: {1: 10}, 1: {1: 11}}.items():
        diag.mapping[t] = m
    from celltracker.data.ctc import Track
    r = diag.division_pr({11: Track(11, 1, 3, 10)})
    assert r["division_gt"] == 0 and r["division_pred"] == 1
    assert r["division_precision"] == 0.0
