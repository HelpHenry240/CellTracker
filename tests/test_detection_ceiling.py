"""C4 检测层天花板（U0–U3）测试：用合成 h5 固化度量口径。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "detection_ceiling.py"


def _write_gt(path: Path) -> None:
    """t=0,1 两条轨迹（1、2）；t=2 时轨迹 1 分裂为 3、4。"""
    tracks = np.array([(1, 0, 1, 0), (2, 0, 1, 0), (3, 2, 2, 1), (4, 2, 2, 1)],
                      dtype=[("label", "i4"), ("begin", "i4"),
                             ("end", "i4"), ("parent", "i4")])
    labels = {0: [1, 2], 1: [1, 2], 2: [3, 4]}
    with h5py.File(path, "w") as f:
        f.create_dataset("tracks", data=tracks)
        g = f.create_group("frames")
        for t, vals in labels.items():
            g.create_group(f"{t:04d}").create_dataset("label", data=np.array(vals))


def _write_pred(path: Path, frame2_gt: list[int]) -> None:
    """预测实例：每帧两个实例，(label, gt_label) 逐帧给出。"""
    mapping = {0: ([1, 2], [1, 2]), 1: ([1, 2], [1, 2]), 2: ([1, 2], frame2_gt)}
    with h5py.File(path, "w") as f:
        g = f.create_group("frames")
        for t, (lab, gt) in mapping.items():
            gg = g.create_group(f"{t:04d}")
            gg.create_dataset("label", data=np.array(lab))
            gg.create_dataset("gt_label", data=np.array(gt))


def _run(tmp_path: Path, pred: Path, gt: Path) -> dict:
    out = tmp_path / "ceiling.json"
    subprocess.run([sys.executable, str(SCRIPT), "--pred-h5", str(pred),
                    "--gt-h5", str(gt), "--out", str(out)],
                   check=True, capture_output=True, text=True)
    import json
    return json.loads(out.read_text())


def test_perfect_detections_reach_all_levels(tmp_path):
    gt = tmp_path / "gt.h5"
    pred = tmp_path / "pred.h5"
    _write_gt(gt)
    _write_pred(pred, frame2_gt=[3, 4])
    r = _run(tmp_path, pred, gt)
    assert r["U0_gt_nodes_detected"]["fraction"] == 1.0
    assert r["U1_move_edges_both_detected"]["fraction"] == 1.0
    # 合成数据：两个球心相距 2*radius+gap 体素，用大一些的 r_max 保证可达
    assert r["U2_move_edges_within_r_max"]["fraction_of_gt"] == 1.0
    assert r["U3_division_edges_parent_and_children_detected"]["fraction"] == 1.0
    assert r["U3_division_edges_parent_and_children_detected"][
        "fraction_children_distinct"] == 1.0


def test_merged_children_are_not_distinct_instances(tmp_path):
    """两个子目标被同一实例吞掉 → 分裂在检测层不可见。

    一个实例只能继承一个 GT id（重叠最大者胜），因此这种情况表现为
    "另一个子标记没有对应实例"：父+两子全被检出的比例直接掉到 0。
    """
    gt = tmp_path / "gt.h5"
    pred = tmp_path / "pred.h5"
    _write_gt(gt)
    _write_pred(pred, frame2_gt=[3, 3])      # 实例 1→gt3；gt4 无实例
    r = _run(tmp_path, pred, gt)
    assert r["U3_division_edges_parent_and_children_detected"]["fraction"] == 0.0
    assert r["U3_division_edges_parent_and_children_detected"][
        "fraction_children_distinct"] == 0.0
    # 移动边仍完好（t=0,1 的实例映射未变）
    assert r["U2_move_edges_within_r_max"]["fraction_of_gt"] == 1.0


def test_drop_isolated_short_tracks_keeps_linked_and_long_tracks():
    """回归：孤立短轨迹过滤只删"短且无父无子"的轨迹。"""
    import numpy as np

    from celltracker.track.base import Track, TrackResult, drop_isolated_short_tracks

    tracks = {
        1: Track(1, 0, 4, 0),      # 长轨迹（保留）
        2: Track(2, 1, 1, 0),      # 短且孤立 → 删
        3: Track(3, 2, 3, 1),      # 短但有父 → 保留
    }
    result = TrackResult(assignment={0: np.array([1, 2]),
                                     1: np.array([1, 2, 3]),
                                     2: np.array([1, 2, 3]),
                                     3: np.array([1, 3]),
                                     4: np.array([1])},
                         tracks=tracks)
    out = drop_isolated_short_tracks(result, max_len=2)
    assert 2 not in out.tracks and out.meta["isolated_short_dropped"] == 1
    assert out.assignment[1][1] == 0        # 被删轨迹的赋值置 0
    assert out.assignment[0][0] == 1        # 其他轨迹不受影响
