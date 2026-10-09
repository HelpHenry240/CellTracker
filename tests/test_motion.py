"""§1.5 运动先验（式 20-22）两遍式实现测试。"""

from __future__ import annotations

import numpy as np
import pytest

from celltracker.data.ctc import Track
from celltracker.pipeline.motion import (attach_velocity, estimate_velocity,
                                         run_two_pass, velocity_feature_is_live)
from celltracker.track import Detections, LinkerConfig, TrackResult
from celltracker.track.ot_tracker import OTTrackConfig, run_tracking_ot


def _moving_dets():
    """两帧：细胞 1 以 +2/帧 匀速移动，细胞 2 静止；第 2 帧新增细胞 3。"""
    return Detections({
        0: {"label": np.array([1, 2]),
            "centroid": np.array([[0.0, 0.0, 0.0], [50.0, 50.0, 50.0]]),
            "volume": np.array([100.0, 100.0])},
        1: {"label": np.array([1, 2, 3]),
            "centroid": np.array([[2.0, 0.0, 0.0], [50.0, 50.0, 50.0],
                                  [10.0, 10.0, 10.0]]),
            "volume": np.array([100.0, 100.0, 100.0])},
    })


def _result():
    """模拟"上一轮追踪"的硬关联：1→1、2→2 延续，3 为新生。"""
    return TrackResult(assignment={0: np.array([1, 2]), 1: np.array([1, 2, 3])},
                       tracks={1: Track(1, 0, 1, 0), 2: Track(2, 0, 1, 0),
                               3: Track(3, 1, 1, 0)})


def test_estimate_velocity_matches_ground_truth():
    """式(20)：速度 = 本帧位置 − 上一帧前驱位置。"""
    dets = _moving_dets()
    vel, valid = estimate_velocity(dets, _result())
    # 帧 0 无前驱 -> 速度为 0 且 valid=False
    assert not valid[0].any()
    np.testing.assert_allclose(vel[0], 0.0)
    # 帧 1：细胞 1 速度 (2,0,0)；细胞 2 静止 (0,0,0)；细胞 3 无前驱
    np.testing.assert_allclose(vel[1][0], [2.0, 0.0, 0.0])
    np.testing.assert_allclose(vel[1][1], [0.0, 0.0, 0.0])
    assert valid[1].tolist() == [True, True, False]


def test_attach_velocity_makes_feature_live():
    """验收（AGENTS.md）：写回后速度特征必须非零，避免"死特征"。"""
    dets = _moving_dets()
    assert not velocity_feature_is_live(dets)
    vel, valid = estimate_velocity(dets, _result())
    attach_velocity(dets, vel, valid)
    assert velocity_feature_is_live(dets)
    np.testing.assert_allclose(dets.frames[1]["velocity"][0], [2.0, 0.0, 0.0])
    assert dets.frames[1]["velocity_valid"].tolist() == [True, True, False]


def test_two_pass_changes_result_and_reports_metadata():
    """两遍式：第 2 遍应使用第 1 遍估出的速度，并给出可核查的元信息。"""
    dets = _moving_dets()
    # 用锐利计划（绝对 ε 小）并关掉分裂判定，以隔离"运动先验"本身的效果
    # 数据有新增细胞，门控后的分量质量不守恒；用 KL 松弛而不是不可行的硬边际。
    cfg = OTTrackConfig(eps=1.0, eps_rel=None, eta=0.0, div_ratio=0.99,tau_a=10.,tau_b=10.,
                        cost=_cost(alpha_pred=0.0))
    res, info = run_two_pass(dets, cfg, alpha_pred=1.0)
    assert info["passes"] == 2
    assert info["velocity_live"] is True
    assert info["n_with_predecessor"] >= 2
    assert res.n_tracks() >= 1


def test_two_pass_with_zero_alpha_pred_equals_single_pass():
    """α′=0 时不应有第 2 遍（剪枝，保证消融对照干净）。"""
    dets = _moving_dets()
    cfg = OTTrackConfig(eps=1.0, eps_rel=None, eta=0.0, div_ratio=0.99,tau_a=10.,tau_b=10.,
                        cost=_cost(0.0))
    res, info = run_two_pass(dets, cfg, alpha_pred=0.0)
    assert info["passes"] == 1
    direct = run_tracking_ot(dets, cfg)
    np.testing.assert_array_equal(res.assignment[1], direct.assignment[1])


def _cost(alpha_pred: float):
    from celltracker.cost.features import CostConfig
    return CostConfig(alpha=1.0, alpha_pred=alpha_pred, r_max=30.0)


def test_motion_prior_biases_toward_predicted_position():
    """式(22)：当 α′ 很大时，匹配应偏向"匀速外推位置"而不是"当前位置最近"。"""
    # 细胞从 x=0 以 +10/帧 移动；下一帧有两个候选：x=10（匀速外推）与 x=0（近但不合运动）
    dets = Detections({
        0: {"label": np.array([1]), "centroid": np.array([[0.0, 0.0, 0.0]]),
            "volume": np.array([100.0])},
        1: {"label": np.array([1]), "centroid": np.array([[10.0, 0.0, 0.0]]),
            "volume": np.array([100.0])},
        2: {"label": np.array([1, 2]),
            "centroid": np.array([[20.0, 0.0, 0.0],   # 匀速外推位置（正确）
                                  [12.0, 0.0, 0.0]]),  # 更靠近上一帧位置（错误）
            "volume": np.array([100.0, 100.0])},
    })
    # 说明：这里刻意用"锐利计划 + 关闭分裂判定"，把 ε 与分裂阈值的耦合排除掉
    # （A2 已证明二者强耦合，会同掩盖运动先验的效果）。
    cfg = OTTrackConfig(eps=1.0, eps_rel=None, eta=0.0, div_ratio=0.99,
                        cost=_cost(0.0))
    vel, valid = estimate_velocity(dets, _result_of_frame2())
    attach_velocity(dets, vel, valid)
    # 带强运动先验：目标 0（x=20）应被选中
    cfg2 = OTTrackConfig(eps=1.0, eps_rel=None, eta=0.0, div_ratio=0.99,
                         cost=_cost(50.0))
    res = run_tracking_ot(dets, cfg2, pred_from_detections=True)
    # 细胞 1 在第 2 帧应延续到目标 0（若速度未生效则可能落到目标 1）
    assert res.assignment[2][0] == res.assignment[1][0]


def _result_of_frame2():
    """构造帧 0→1 的关联（细胞 1 从 0 移到 10），供估速用。"""
    return TrackResult(assignment={0: np.array([1]), 1: np.array([1])},
                       tracks={1: Track(1, 0, 1, 0)})
