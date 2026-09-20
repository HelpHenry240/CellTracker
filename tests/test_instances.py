"""C2 实例拆分测试（合成粘连细胞 + 检测召回评估）。"""

from __future__ import annotations

import numpy as np
import pytest

from celltracker.detect.instances import (InstanceSplitConfig,
                                          detection_recall_vs_markers,
                                          split_instances)


def _two_balls(radius=4, gap=0, shape=(32, 32, 32)):
    """两个球：gap=0 时相互接触（分水岭应能切开），gap>0 时分离。"""
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    c1 = np.array([10.0, 16.0, 16.0])
    c2 = np.array([10.0 + 2 * radius + gap, 16.0, 16.0])
    m1 = (z - c1[0]) ** 2 + (y - c1[1]) ** 2 + (x - c1[2]) ** 2 <= radius ** 2
    m2 = (z - c2[0]) ** 2 + (y - c2[1]) ** 2 + (x - c2[2]) ** 2 <= radius ** 2
    return m1, m2


def test_separated_balls_give_two_instances():
    m1, m2 = _two_balls(gap=4)
    labels = split_instances(m1 | m2, InstanceSplitConfig(min_distance=3, min_volume=10))
    assert len(np.unique(labels)) - 1 == 2


def test_touching_balls_are_split_by_watershed():
    """粘连细胞必须被切开（纯连通域只会给 1 个实例）。"""
    m1, m2 = _two_balls(gap=0)
    mask = m1 | m2
    cc = split_instances(mask, InstanceSplitConfig(use_watershed=False, min_volume=10))
    ws = split_instances(mask, InstanceSplitConfig(use_watershed=True, min_distance=3,
                                                   min_volume=10))
    assert len(np.unique(cc)) - 1 == 1            # 连通域：粘成一个
    assert len(np.unique(ws)) - 1 == 2            # 分水岭：切开


def test_min_volume_filters_noise():
    mask = np.zeros((16, 16, 16), dtype=bool)
    mask[0, 0, 0] = True                          # 单像素噪点
    mask[8:12, 8:12, 8:12] = True                 # 真实细胞
    labels = split_instances(mask, InstanceSplitConfig(min_distance=2, min_volume=20))
    assert len(np.unique(labels)) - 1 == 1


def test_detection_recall_perfect_when_mask_equals_markers():
    m1, m2 = _two_balls(gap=4)
    pred = split_instances(m1 | m2, InstanceSplitConfig(min_distance=3, min_volume=10))
    gt_markers = np.zeros_like(pred)
    gt_markers[m1] = 1
    gt_markers[m2] = 2
    stats = detection_recall_vs_markers(pred, gt_markers)
    assert stats["recall"] == pytest.approx(1.0)
    assert stats["precision"] == pytest.approx(1.0)


def test_detection_recall_zero_when_prediction_empty():
    m1, m2 = _two_balls(gap=4)
    gt = np.zeros_like(m1, dtype=np.int32)
    gt[m1], gt[m2] = 1, 2
    stats = detection_recall_vs_markers(np.zeros_like(gt), gt)
    assert stats["recall"] == 0.0


def test_h_frac_positive_ignores_min_distance():
    """回归：`h_frac>0` 时种子由 h-maxima 决定，`min_distance` **不生效**。

    背景（C0 留痕复核）：部署用的 `pred.h5` 实际跑的是 `h_frac=0.35`（类默认值），
    而标定脚本默认 `h_frac=0.0`，两者的 `min_distance` 语义完全不同——
    这就是"标定配置 ≠ 部署配置"的事故来源，故用测试固化该语义。
    """
    m1, m2 = _two_balls(gap=0)
    mask = m1 | m2
    a = split_instances(mask, InstanceSplitConfig(min_distance=2, min_volume=10, h_frac=0.35))
    b = split_instances(mask, InstanceSplitConfig(min_distance=20, min_volume=10, h_frac=0.35))
    assert np.array_equal(a, b)


def test_h_frac_positive_keeps_single_cell_intact():
    """回归：单个细胞核在尺度自适应种子下必须仍是 1 个实例（不产生碎片）。"""
    m1, _ = _two_balls(gap=0)
    labels = split_instances(m1, InstanceSplitConfig(min_volume=10, h_frac=0.35))
    assert len(np.unique(labels)) - 1 == 1
