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


def test_anisotropic_spacing_changes_distance_map():
    """回归：各向异性 spacing 必须传给 EDT（否则距离图按体素单位算，尺度失真）。

    构造 z 方向拉长 4 倍的椭球（体素空间半轴 12/3/3）。真实 CE 间距是
    (0.09, 0.09, 1.0) µm，即数组顺序 (z,y,x) 的 spacing 为 (1.0, 0.09, 0.09)：
    此时 x/y 半轴只有 3×0.09 = 0.27 µm，最大内切半径应 ≈ 0.27 µm，
    而按各向同性体素算是 3.16 —— 量级差 11 倍，正是"没传 sampling"的后果。
    """
    z, y, x = np.ogrid[:40, :40, :40]
    ellipsoid = ((z - 20) / 12.0) ** 2 + ((y - 20) / 3.0) ** 2 + ((x - 20) / 3.0) ** 2 <= 1
    from scipy import ndimage
    iso = ndimage.distance_transform_edt(ellipsoid, sampling=(1.0, 1.0, 1.0)).max()
    aniso = ndimage.distance_transform_edt(
        ellipsoid, sampling=(1.0, 0.09, 0.09)).max()
    assert iso == pytest.approx(3.16, abs=0.1)
    assert aniso == pytest.approx(0.285, abs=0.02)
    assert iso / aniso > 10


def test_split_uses_physical_spacing(monkeypatch):
    """回归：`split_instances` 必须把 spacing 传给 EDT，且 sigma 按 µm 分轴折算。

    用 monkeypatch 直接检查调用参数（比构造"恰好会变"的合成形状更可靠）：
    这正是本次修复的接线点——配置里有 spacing 但没传给 EDT，是实际的 bug。
    """
    from celltracker.detect import instances as mod

    seen: dict[str, object] = {}
    real_edt = mod.ndimage.distance_transform_edt
    real_gauss = mod.ndimage.gaussian_filter

    def spy_edt(binary, sampling=None):      # noqa: ANN001
        seen["sampling"] = sampling
        return real_edt(binary, sampling=sampling)

    def spy_gauss(dist, sigma=None):         # noqa: ANN001
        seen["sigma"] = sigma
        return real_gauss(dist, sigma=sigma)

    monkeypatch.setattr(mod.ndimage, "distance_transform_edt", spy_edt)
    monkeypatch.setattr(mod.ndimage, "gaussian_filter", spy_gauss)

    m1, m2 = _two_balls(gap=0)
    split_instances(m1 | m2, InstanceSplitConfig(
        min_volume=10, h_frac=0.35, gaussian_sigma=0.9, spacing_zyx=(1.0, 0.09, 0.09)))

    assert seen["sampling"] == (1.0, 0.09, 0.09)
    # 0.9 µm 在各轴的体素 sigma = (0.9/1.0, 0.9/0.09, 0.9/0.09) = (0.9, 10, 10)
    assert seen["sigma"] == pytest.approx((0.9, 10.0, 10.0))


def _seeds_from_balls(m1, m2, shape=(32, 32, 32)):
    """在两个球心各放一个单体素种子，模拟 GT 标记。"""
    seeds = np.zeros(shape, dtype=np.int32)
    seeds[10, 16, 16] = 1
    seeds[18, 16, 16] = 2
    return seeds


def test_external_seeds_split_touching_balls():
    """给定外部种子时跳过 h-maxima，直接把种子灌满粘连区域。"""
    m1, m2 = _two_balls(gap=0)
    mask = m1 | m2
    labels = split_instances(mask, InstanceSplitConfig(min_volume=5, h_frac=0.35,
                                                       spacing_zyx=(1.0, 1.0, 1.0)),
                             seeds=_seeds_from_balls(m1, m2))
    assert len(np.unique(labels)) - 1 == 2
    # 每个种子必须落在**自己**的实例里（源到 region 的对应关系不能错）
    assert labels[10, 16, 16] == 1
    assert labels[18, 16, 16] == 2
    assert labels[10, 16, 16] != labels[18, 16, 16]


def test_seed_outside_mask_produces_no_instance():
    """种子落在掩码外 → 该目标没有实例（oracle 实验里必须如实计入缺失）。"""
    m1, _ = _two_balls(gap=0)                 # 只有第 1 个球在掩码里
    seeds = _seeds_from_balls(m1, m1)         # 种子 2 落在掩码外
    labels = split_instances(m1, InstanceSplitConfig(min_volume=5, h_frac=0.35),
                             seeds=seeds)
    assert len(np.unique(labels)) - 1 == 1
    assert labels[10, 16, 16] == 1
    assert labels[18, 16, 16] == 0            # 掩码外仍是背景


def test_unseeded_component_policy_keep():
    """掩码内没有种子的连通域：drop 丢弃、keep_component 各自成一个实例。"""
    m1, m2 = _two_balls(gap=4)
    seeds = np.zeros_like(m1, dtype=np.int32)
    seeds[10, 16, 16] = 1                     # 只给第 1 个球种子
    mask = m1 | m2
    drop = split_instances(mask, InstanceSplitConfig(
        min_volume=5, h_frac=0.35, unseeded_policy="drop"), seeds=seeds)
    keep = split_instances(mask, InstanceSplitConfig(
        min_volume=5, h_frac=0.35, unseeded_policy="keep_component"), seeds=seeds)
    assert len(np.unique(drop)) - 1 == 1
    assert len(np.unique(keep)) - 1 == 2


def test_refine_oversized_splits_only_the_big_blob():
    """回归：后处理再切只应作用于"体积异常大"的实例，其他实例原样保留。

    参数只有一个相对量 k（相对同帧实例体积中位数），因此不依赖数据集尺度。
    两种情形都要覆盖：
      - 大实例内部有**两个峰**（两个粘连核）→ 应被切开；
      - 大实例是**单一平滑团块**（无内部峰）→ 不应被硬切（避免制造过分割）。
    """
    from celltracker.detect.instances import refine_oversized_instances

    z, y, x = np.ogrid[:40, :40, :40]
    # 两个粘连的球，合成**同一个**实例（模拟 nnU-Net 把两个核粘成一个）
    b1 = (z - 20) ** 2 + (y - 13) ** 2 + (x - 20) ** 2 <= 36
    b2 = (z - 20) ** 2 + (y - 27) ** 2 + (x - 20) ** 2 <= 36
    big = b1 | b2
    lab = np.zeros((40, 40, 40), dtype=np.int32)
    lab[big] = 1                       # 大实例（应被再切）
    lab[2:4, 2:4, 2:4] = 2             # 小实例（保持不动）
    out = refine_oversized_instances(big | (lab == 2), lab,
                                     InstanceSplitConfig(min_volume=5, h_frac=0.35,
                                                         spacing_zyx=(1.0, 1.0, 1.0)),
                                     k=1.6)
    small_id = int(out[3, 3, 3])
    assert small_id > 0 and len(np.unique(out[lab == 2])) == 1     # 小实例仍是一个实例
    assert len(np.unique(out[big])) > 1               # 大实例被切成 ≥2 份
    assert set(np.unique(out)) - {0} == set(range(1, out.max() + 1))   # 编号连续


def test_refine_oversized_does_not_split_single_peak_blob():
    """回归：内部只有一个峰的平滑团块不得被硬切（避免制造过分割）。"""
    from celltracker.detect.instances import refine_oversized_instances

    z, y, x = np.ogrid[:40, :40, :40]
    smooth = ((z - 20) / 6.0) ** 2 + ((y - 20) / 10.0) ** 2 + ((x - 20) / 4.0) ** 2 <= 1
    lab = np.zeros((40, 40, 40), dtype=np.int32)
    lab[smooth] = 1
    lab[2:4, 2:4, 2:4] = 2
    out = refine_oversized_instances(smooth | (lab == 2), lab,
                                     InstanceSplitConfig(min_volume=5, h_frac=0.35,
                                                         spacing_zyx=(1.0, 1.0, 1.0)),
                                     k=1.6)
    assert out[smooth].max() == 1 and out[smooth].min() == 1
