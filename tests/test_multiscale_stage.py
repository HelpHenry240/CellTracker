"""A4 多尺度时间正则（式17-19）阶段测试。"""

from __future__ import annotations

import numpy as np

from celltracker.pipeline.config import MeasureConfig, MultiscaleConfig, OTConfig
from celltracker.pipeline.multiscale_stage import refine_couplings
from celltracker.pipeline.ot_stage import compute_pairwise_plan
from celltracker.track import Detections


def _chain_dets(n_frames=5, n_cells=6, seed=0):
    """构造一条"细胞匀速右移"的简单序列。"""
    rng = np.random.default_rng(seed)
    base = rng.random((n_cells, 3)) * 20
    frames = {}
    for t in range(n_frames):
        xy = base + np.array([2.0 * t, 0.0, 0.0])
        frames[t] = {"label": np.arange(1, n_cells + 1),
                     "centroid": xy, "volume": np.full(n_cells, 100.0)}
    return Detections(frames)


def _base_couplings(dets, ot_cfg):
    ts = dets.t_range
    out = {}
    for i in range(len(ts) - 1):
        out[i] = compute_pairwise_plan(dets.centroid(ts[i]), dets.centroid(ts[i + 1]),
                                       ot_cfg, dets.volume(ts[i]), dets.volume(ts[i + 1]),
                                       measure=MeasureConfig())
    return out


def test_disabled_is_identity():
    """enabled=False 必须是完全恒等映射（消融对照要干净）。"""
    dets = _chain_dets()
    base = _base_couplings(dets, OTConfig(eps_rel=0.1))
    res = refine_couplings(dets, base, MultiscaleConfig(enabled=False),
                           OTConfig(eps_rel=0.1))
    assert res.info["enabled"] is False
    assert res.history == []
    for i, c in base.items():
        np.testing.assert_array_equal(res.couplings[i].plan, c.plan)


def test_enabled_refines_and_reports_regularizer():
    """enabled=True：应产出与原计划不同的耦合，并返回每轮时间正则值。"""
    dets = _chain_dets()
    ot_cfg = OTConfig(eps_rel=0.1, r_max=50.0)
    base = _base_couplings(dets, ot_cfg)
    cfg = MultiscaleConfig(enabled=True, ks=(2, 3), lambda_temp=0.5, n_rounds=2)
    res = refine_couplings(dets, base, cfg, ot_cfg)

    assert res.info["enabled"] is True
    assert len(res.history) == 2
    assert all(np.isfinite(res.history))
    assert set(res.direct) == {(0, 2), (1, 2), (2, 2), (0, 3), (1, 3)}
    # 计划应被修改（否则正则没起作用）
    changed = any(not np.allclose(res.couplings[i].plan, base[i].plan)
                  for i in base)
    assert changed
    # 形状与边际约束保持
    for i, c in res.couplings.items():
        assert c.plan.shape == base[i].plan.shape
        np.testing.assert_allclose(c.plan.sum(axis=1), c.mass_a, atol=5e-4)


def test_regularizer_is_monotone_non_increasing():
    """加入回溯线搜索后，时间正则必须**单调不增**。

    历史问题：无线搜索时固定步长会把原本一致的解破坏掉
    （合成长序列实测正则会从 0.0037 跳升到 2.32）。
    现按"正则不增"回溯半砍步长，故应单调不增。
    """
    dets = _chain_dets(n_frames=6, n_cells=8, seed=3)
    ot_cfg = OTConfig(eps_rel=0.1, r_max=50.0)
    base = _base_couplings(dets, ot_cfg)
    # λ_temp 取推荐区间内的值（见 MultiscaleConfig 注释：交替优化无线搜索，
    # λ_temp 过大会震荡/发散）
    cfg = MultiscaleConfig(enabled=True, ks=(2,), lambda_temp=0.1, n_rounds=3)
    res = refine_couplings(dets, base, cfg, ot_cfg)
    assert all(np.isfinite(res.history)), f"正则出现非有限值: {res.history}"
    assert all(res.history[k+1] <= res.history[k] + 1e-12 for k in range(len(res.history)-1)), \
        f"正则发散: {res.history}"


def test_refinement_actually_changes_plans():
    """回归测试：精炼必须真的改变计划。

    历史 bug：梯度归一化用 `median|grad|`，而耦合矩阵在 log-domain Sinkhorn 下
    是稀疏的（多数元素下溢为 0），median 恰为 0 → 扰动被静默跳过 → 计划完全不变
    （表现为"正则值跨轮次恒定且严格线性于 λ_temp"）。改用 RMS 后修复。
    """
    dets = _chain_dets(n_frames=5, n_cells=6, seed=1)
    ot_cfg = OTConfig(eps_rel=0.1, r_max=50.0)
    base = _base_couplings(dets, ot_cfg)
    cfg = MultiscaleConfig(enabled=True, ks=(2,), lambda_temp=0.5, n_rounds=2)
    res = refine_couplings(dets, base, cfg, ot_cfg)
    n_changed = sum(1 for i in base
                    if not np.allclose(res.couplings[i].plan, base[i].plan))
    assert n_changed >= len(base) // 2, f"精炼几乎没有生效（仅 {n_changed} 对改变）"


def test_lambda_zero_keeps_plans_close_to_base():
    """λ_temp=0 时正则项不参与，解应与初始解接近（等价性检查）。"""
    dets = _chain_dets()
    ot_cfg = OTConfig(eps_rel=0.1, r_max=50.0)
    base = _base_couplings(dets, ot_cfg)
    cfg = MultiscaleConfig(enabled=True, ks=(2,), lambda_temp=0.0, n_rounds=1)
    res = refine_couplings(dets, base, cfg, ot_cfg)
    for i in base:
        np.testing.assert_allclose(res.couplings[i].plan, base[i].plan, atol=1e-6)
