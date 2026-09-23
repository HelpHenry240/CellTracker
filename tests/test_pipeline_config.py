"""Phase A 接口层测试：配置 schema、消融开关、OT 阶段。"""

from __future__ import annotations

import numpy as np
import pytest

from celltracker.pipeline import (ABLATIONS, PipelineConfig, apply_ablation,
                                  load_config, save_config)
from celltracker.pipeline.ot_stage import compute_pairwise_plan
from celltracker.pipeline.config import OTConfig


def test_default_config_is_paper_faithful():
    """默认配置应覆盖论文各模块，且默认不启用"论文之外的补充"。"""
    cfg = PipelineConfig()
    assert cfg.ot.eta == 0.0                      # FGW 结构项默认关闭
    assert cfg.ot.tau_a is None                   # 默认平衡 OT
    assert cfg.ot.alpha_pred == 0.0               # 运动先验默认关闭（两遍式第二步才开）
    assert cfg.multiscale.enabled is False
    assert cfg.tracklet.enabled is False
    assert cfg.graph.cand_from_ot is True         # 式(26) 默认启用
    assert cfg.measure.mass_mode == "uniform"     # 决策记录 0001


def test_ablation_switches_turn_modules_off():
    cfg = apply_ablation(PipelineConfig(), {"fgw", "multiscale", "ot_cand"})
    assert cfg.ot.eta == 0.0
    assert cfg.multiscale.enabled is False
    assert cfg.graph.cand_from_ot is False
    assert set(cfg.ablated) == {"fgw", "multiscale", "ot_cand"}
    # 原配置不被修改（不可变语义）
    base = PipelineConfig()
    apply_ablation(base, {"fgw"})
    assert base.ablated == ()


def test_ablation_rejects_unknown_name():
    with pytest.raises(ValueError, match="未知消融项"):
        apply_ablation(PipelineConfig(), {"nonexistent"})


def test_all_ablation_names_are_implemented():
    """ABLATIONS 里声明的每一项都必须真的能改到配置。"""
    for name in ABLATIONS:
        cfg = apply_ablation(PipelineConfig(), {name})
        assert name in cfg.ablated


def test_config_roundtrip(tmp_path):
    cfg = PipelineConfig()
    cfg.ot.eta = 0.3
    cfg.ot.tau_a = cfg.ot.tau_b = 1.0
    cfg.multiscale.enabled = True
    cfg.graph.theta_gamma = 0.05
    path = save_config(cfg, tmp_path / "cfg.yaml")
    back = load_config(path)
    assert back.ot.eta == 0.3
    assert back.ot.tau_a == 1.0
    assert back.multiscale.enabled is True
    assert back.graph.theta_gamma == 0.05


def _two_frame_pair(seed=0):
    rng = np.random.default_rng(seed)
    src = rng.random((6, 3)) * 20
    dst = src + rng.normal(0, 2.0, size=(6, 3))
    return src, dst


def test_ot_stage_balanced_vs_unbalanced():
    src, dst = _two_frame_pair()
    bal = compute_pairwise_plan(src, dst, OTConfig(eta=0.0, tau_a=None, tau_b=None))
    unb = compute_pairwise_plan(src, dst, OTConfig(eta=0.0, tau_a=1.0, tau_b=1.0))
    # 平衡 OT：行和 = a。
    # 容差说明：ε 取 eps_rel×median(C)（此处 ~10¹）时，交替式 Sinkhorn 的
    # 边际误差收敛到 ~2e-6（绝对），与 POT 的解最大差 ~1e-6、目标值一致到 4 位；
    # 该精度比下游阈值（θ_Γ≈0.02）小 4 个数量级，工程上完全够用。
    np.testing.assert_allclose(bal.plan.sum(axis=1), bal.mass_a, atol=1e-5)
    # 非平衡：行和不再等于 a（软约束）
    assert not np.allclose(unb.plan.sum(axis=1), unb.mass_a, atol=1e-3)
    # τ 很大时应逼近平衡解
    big = compute_pairwise_plan(src, dst, OTConfig(eta=0.0, tau_a=1e7, tau_b=1e7))
    np.testing.assert_allclose(big.plan, bal.plan, atol=1e-6)


def test_ot_stage_fgw_runs_and_keeps_marginals():
    src, dst = _two_frame_pair()
    art = compute_pairwise_plan(src, dst, OTConfig(eta=0.5, tau_a=None, tau_b=None))
    assert art.plan.shape == (src.shape[0], dst.shape[0])
    # FGW 的条件梯度会把多次 Sinkhorn 解做凸组合，边际残差比单次 Sinkhorn 略大
    # （实测 ~1e-5 绝对，仍比下游阈值小 3 个数量级）。
    np.testing.assert_allclose(art.plan.sum(axis=1), art.mass_a, atol=5e-5)
    assert art.info["eta"] == 0.5


def test_ot_stage_motion_prior_changes_cost():
    """式(22)：给定时 x̂ 后，代价矩阵应发生变化（α′ 生效）。"""
    src, dst = _two_frame_pair()
    pred = src + 3.0                      # 与真实位移不同的外推
    no_prior = compute_pairwise_plan(src, dst, OTConfig(alpha_pred=0.0))
    with_prior = compute_pairwise_plan(src, dst, OTConfig(alpha_pred=1.0), pred_xy=pred)
    assert not np.allclose(no_prior.cost, with_prior.cost)
    assert with_prior.info["alpha_pred"] == 1.0


def test_eps_is_scale_adaptive():
    """ε 应按代价尺度自适应：坐标放大 2 倍（代价 ×4）时 eps_eff 也应 ×4。

    注意 R_max 门限会破坏自相似性（放大会让远距离配对变成 +inf 被剔除），
    所以这里同步放大 r_max 以隔离"ε 随代价尺度缩放"这一性质。
    """
    src, dst = _two_frame_pair()
    a = compute_pairwise_plan(src, dst, OTConfig(eps_rel=0.1, r_max=200.0))
    b = compute_pairwise_plan(src * 2.0, dst * 2.0,
                              OTConfig(eps_rel=0.1, r_max=800.0))
    assert b.eps_eff == pytest.approx(4.0 * a.eps_eff, rel=0.05)


def test_coupling_roundtrip(tmp_path):
    from celltracker.pipeline.ot_stage import load_coupling

    src, dst = _two_frame_pair()
    art = compute_pairwise_plan(src, dst, OTConfig(eta=0.0))
    path = art.save(tmp_path / "coupling.npz")
    back = load_coupling(path)
    np.testing.assert_allclose(back.plan, art.plan, atol=1e-12)
    np.testing.assert_allclose(back.cost, art.cost, atol=1e-12)
    assert back.eps_eff == pytest.approx(art.eps_eff)


def test_topk_fallback_handles_fewer_targets_than_k():
    """回归测试：目标数 < cand_topk 时 top-k 保底不能崩。

    历史 bug：行索引按 k 构造、列索引只有 min(k, n_dst) 列 → 长度不匹配
    → IndexError（seq02 早期帧只有 2 个目标时触发；seq01 每帧 ≥3 个目标未暴露）。
    """
    from celltracker.graph import GraphConfig, build_pair_graph
    from celltracker.track.base import Detections

    dets = Detections({
        0: {"label": np.array([1, 2]),
            "centroid": np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]),
            "volume": np.array([100.0, 100.0])},
        1: {"label": np.array([1, 2]),
            "centroid": np.array([[1.0, 0.0, 0.0], [6.0, 0.0, 0.0]]),
            "volume": np.array([100.0, 100.0])},
    })
    cfg = GraphConfig(r_max=30.0, cand_from_ot=True, theta_gamma=0.02, cand_topk=3)
    g = build_pair_graph(dets, 0, 1, cfg, {1: 0, 2: 0},
                         shape=np.array([4.0, 64.0, 64.0]), t_total=2)
    assert g["cand_edges"].shape[0] >= 2


def test_false_positive_edges_are_not_labelled_division():
    """回归测试：未匹配检测（gt_label=0）的边不得被标成分裂。

    历史 bug：标签规则 `gt_parent.get(gl_d, 0) == gl_s` 在 `gt_parent` 为空
    （预测 h5 没有 tracks 表）且源检测是假阳性（gl_s=0）时退化成 `0 == 0`，
    把所有假阳性源边标成 DIV → 分裂头在垃圾标签上训练
    （C5.0 漏斗：分裂事件 S2 仅 13.8% 的直接原因之一）。
    """
    from celltracker.graph import GraphConfig, build_pair_graph
    from celltracker.track.base import Detections

    dets = Detections({
        0: {"label": np.array([1, 2]),
            "gt_label": np.array([0, 1]),          # 实例 1 是假阳性
            "centroid": np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]),
            "volume": np.array([100.0, 100.0])},
        1: {"label": np.array([1, 2]),
            "gt_label": np.array([2, 1]),          # 与上一帧身份不同
            "centroid": np.array([[0.5, 0.0, 0.0], [5.5, 0.0, 0.0]]),
            "volume": np.array([100.0, 100.0])},
    })
    cfg = GraphConfig(r_max=30.0, cand_from_ot=True, theta_gamma=0.02, cand_topk=2)
    # gt_parent 为空（模拟预测 h5 无 tracks 表）
    g = build_pair_graph(dets, 0, 1, cfg, {},
                         shape=np.array([4.0, 64.0, 64.0]), t_total=2)
    assert (np.asarray(g["cand_label"]) == 2).sum() == 0


def test_multi_step_context_window_runs():
    """回归：`window>0`（多步时间上下文）此前直接 NameError（`cost_cfg` 未定义）。

    该路径对应 E4.4 的多步上下文消融；重构代价函数时变量被删掉，导致这条
    消融链路静默失效（默认 window=0 所以没暴露）。
    """
    from celltracker.graph import GraphConfig, build_pair_graph
    from celltracker.track.base import Detections

    dets = Detections({
        t: {"label": np.array([1, 2]),
            "centroid": np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]) + t * 0.2,
            "volume": np.array([100.0, 100.0])}
        for t in range(4)
    })
    g = build_pair_graph(dets, 1, 2, GraphConfig(window=1, r_max=30.0), {1: 0, 2: 0},
                         shape=np.array([4.0, 64.0, 64.0]), t_total=4)
    assert g["cand_edges"].shape[0] >= 2
    # 上下文边被并入 edge_index（类型标志 2.0）；window=1 时节点覆盖 4 帧（t-1..t+2）
    assert np.unique(g["node_frame"]).size == 4
    assert g["edge_index"].shape[1] >= g["cand_edges"].shape[0]


def test_node_feature_dim_matches_documented_constant():
    """回归：节点特征实际维度必须与 `GraphConfig.node_feat_dim` 一致。

    背景：该常量曾写 8，而实际产出 7 列（坐标 3 + log 体积 1 + 强度均值/方差 2 + 时间 1），
    文档字符串也跟着写成 (n,8)。这类"文档与实现漂移"会让后续按列取特征时静默出错。
    """
    from celltracker.graph import GraphConfig
    from celltracker.graph.build import _node_features

    n = 5
    xy = np.random.default_rng(0).random((n, 3)) * 10
    vol = np.full(n, 100.0)
    feats = _node_features(xy, vol, np.full(n, 50.0), np.full(n, 5.0),
                           t=0, shape=np.array([4.0, 64.0, 64.0]), t_total=4)
    assert feats.shape == (n, GraphConfig().node_feat_dim)
