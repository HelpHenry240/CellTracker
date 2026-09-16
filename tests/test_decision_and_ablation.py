"""决策规则测试：论文口径（GNN 概率 + 阈值）与消融口径（λ 加权融合）。"""

from __future__ import annotations

import numpy as np

from celltracker.gnn.infer import InferConfig, reconstruct_tracks
from celltracker.track import Detections


def _dets():
    """两帧、3 个细胞：1 号在 t=1 处附近出现两个候选（可能是分裂），2 号正常移动。"""
    return Detections({
        0: {"label": np.array([1, 2]),
            "centroid": np.array([[0.0, 0.0, 0.0], [50.0, 50.0, 50.0]]),
            "volume": np.array([100.0, 100.0])},
        1: {"label": np.array([1, 2, 3]),
            "centroid": np.array([[1.0, 0.0, 0.0], [4.0, 0.0, 0.0], [51.0, 50.0, 50.0]]),
            "volume": np.array([100.0, 100.0, 100.0])},
    })


def _preds(rnorm_values, prob, pairs):
    """构造符合真实布局的边特征：(E, 11)，第 8 列是 OT 行归一化质量 rnorm。"""
    feat = np.zeros((len(pairs), 11), dtype=np.float32)
    feat[:, 8] = rnorm_values
    return {0: {"prob": prob, "feat": feat, "pairs": pairs,
                "n_src": 2, "n_dst": 3, "t_next": 1, "cost": np.zeros(len(pairs))}}


def test_paper_mode_links_by_model_move_probability():
    """论文口径：决策只看 GNN 的 P(move)，OT 质量不参与决策。"""
    pairs = np.array([[0, 0], [0, 1], [1, 2]])
    rnorm = np.array([0.01, 0.01, 0.01])          # OT 质量故意全低
    prob = np.array([[0.1, 0.8, 0.1],             # (0→0) 强 move
                     [0.1, 0.1, 0.8],             # (0→1) 强 div
                     [0.1, 0.8, 0.1]])            # (1→2) 强 move
    res = reconstruct_tracks(_dets(), _preds(rnorm, prob, pairs),
                             InferConfig(tau_move=0.5, tau_div=0.5))
    ids0 = set(np.unique(res.assignment[0]).tolist())
    ids1 = set(np.unique(res.assignment[1]).tolist())
    assert ids0.issubset(ids1), f"细胞未延续: {ids0} -> {ids1}"


def test_paper_mode_detects_division():
    """论文口径：两个目标 P(div) 都高 -> 判为分裂。"""
    pairs = np.array([[0, 0], [0, 1], [1, 2]])
    rnorm = np.array([0.5, 0.4, 0.9])
    prob = np.array([[0.1, 0.1, 0.8],             # (0→0) div
                     [0.1, 0.1, 0.8],             # (0→1) div
                     [0.1, 0.8, 0.1]])            # (1→2) move
    res = reconstruct_tracks(_dets(), _preds(rnorm, prob, pairs),
                             InferConfig(tau_move=0.5, tau_div=0.5))
    parents = [tr.parent for tr in res.tracks.values() if tr.parent]
    assert len(parents) == 2, f"应识别出二分裂，实际子轨迹 {parents}"


def test_paper_mode_respects_move_threshold():
    """论文口径：P(move) 低于阈值 -> 不建立关联（目标成为新生轨迹）。"""
    pairs = np.array([[0, 0], [0, 1], [1, 2]])
    rnorm = np.array([0.95, 0.9, 0.9])
    prob = np.array([[0.6, 0.2, 0.2], [0.5, 0.2, 0.3], [0.3, 0.4, 0.3]])
    res = reconstruct_tracks(_dets(), _preds(rnorm, prob, pairs),
                             InferConfig(tau_move=0.5, tau_div=0.5))
    ids0 = set(np.unique(res.assignment[0]).tolist())
    ids1 = set(np.unique(res.assignment[1]).tolist())
    assert not ids0.issubset(ids1), "阈值应挡住低置信关联"


def test_ablation_scores_decompose_into_existence_times_semantics():
    """消融口径的分数分解：P(move)/P(div) = 存在性 × 语义。

    说明：λ=0 时存在性完全来自 OT 先验（A=rnorm），但**语义仍来自模型**，
    因此 λ=0 并非"纯 OT 规则"，而是"OT 定存在性 + 模型定语义"的混合。
    这正是当初把该公式列为消融项的原因——它不是论文的机制。
    """
    from celltracker.gnn.fusion import (conditional_division_prob,
                                        fused_existence, move_division_scores)

    rnorm = np.array([0.9, 0.2, 0.5])
    p_move = np.array([0.8, 0.1, 0.4])
    p_div = np.array([0.2, 0.7, 0.1])
    e_link, e_div = move_division_scores(rnorm, p_move, p_div, lam=0.0)
    sem = conditional_division_prob(p_move, p_div)
    a = fused_existence(rnorm, p_move + p_div, 0.0)
    np.testing.assert_allclose(a, rnorm)          # λ=0：存在性=OT 先验
    np.testing.assert_allclose(e_link, a * (1 - sem))
    np.testing.assert_allclose(e_div, a * sem)
    # λ=1：存在性完全来自模型
    e_link1, e_div1 = move_division_scores(rnorm, p_move, p_div, lam=1.0)
    np.testing.assert_allclose(e_link1, p_move)
    np.testing.assert_allclose(e_div1, p_div)
