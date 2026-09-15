"""FGW 结构项与梯度正确性测试。"""

from __future__ import annotations

import numpy as np
import pytest

from celltracker.ot.fgw import fgw_objective, fused_gw, structural_grad, structural_term


def _geom(n=6, seed=0):
    rng = np.random.default_rng(seed)
    xy = rng.random((n, 2))
    D = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
    return D


def test_structural_term_zero_for_isometric_permutation():
    """D == D' 且耦合为置换时，结构扭曲应为 0。"""
    D = _geom(5)
    P = np.eye(5) / 5.0
    assert structural_term(P, D, D) == pytest.approx(0.0, abs=1e-12)


def test_structural_term_positive_for_distorted_coupling():
    D = _geom(5)
    Dp = _geom(5, seed=7)
    P = np.ones((5, 5)) / 25.0
    assert structural_term(P, D, Dp) > 0


def test_structural_grad_matches_finite_difference():
    rng = np.random.default_rng(2)
    n = 4
    D = _geom(n, seed=1)
    Dp = _geom(n, seed=2)
    P = rng.random((n, n)) + 0.1
    G = structural_grad(P, D, Dp)

    eps = 1e-6
    for i in range(n):
        for j in range(n):
            Pp, Pm = P.copy(), P.copy()
            Pp[i, j] += eps
            Pm[i, j] -= eps
            fd = (structural_term(Pp, D, Dp) - structural_term(Pm, D, Dp)) / (2 * eps)
            assert G[i, j] == pytest.approx(fd, rel=1e-5, abs=1e-5)


def test_fgw_reduces_objective_and_keeps_marginals():
    D = _geom(5, seed=3)
    Dp = _geom(5, seed=4)
    a = np.full(5, 1 / 5)
    b = np.full(5, 1 / 5)
    C = np.abs(np.arange(5)[:, None] - np.arange(5)[None, :]).astype(float)

    P, info = fused_gw(C, a, b, D, Dp, eta=0.5, eps=0.1, n_outer=30)
    assert P.shape == (5, 5)
    assert np.all(P >= 0)
    np.testing.assert_allclose(P.sum(axis=1), a, atol=1e-6)
    np.testing.assert_allclose(P.sum(axis=0), b, atol=1e-6)
    # 条件梯度应使目标单调不增
    hist = info["history"]
    assert all(hist[i + 1] <= hist[i] + 1e-9 for i in range(len(hist) - 1))
    # 结构项确实被优化：最终解优于初始 Sinkhorn 解
    from celltracker.ot import sinkhorn_log

    P0 = sinkhorn_log(C, a, b, eps=0.1)
    assert fgw_objective(P, C, D, Dp, 0.5) <= fgw_objective(P0, C, D, Dp, 0.5) + 1e-9
