"""OT 求解器单测：正确性（对硬约束/线性规划）与数值稳定性。"""

from __future__ import annotations

import numpy as np
import pytest

from celltracker.ot import sinkhorn_log


def _random_problem(n=6, m=7, seed=0):
    rng = np.random.default_rng(seed)
    pts_a = rng.random((n, 2))
    pts_b = rng.random((m, 2))
    C = ((pts_a[:, None, :] - pts_b[None, :, :]) ** 2).sum(-1)
    a = rng.random(n) + 0.5
    a /= a.sum()
    b = rng.random(m) + 0.5
    b /= b.sum()
    return C, a, b


def test_balanced_marginals():
    C, a, b = _random_problem()
    P = sinkhorn_log(C, a, b, eps=0.01)
    assert P.shape == C.shape
    assert np.all(P >= 0)
    np.testing.assert_allclose(P.sum(axis=1), a, atol=1e-8)
    np.testing.assert_allclose(P.sum(axis=0), b, atol=1e-8)


def test_total_mass_preserved_unbalanced():
    C, a, b = _random_problem()
    P = sinkhorn_log(C, a, b, eps=0.05, tau_a=1.0, tau_b=1.0)
    # 非平衡：边际被软化为 KL 惩罚，行和不再等于 a，但应保持在同一量级
    assert 0.5 < P.sum() < 1.5
    assert P.sum() > 0
    # 行和与 a 明显不同（说明约束确实是软的）
    assert not np.allclose(P.sum(axis=1), a, atol=1e-3)
    # 但不应偏离太远
    assert np.max(np.abs(P.sum(axis=1) / a - 1)) < 0.5


def test_large_tau_recovers_balanced():
    C, a, b = _random_problem()
    P_bal = sinkhorn_log(C, a, b, eps=0.05)
    P_unb = sinkhorn_log(C, a, b, eps=0.05, tau_a=1e7, tau_b=1e7)
    np.testing.assert_allclose(P_unb, P_bal, atol=1e-6)


def test_matches_exact_emd_as_eps_to_zero():
    """eps → 0 时熵正则解的费用应逼近精确 OT（用 POT 的 emd 作为参考）。"""
    ot = pytest.importorskip("ot")
    C, a, b = _random_problem(n=8, m=8, seed=3)
    G = ot.emd(a, b, C)
    cost_exact = float((G * C).sum())
    P = sinkhorn_log(C, a, b, eps=1e-3)
    cost_reg = float((P * C).sum())
    assert cost_reg == pytest.approx(cost_exact, rel=1e-3)


def test_infinite_cost_blocks_matching():
    C = np.array([[0.0, np.inf], [np.inf, 0.0]])
    a = np.array([1.0, 1.0])
    b = np.array([1.0, 1.0])
    P = sinkhorn_log(C, a, b, eps=0.1)
    assert P[0, 1] == pytest.approx(0.0, abs=1e-12)
    assert P[1, 0] == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(P.sum(axis=1), a, atol=1e-8)


def test_unbalanced_mass_ratio_scaling():
    """a 的总质量是 b 的两倍时，非平衡解的总质量由两侧 KL 共同决定。"""
    C = np.zeros((1, 1))
    P = sinkhorn_log(C, np.array([2.0]), np.array([1.0]), eps=0.1,
                     tau_a=1.0, tau_b=1.0)
    # 对称惩罚下，最优总质量是两侧质量的几何折中（介于 1 与 2 之间）
    assert 1.0 < float(P[0, 0]) < 2.0
