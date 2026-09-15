"""多尺度时间一致性（式 17–19）测试。"""

from __future__ import annotations

import numpy as np
import pytest

from celltracker.ot.multiscale import temporal_regularizer


def test_temporal_regularizer_zero_for_consistent_product():
    """当跳帧直接耦合恰好等于两步复合时，正则项应为 0（Markov 一致性）。"""
    rng = np.random.default_rng(0)
    P0 = rng.random((3, 4))
    P1 = rng.random((4, 5))
    P0 /= P0.sum(axis=1, keepdims=True)
    P1 /= P1.sum(axis=1, keepdims=True)
    D = P0 @ P1
    assert temporal_regularizer([P0, P1], 0, 2, D) == pytest.approx(0.0, abs=1e-15)


def test_temporal_regularizer_positive_for_inconsistent_product():
    rng = np.random.default_rng(1)
    P0 = rng.random((3, 4))
    P1 = rng.random((4, 5))
    D = rng.random((3, 5))
    assert temporal_regularizer([P0, P1], 0, 2, D) > 0


def test_temporal_regularizer_single_step():
    P = np.eye(3) / 3
    assert temporal_regularizer([P], 0, 1, P) == pytest.approx(0.0, abs=1e-15)
