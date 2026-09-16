"""训练侧/推理侧融合公式一致性测试（防止两侧口径漂移）。"""

from __future__ import annotations

import numpy as np

from celltracker.gnn.fusion import conditional_division_prob, fused_existence


def test_fused_existence_limits():
    rnorm = np.array([0.9, 0.3, 0.001])
    p_rel = np.array([0.1, 0.95, 0.8])
    # λ=0：完全由 OT 先验决定；λ=1：完全由模型决定
    np.testing.assert_allclose(fused_existence(rnorm, p_rel, 0.0), rnorm)
    np.testing.assert_allclose(fused_existence(rnorm, p_rel, 1.0), p_rel)
    # 中间值落在两者之间
    mid = fused_existence(rnorm, p_rel, 0.5)
    assert np.all(mid >= np.minimum(rnorm, p_rel) - 1e-12)
    assert np.all(mid <= np.maximum(rnorm, p_rel) + 1e-12)


def test_conditional_division_prob_torch_and_numpy_agree():
    torch = __import__("pytest").importorskip("torch")
    pm_np = np.array([0.8, 0.2, 0.5])
    pd_np = np.array([0.2, 0.6, 0.5])
    got_np = conditional_division_prob(pm_np, pd_np)
    got_t = conditional_division_prob(torch.tensor(pm_np), torch.tensor(pd_np)).numpy()
    np.testing.assert_allclose(got_np, got_t, atol=1e-7)
    np.testing.assert_allclose(got_np, [0.2, 0.75, 0.5])


def test_training_and_inference_use_same_function():
    """结构性检查：train.py 与 infer.py 都必须从 fusion 模块取融合公式。"""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "src" / "celltracker" / "gnn"
    for name in ("train.py", "infer.py"):
        text = (root / name).read_text()
        assert "from .fusion import" in text, f"{name} 未从 fusion 模块导入融合公式"
