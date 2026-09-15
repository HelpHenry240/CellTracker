"""相邻帧代价矩阵与帧内结构图（对应 ideas.pdf 式 (6)(7)(8)(22)）。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CostConfig:
    """式 (8)(22) 的代价参数。"""

    alpha: float = 1.0          # α: 当前位移代价
    alpha_pred: float = 0.0     # α': 运动先验（匀速预测）代价
    beta: float = 0.0           # β: 尺寸变化代价
    sigma_s: float = 1.0        # σ_s: 尺寸归一化
    r_max: float = 30.0         # R_max: 候选位移上限（超过则置 inf）
    mass_mode: str = "uniform"  # "uniform" | "volume"


def pairwise_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """(n,d) 与 (m,d) 之间的欧氏距离矩阵 (n,m)。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return np.linalg.norm(x[:, None, :] - y[None, :, :], axis=-1)


def build_cost(src_xy: np.ndarray, dst_xy: np.ndarray,
               src_vol: np.ndarray | None = None, dst_vol: np.ndarray | None = None,
               pred_xy: np.ndarray | None = None, cfg: CostConfig | None = None
               ) -> tuple[np.ndarray, dict]:
    """构造 `C_feat`（含 R_max 门限，超出记为 +inf）。

    返回 (C, info)，info 中含位移矩阵 `d_cur`（用于阈值筛选与诊断）。
    """
    cfg = cfg or CostConfig()
    d_cur = pairwise_distance(src_xy, dst_xy)
    C = cfg.alpha * d_cur ** 2

    if cfg.alpha_pred > 0 and pred_xy is not None:
        d_pred = pairwise_distance(pred_xy, dst_xy)
        C = C + cfg.alpha_pred * d_pred ** 2

    if cfg.beta > 0 and src_vol is not None and dst_vol is not None:
        s_src = np.asarray(src_vol, dtype=float)
        s_dst = np.asarray(dst_vol, dtype=float)
        scale = cfg.sigma_s * max(float(np.mean(s_dst)), 1e-9)
        C = C + cfg.beta * ((s_src[:, None] - s_dst[None, :]) / scale) ** 2

    C = np.where(d_cur <= cfg.r_max, C, np.inf)
    return C, {"d_cur": d_cur}


def masses(vol: np.ndarray | None, n: int, mode: str = "uniform") -> np.ndarray:
    """式 (1) 的质量向量 a_i（归一到和为 1）。"""
    if mode == "volume" and vol is not None and np.asarray(vol).size == n:
        w = np.maximum(np.asarray(vol, dtype=float), 1e-9)
    else:
        w = np.ones(n, dtype=float)
    return w / w.sum()


def gaussian_knn_graph(xy: np.ndarray, k: int = 6, sigma_x: float | None = None,
                       feat: np.ndarray | None = None,
                       sigma_f: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """帧内 kNN 图（式 (4)(5)(6)）。

    返回 (D, W)：
      - `D` (n,n) 欧氏距离矩阵（式 (7)，非邻接处为 0 作为"无结构项贡献"）
      - `W` (n,n) 边权（仅 kNN 边非零）
    """
    xy = np.asarray(xy, dtype=float)
    n = xy.shape[0]
    D_full = pairwise_distance(xy, xy)
    if n == 0:
        return D_full, D_full
    k_eff = min(k, max(n - 1, 1))

    order = np.argsort(D_full, axis=1)
    mask = np.zeros((n, n), dtype=bool)
    rows = np.repeat(np.arange(n), k_eff)
    cols = order[:, 1:k_eff + 1].reshape(-1)
    mask[rows, cols] = True
    mask = mask | mask.T  # 对称化

    D = np.where(mask, D_full, 0.0)
    sx = sigma_x if sigma_x is not None else float(np.median(D_full[D_full > 0]) + 1e-9)
    W = np.exp(-D_full ** 2 / (2 * sx ** 2))
    if feat is not None and sigma_f is not None:
        Df = pairwise_distance(feat, feat)
        W = W * np.exp(-Df ** 2 / (2 * sigma_f ** 2))
    W = np.where(mask, W, 0.0)
    return D, W
