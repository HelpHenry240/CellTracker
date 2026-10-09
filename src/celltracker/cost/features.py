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
    # 体素物理间距 (z, y, x)，单位 µm。**None = 沿用历史的"体素单位"**。
    #
    # 为什么需要它：CE 的间距是 (1.0, 0.09, 0.09) µm，z 比 x/y 粗 11 倍。
    # 按体素单位算距离时，同一个 `r_max=30` 在平面内等于 2.7 µm、在 z 方向等于
    # **30 µm**（而整个体数据的 z 深度只有 35 µm）→ 门控在 z 方向近乎全放行；
    # 代价 `α·d²` 也把"z 移动 1 µm"与"平面移动 0.09 µm"算成一样贵。
    # 给定本字段后：距离用物理长度，`r_max` 的单位随之变成 **µm**。
    # 参考量级（CE 真实移动边）：位移 p50=0.58 / p95=1.31 / p99.9=2.71 µm。
    spacing_zyx: tuple[float, float, float] | None = None


def pairwise_distance(x: np.ndarray, y: np.ndarray,
                      spacing: tuple[float, ...] | None = None) -> np.ndarray:
    """(n,d) 与 (m,d) 之间的欧氏距离矩阵 (n,m)。

    `spacing` 给定时按物理长度计算（各轴先乘间距再取范数）；None = 体素单位。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    diff = x[:, None, :] - y[None, :, :]
    if spacing is not None:
        diff = diff * np.asarray(spacing, dtype=float)[None, None, :]
    return np.linalg.norm(diff, axis=-1)


def build_cost(src_xy: np.ndarray, dst_xy: np.ndarray,
               src_vol: np.ndarray | None = None, dst_vol: np.ndarray | None = None,
               pred_xy: np.ndarray | None = None, cfg: CostConfig | None = None
               ) -> tuple[np.ndarray, dict]:
    """构造 `C_feat`（含 R_max 门限，超出记为 +inf）。

    返回 (C, info)，info 中含位移矩阵 `d_cur`（用于阈值筛选与诊断）。
    """
    cfg = cfg or CostConfig()
    d_cur = pairwise_distance(src_xy, dst_xy, cfg.spacing_zyx)
    C = cfg.alpha * d_cur ** 2

    if cfg.alpha_pred > 0 and pred_xy is not None:
        d_pred = pairwise_distance(pred_xy, dst_xy, cfg.spacing_zyx)
        C = C + cfg.alpha_pred * d_pred ** 2

    if cfg.beta > 0 and src_vol is not None and dst_vol is not None:
        s_src = np.asarray(src_vol, dtype=float)
        s_dst = np.asarray(dst_vol, dtype=float)
        scale = cfg.sigma_s * max(float(np.mean(s_dst)), 1e-9)
        C = C + cfg.beta * ((s_src[:, None] - s_dst[None, :]) / scale) ** 2

    C = np.where(d_cur <= cfg.r_max, C, np.inf)
    return C, {"d_cur": d_cur,
               "units": "physical_um" if cfg.spacing_zyx else "voxel"}


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
      - `D` (n,n) 局部截断距离矩阵；非邻接处为 0 是近似，不等价于删除 GW 四重和项
      - `W` (n,n) 边权（仅 kNN 边非零）

    此历史核保留分母中的系数2；论文式(6)的字面实现位于 papertrack.representation.measure。
    """
    xy = np.asarray(xy, dtype=float)
    n = xy.shape[0]
    D_full = pairwise_distance(xy, xy)
    if n == 0:
        return D_full, D_full
    k_eff = min(max(int(k), 0), max(n - 1, 0))

    order = np.argsort(D_full, axis=1)
    # ideas.pdf 式(5)：按节点身份排除自身，保留位置重合的其他实例。
    order = order[order != np.arange(n)[:, None]].reshape(n, n - 1)
    mask = np.zeros((n, n), dtype=bool)
    rows = np.repeat(np.arange(n), k_eff)
    cols = order[:, :k_eff].reshape(-1)
    mask[rows, cols] = True
    mask = mask | mask.T  # 对称化

    D = np.where(mask, D_full, 0.0)
    positive = D_full[D_full > 0]
    sx = sigma_x if sigma_x is not None else (float(np.median(positive) + 1e-9) if positive.size else 1.0)
    W = np.exp(-D_full ** 2 / (2 * sx ** 2))
    if feat is not None and sigma_f is not None:
        Df = pairwise_distance(feat, feat)
        W = W * np.exp(-Df ** 2 / (2 * sigma_f ** 2))
    W = np.where(mask, W, 0.0)
    return D, W
