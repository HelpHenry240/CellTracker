"""§1.2 帧内表示：经验测度与邻域图（ideas.pdf 式 1–7）。

原文公式（逐字抄录，R1）
------------------------
式(1)  a_i^t = s_i^t / Σ_{k=1}^{n_t} s_k^t ,  μ^t = Σ_i a_i^t δ_{f_i^t}
       "最简单（且与物理直觉一致）的方式是让质量与尺寸成正比。"
式(2)  f_i^t = [x_i^t, s_i^t] ∈ R^{d+1}
式(3)  a_j^{t+1} = s_j^{t+1} / Σ_ℓ s_ℓ^{t+1}
式(4)  G^t = (V^t, E^t, W^t),  V^t = {1,...,n^t}
式(5)  E^t = { (i,k) | x_k^t ∈ kNN(x_i^t) }
式(6)  w_ik^t = exp( −‖x_i^t−x_k^t‖²/(2σ_x²) − ‖f_i^t−f_k^t‖²/(2σ_f²) ),  (i,k)∈E^t
式(7)  D_ik^t = ‖x_i^t − x_k^t‖₂,  i,k ∈ V^t
       "也可以选择图上的测地距离（例如最短路径长度）来更好地反映局部拓扑结构。"

实现口径
--------
* **式(1) 质量 ∝ 尺寸**：直接复用仓库里已与原文一致的 `celltracker.cost.features.masses`
  （两种模式都归一到和为 1，对应式(1)(3) 的分母）。
* **式(6) 的 W**：原仓库把 W 算出来又全部丢掉（`D, _ = gaussian_knn_graph(...)`），
  导致"声称用了式(6) 实际没用"。本模块**真正产出并下游使用 W**（作为帧内边权重，
  见 `graph.py` 的式(28) 边特征）。
* **式(7)**：原文是全对距离；§1.7 明确允许"结合稀疏邻接（有限度地截断 D^t）来降低
  复杂度"，故默认按 kNN 截断（`d_full=False` 可切回全对）。
* **R3 单位**：所有距离按 `spacing_zyx` 换算成 **µm** 后再进入式(6)(7)(8)。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# 复用：与式(1)(3) 完全一致的质量归一化（原仓库实现）
from celltracker.cost.features import pairwise_distance  # noqa: F401  (式7 的距离原语)

from .config import MeasureConfig

__all__ = ["resolve_spacing", "masses", "point_features", "knn_structure",
           "sigma_from", "pairwise_distance"]


def masses(vol: np.ndarray | None, n: int, mode: str = "volume") -> np.ndarray:
    """式(1)/(3)：质量向量（归一化到和为 1）。

    复用 `celltracker.cost.features.masses`：其实现与原文一致，只是默认模式不同
    （原仓库默认 uniform，本 pipeline 按论文取 volume）。
    """
    from celltracker.cost.features import masses as _masses

    return _masses(vol, n, mode=mode)


def resolve_spacing(cfg_spacing=None, h5_path: str | Path | None = None,
                    raw_image: str | Path | None = None,
                    z_default: float = 1.0) -> tuple[float, float, float] | None:
    """解析体素物理间距 (z,y,x) µm（R3：物理量必须按 spacing 换算）。

    优先级：显式配置 → h5 attrs(`spacing_zyx`) → 原始图像 TIFF 头（xy 分辨率，
    z 用 `z_default`）→ None（调用方须警告"当前按体素单位"）。
    CE 数据集实测 xy = 0.09 µm（TIFF XResolution=111111/cm）、z = 1.0 µm。
    """
    if cfg_spacing is not None:
        return tuple(float(x) for x in cfg_spacing)
    if h5_path is not None:
        try:
            import h5py

            with h5py.File(h5_path, "r") as f:
                if "spacing_zyx" in f.attrs:
                    return tuple(float(x) for x in np.asarray(f.attrs["spacing_zyx"]))
        except OSError:
            pass
    if raw_image is not None:
        try:
            import tifffile

            with tifffile.TiffFile(str(raw_image)) as tf:
                tags = tf.pages[0].tags
                xres = tags.get("XResolution")
                unit = tags.get("ResolutionUnit")
                if xres is not None:
                    num, den = xres.value
                    per_cm = num / max(den, 1)
                    unit_code = int(unit.value) if unit is not None else 3
                    # ResolutionUnit: 3 = cm（CE 数据实测），2 = inch
                    scale = 1e4 if unit_code == 3 else 2.54e4
                    xy = scale / per_cm
                    return (float(z_default), float(xy), float(xy))
        except Exception:  # noqa: BLE001  头信息缺失时退回 None，由调用方决定
            pass
    return None


def point_features(xy: np.ndarray, vol: np.ndarray | None,
                   spacing: tuple[float, ...] | None = None) -> np.ndarray:
    """式(2)：f_i^t = [x_i^t, s_i^t] ∈ R^{d+1}。

    `x` 按 µm 表示（若给 spacing），`s` 取体积原值（式(6) 只用其差值的相对大小）。
    """
    xy = np.asarray(xy, dtype=float)
    if spacing is not None and xy.size:
        xy = xy * np.asarray(spacing, dtype=float)[None, :]
    if vol is None:
        return xy
    return np.concatenate([xy, np.asarray(vol, dtype=float)[:, None]], axis=1)


def sigma_from(values: np.ndarray, given: float | None, positive: bool = True) -> float:
    """式(6) 的 σ_x / σ_f：论文未给数值 → 取"正距离中位数"（CALIBRATED）。

    量纲依据：σ 是被 ‖·‖² 归一化的尺度量，取数据自身的中位尺度是唯一与
    数据规模无关的选择（用固定常数会让核宽随数据尺度失配）。
    """
    if given is not None:
        return float(given)
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if positive:
        v = v[v > 0]
    if v.size == 0:
        return 1.0
    return float(np.median(v))


def knn_structure(xy: np.ndarray, vol: np.ndarray | None, cfg: MeasureConfig,
                  spacing: tuple[float, ...] | None = None,
                  d_full: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """式(4)(5)(6)(7)：帧内 kNN 图，返回 `(D, W, adj)`。

    * `adj` (n,n) bool：kNN 邻接（对称化，对应式(5) 的 E^t）
    * `D`   (n,n)：式(7) 的帧内距离；`d_full=False`（默认）时只保留邻接项，
      非邻接处为 0（= 结构项无贡献），这是 §1.7 允许的稀疏近似。
    * `W`   (n,n)：式(6) 的高斯边权（空间项 × 特征项）。**本实现会真正使用它。**
    """
    xy = np.asarray(xy, dtype=float)
    n = xy.shape[0]
    if n == 0:
        z = np.zeros((0, 0))
        return z, z, np.zeros((0, 0), dtype=bool)

    D_full = pairwise_distance(xy, xy, spacing)                 # 式(7) 全对距离
    k_eff = min(int(cfg.knn_k), max(n - 1, 1))
    order = np.argsort(D_full, axis=1)
    adj = np.zeros((n, n), dtype=bool)
    rows = np.repeat(np.arange(n), k_eff)
    cols = order[:, 1:k_eff + 1].reshape(-1)
    adj[rows, cols] = True
    adj |= adj.T                                                # 对称化

    sx = sigma_from(D_full, cfg.sigma_x)
    W = np.exp(-D_full ** 2 / (2.0 * sx ** 2))                  # 式(6) 空间项
    feat = point_features(xy, vol, spacing)                     # 式(2) f = [x, s]
    if n > 1:
        Df = pairwise_distance(feat, feat)
        sf = sigma_from(Df, cfg.sigma_f)
        W = W * np.exp(-Df ** 2 / (2.0 * sf ** 2))              # 式(6) 特征项
    W = np.where(adj, W, 0.0)
    D = D_full if d_full else np.where(adj, D_full, 0.0)
    return D, W, adj


def sigma_s_from(src_vol: np.ndarray | None, dst_vol: np.ndarray | None,
                 given: float | None) -> float:
    """式(8) 的 σ_s：论文只要求它"用于尺寸归一"。

    CALIBRATED：`given=None` 时取两帧体积的**中位数**（与数据规模无关的量纲标定）。
    注意：原仓库写作 `σ_s · mean(s_dst)`，即在分母上额外乘了目标帧平均尺寸，
    与式(8) 的 (s_i−s_j)/σ_s 不同；本实现按原文恢复。
    """
    if given is not None:
        return float(given)
    vals = [v for v in (src_vol, dst_vol) if v is not None]
    if not vals:
        return 1.0
    allv = np.concatenate([np.asarray(v, dtype=float).ravel() for v in vals])
    allv = allv[np.isfinite(allv)]
    return float(np.median(allv)) if allv.size else 1.0


def build_cost(src_xy: np.ndarray, dst_xy: np.ndarray,
               src_vol: np.ndarray | None = None,
               dst_vol: np.ndarray | None = None,
               pred_xy: np.ndarray | None = None,
               cfg=None, spacing: tuple[float, ...] | None = None
               ) -> tuple[np.ndarray, np.ndarray, dict]:
    """式(8) 与式(22) 的特征代价 `C_feat`（含 R_max 门限）。

    式(8)  C_feat_ij = α‖x_i^t − x_j^{t+1}‖₂² + β((s_i^t − s_j^{t+1})/σ_s)²
           ‖x_i^t − x_j^{t+1}‖₂ > R_max ⇒ C_feat_ij = +∞
    式(22) C̃_ij = α‖x_i−x_j‖² + α′‖x̂_i^{t+1}−x_j^{t+1}‖² + β((s_i−s_j)/σ_s)²

    返回 `(C, d_cur, info)`；`d_cur` 是式(8) 的位移矩阵（µm），下游用于 θ_C 与诊断。
    """
    from .config import CouplingConfig

    cfg = cfg or CouplingConfig()
    d_cur = pairwise_distance(src_xy, dst_xy, spacing)
    C = cfg.alpha * d_cur ** 2

    if cfg.alpha_pred > 0 and pred_xy is not None:
        d_pred = pairwise_distance(pred_xy, dst_xy, spacing)
        C = C + cfg.alpha_pred * d_pred ** 2                    # 式(22) α′ 项

    if cfg.beta > 0 and src_vol is not None and dst_vol is not None:
        s_src = np.asarray(src_vol, dtype=float)
        s_dst = np.asarray(dst_vol, dtype=float)
        scale = sigma_s_from(s_src, s_dst, cfg.sigma_s)
        C = C + cfg.beta * ((s_src[:, None] - s_dst[None, :]) / max(scale, 1e-9)) ** 2

    C = np.where(d_cur <= cfg.r_max, C, np.inf)                  # R_max 门限
    return C, d_cur, {"sigma_s": sigma_s_from(src_vol, dst_vol, cfg.sigma_s),
                      "r_max": float(cfg.r_max),
                      "units": "um" if spacing is not None else "voxel"}
