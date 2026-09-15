"""OT 专用可视化：耦合矩阵热图、质量流（Sankey 风格）、结构扭曲示意。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .style import PALETTE, savefig


def plot_coupling(P: np.ndarray, src_xy: np.ndarray, dst_xy: np.ndarray,
                  path: str | Path, title: str = "",
                  C: np.ndarray | None = None) -> list[Path]:
    """左：传输计划热图；右：空间中的质量流。"""
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))

    im = axes[0].imshow(P, cmap="viridis", aspect="auto")
    axes[0].set_xlabel("target index (t+1)")
    axes[0].set_ylabel("source index (t)")
    axes[0].set_title("transport plan  Γ")
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    # 空间质量流
    for i in range(src_xy.shape[0]):
        axes[1].scatter(src_xy[i, -1], src_xy[i, -2], s=18, color=PALETTE["blue"],
                        zorder=3)
    for j in range(dst_xy.shape[0]):
        axes[1].scatter(dst_xy[j, -1], dst_xy[j, -2], s=18, color=PALETTE["orange"],
                        zorder=3)
    p_max = P.max() if P.size else 1.0
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            if P[i, j] <= 1e-6:
                continue
            lw = 0.5 + 4.0 * (P[i, j] / p_max)
            axes[1].plot([src_xy[i, -1], dst_xy[j, -1]],
                         [src_xy[i, -2], dst_xy[j, -2]],
                         color=PALETTE["grey"], lw=lw, alpha=0.6, zorder=1)
    axes[1].set_title("mass flow (blue t → orange t+1)")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    if title:
        fig.suptitle(title, y=1.02)
    return savefig(fig, path)


def plot_cost_and_coupling(P: np.ndarray, C: np.ndarray, path: str | Path,
                           title: str = "") -> list[Path]:
    """代价矩阵与传输计划并排（观察"代价低处是否拿到质量"）。"""
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    C_show = np.where(np.isfinite(C), C, np.nan)
    im0 = axes[0].imshow(C_show, cmap="magma_r", aspect="auto")
    axes[0].set_title("cost  C_feat  (white = gated by R_max)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(P, cmap="viridis", aspect="auto")
    axes[1].set_title("transport plan  Γ")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    for ax in axes:
        ax.set_xlabel("target index")
        ax.set_ylabel("source index")
    if title:
        fig.suptitle(title, y=1.02)
    return savefig(fig, path)


def plot_metric_heatmap(grid: dict[str, list], values: dict[tuple, float],
                        metric_name: str, path: str | Path) -> list[Path]:
    """二维参数扫描热图，例如 (eta, eps) → TRA。"""
    xs, ys = sorted(values.keys())[0], None
    x_keys = sorted({k[0] for k in values})
    y_keys = sorted({k[1] for k in values})
    M = np.full((len(y_keys), len(x_keys)), np.nan)
    for (xk, yk), v in values.items():
        M[y_keys.index(yk), x_keys.index(xk)] = v

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    im = ax.imshow(M, cmap="viridis", aspect="auto", origin="lower")
    ax.set_xticks(range(len(x_keys)), [str(k) for k in x_keys])
    ax.set_yticks(range(len(y_keys)), [str(k) for k in y_keys])
    ax.set_xlabel(grid["x_label"])
    ax.set_ylabel(grid["y_label"])
    ax.set_title(metric_name)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center",
                        color="white", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.grid(False)
    return savefig(fig, path)
