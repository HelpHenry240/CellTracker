"""绘图统一风格与保存工具。

注意：本机无中文字体，所有图内文字使用英文（论文投稿也更合适）。
"""

from __future__ import annotations

import os
from pathlib import Path

_CACHE = Path(__file__).resolve().parents[2] / ".cache" / "matplotlib"
_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# 色盲友好（Okabe–Ito）
PALETTE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "grey": "#7F7F7F",
}
SERIES = [PALETTE["blue"], PALETTE["orange"], PALETTE["green"], PALETTE["red"],
          PALETTE["purple"], PALETTE["sky"]]


def setup() -> None:
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "figure.autolayout": True,
    })


setup()


def savefig(fig, path: str | Path, also_pdf: bool = True) -> list[Path]:
    """保存为 png（+pdf），返回写出的文件列表。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = [path.with_suffix(".png")]
    fig.savefig(out[0], dpi=200)
    if also_pdf:
        pdf = path.with_suffix(".pdf")
        fig.savefig(pdf)
        out.append(pdf)
    plt.close(fig)
    return out
