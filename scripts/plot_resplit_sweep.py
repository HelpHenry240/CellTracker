#!/usr/bin/env python3
"""C5.0f：超大实例再切规则的参数扫描 + 跨序列验证图。

左图：seq01 上 TRA 随 k 的变化（k 是"体积超过同帧中位数多少倍才再切"的唯一参数）
右图：seq01 / seq02 在"无再切 vs 再切 k=1.6"下的 DET/SEG/TRA 对比

用法::

    python scripts/plot_resplit_sweep.py --out experiments/C5.0f_resplit_sweep/figures/resplit_sweep
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from celltracker.viz import PALETTE, savefig, setup  # noqa: E402

setup()
import matplotlib.pyplot as plt  # noqa: E402

SEQ01 = [("2.0", "experiments/C5.0f_k20_official/metrics_official_01.json"),
         ("1.6", "experiments/C5.0e_official_resplit/metrics_official_01.json"),
         ("1.3", "experiments/C5.0f_k13_official/metrics_official_01.json")]
BASE01 = "experiments/C4A_official_gnn_topk3/metrics_official_01.json"
PAIRS = [("seq01 base", "experiments/C4A_official_gnn_topk3/metrics_official_01.json"),
         ("seq01 resplit", "experiments/C5.0e_official_resplit/metrics_official_01.json"),
         ("seq02 base", "experiments/C5.0f_seq02_base_official/metrics_official_02.json"),
         ("seq02 resplit", "experiments/C5.0f_seq02_rs_official/metrics_official_02.json")]


def _load(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "experiments" / "C5.0f_resplit_sweep"
                                         / "figures" / "resplit_sweep"))
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))

    # --- 左：k 扫描 ---
    ax = axes[0]
    ks = [float(k) for k, _ in SEQ01]
    tra = [_load(p)["TRA"] for _, p in SEQ01]
    base = _load(BASE01)["TRA"]
    ax.plot(ks, tra, "o-", color=PALETTE["blue"], label="re-split (seq01)")
    ax.axhline(base, color=PALETTE["grey"], ls="--", lw=1.0, label="no re-split")
    for k, v in zip(ks, tra):
        ax.annotate(f"{v:.4f}", (k, v), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=7)
    ax.set_xlabel("k  (split if volume > k x frame-median)")
    ax.set_ylabel("official TRA")
    ax.set_title("Re-split threshold sweep (seq01)")
    ax.legend(frameon=False, fontsize=7)

    # --- 右：跨序列 ---
    ax = axes[1]
    labels = [n for n, _ in PAIRS]
    data = [_load(p) for _, p in PAIRS]
    x = np.arange(len(labels))
    width = 0.26
    for i, (metric, color) in enumerate((("DET", PALETTE["green"]),
                                         ("SEG", PALETTE["orange"]),
                                         ("TRA", PALETTE["blue"]))):
        vals = [d[metric] for d in data]
        ax.bar(x + (i - 1) * width, vals, width, label=metric, color=color, alpha=0.9)
        for xi, v in zip(x + (i - 1) * width, vals):
            ax.annotate(f"{v:.3f}", (xi, v), textcoords="offset points",
                        xytext=(0, 2), ha="center", fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylim(0.6, 1.02)
    ax.set_title("Cross-sequence check (k=1.6 chosen on seq01)")
    ax.legend(frameon=False, fontsize=7, ncol=3)
    savefig(fig, Path(args.out))
    print(f"写出 {args.out}.png / .pdf")


if __name__ == "__main__":
    main()
