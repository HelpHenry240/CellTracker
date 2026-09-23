#!/usr/bin/env python3
"""把"三档检测来源"的官方指标画成对比图（论文用）。

读取各实验目录下的 `metrics_official_*.json`，画出 DET/SEG/TRA 的分组柱状图，
并在 TRA 轴上标注噪声地板 0.0007 的幅度，避免把噪声内差异当结论。

用法::

    python scripts/plot_official_tiers.py --out experiments/C4A_official_gnn_topk3/figures/official_tiers
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

NOISE_FLOOR = 0.0007

# (标签, 官方指标 json 路径, 色)
SERIES = [
    ("GT markers (upper bound)", "experiments/B1_eval_ce01_gnn/metrics_official_01.json",
     PALETTE["grey"]),
    ("nnU-Net masks + GT seeds (oracle)",
     "experiments/C4O_official_gnn/metrics_official_01.json", PALETTE["green"]),
    ("nnU-Net preds, top-k=3 (A)", "experiments/C4A_official_gnn_topk3/metrics_official_01.json",
     PALETTE["blue"]),
    ("+ oversized re-split (k=1.6)", "experiments/C5.0e_official_resplit/metrics_official_01.json",
     PALETTE["purple"]),
    ("+ isolated-track filter (L=2)", "experiments/C5.0e_official_dropiso/metrics_official_01.json",
     PALETTE["red"]),
    ("nnU-Net preds, top-k=5 (B)", "experiments/C4B_official_gnn_topk5/metrics_official_01.json",
     PALETTE["orange"]),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "experiments"
                                         / "C4A_official_gnn_topk3" / "figures" / "official_tiers"))
    args = ap.parse_args()

    labels, rows = [], []
    for name, rel, color in SERIES:
        p = ROOT / rel
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        labels.append(name)
        rows.append((d["DET"], d["SEG"], d["TRA"], color))

    metrics = ["DET", "SEG", "TRA"]
    x = np.arange(len(metrics))
    width = 0.8 / max(len(rows), 1)

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for k, (name, (det, seg, tra, color)) in enumerate(zip(labels, rows)):
        vals = [det, seg, tra]
        ax.bar(x + k * width - 0.4 + width / 2, vals, width,
               label=name, color=color, alpha=0.9)
        for xi, v in zip(x + k * width - 0.4 + width / 2, vals):
            ax.annotate(f"{v:.4f}", (xi, v), textcoords="offset points",
                        xytext=(0, 3), ha="center", fontsize=6.5)

    # TRA 上的噪声地板：以 A 为基准画一条 ±0.0007 的带
    tra_a = next((r[2] for n, r in zip(labels, rows) if "top-k=3" in n), None)
    if tra_a is not None:
        ax.axhspan(tra_a - NOISE_FLOOR, tra_a + NOISE_FLOOR, color=PALETTE["red"],
                   alpha=0.15, zorder=0)
        ax.annotate("noise floor ±0.0007\n(A vs B: +0.00084)",
                    (2.28, tra_a), fontsize=6.5, color=PALETTE["red"], va="center")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0.6, 1.03)
    ax.set_ylabel("official CTC measure")
    ax.set_title("Detection source matters more than the tracking tweak (Fluo-N3DH-CE seq01)")
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    savefig(fig, Path(args.out))
    print(f"写出 {args.out}.png / .pdf")


if __name__ == "__main__":
    main()
