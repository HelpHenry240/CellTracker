#!/usr/bin/env python3
"""汇总某数据集/序列上的所有实验结果：对比表 + 分组柱状图。

用法::

    python scripts/make_comparison.py --dataset Fluo-N3DH-CE --seq 01 \
        --out-md experiments/CE01_summary.md \
        --out-fig figures/CE01_comparison
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from celltracker.viz import PALETTE, savefig  # noqa: E402


def collect(dataset: str, seq: str) -> list[dict]:
    rows = []
    for exp_dir in sorted((ROOT / "experiments").iterdir()):
        if not exp_dir.is_dir():
            continue
        off = exp_dir / f"metrics_official_{seq}.json"
        loc = exp_dir / "metrics.json"
        if not loc.exists():
            continue
        try:
            m = json.loads(loc.read_text())
        except json.JSONDecodeError:
            continue
        if m.get("dataset") != dataset or str(m.get("seq")) != str(seq):
            continue
        row = {
            "experiment": exp_dir.name,
            "method": m.get("method", m.get("experiment", "")),
            "n_tracks": m.get("n_tracks_pred"),
            "IDsw": m.get("id_switches"),
            "frag": m.get("fragmentation"),
            "FN": m.get("fn"),
            "FP": m.get("fp"),
        }
        if off.exists():
            o = json.loads(off.read_text())
            row["DET"] = o.get("DET")
            row["TRA"] = o.get("TRA")
            row["SEG"] = o.get("SEG")
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--out-md", type=Path, default=None)
    ap.add_argument("--out-csv", type=Path, default=None)
    ap.add_argument("--out-fig", type=Path, default=None)
    args = ap.parse_args()

    rows = collect(args.dataset, args.seq)
    if not rows:
        print("没有找到匹配的实验")
        return

    keys = ["experiment", "method", "n_tracks", "FN", "FP", "IDsw", "frag",
            "SEG", "DET", "TRA"]
    lines = [f"# {args.dataset} seq {args.seq} 实验结果汇总", "",
             "| " + " | ".join(keys) + " |",
             "|" + "---|" * len(keys)]
    for r in rows:
        lines.append("| " + " | ".join(
            ("—" if r.get(k) is None else
             (f"{r[k]:.4f}" if isinstance(r.get(k), float) else str(r.get(k))))
            for k in keys) + " |")
    table = "\n".join(lines) + "\n"
    print(table)
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(table)
    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows([{k: r.get(k) for k in keys} for r in rows])

    if args.out_fig:
        import matplotlib.pyplot as plt
        import numpy as np

        have_official = [r for r in rows if r.get("TRA") is not None]
        labels = [r["experiment"].replace("_", "\n", 1) for r in have_official]
        x = np.arange(len(labels))
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
        w = 0.38
        axes[0].bar(x - w / 2, [r["TRA"] for r in have_official], w,
                    label="TRA", color=PALETTE["blue"])
        axes[0].bar(x + w / 2, [r["DET"] for r in have_official], w,
                    label="DET", color=PALETTE["sky"])
        axes[0].set_ylim(min(0.99, min(r["TRA"] for r in have_official) - 0.002), 1.001)
        axes[0].set_title("official CTC metrics")
        axes[0].legend()
        axes[1].bar(x, [r["IDsw"] for r in have_official], color=PALETTE["orange"])
        axes[1].set_title("ID switches")
        axes[2].bar(x, [r["frag"] for r in have_official], color=PALETTE["red"])
        axes[2].set_title("fragmentation")
        for ax in axes:
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=6, rotation=0)
        fig.suptitle(f"{args.dataset} seq {args.seq} — method comparison", y=1.04)
        savefig(fig, args.out_fig)
        print(f"图 -> {args.out_fig}")


if __name__ == "__main__":
    main()
