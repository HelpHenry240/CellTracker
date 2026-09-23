#!/usr/bin/env python3
"""训练 §2.0.1 的边分类 GNN（式29 标签 / 式34 BCE / 式35 OT 一致性正则）。

用法::

    python paperpipe/scripts/train_paper_gnn.py \
        --graphs data/interim/pg_graphs_01 --out paperpipe/runs/gnn_ce01 --epochs 60
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "paperpipe" / "src"))

import papertrack  # noqa: E402,F401  —— 触发 _paths：把 vendor 的 celltracker 副本加入 sys.path

from papertrack.gnn.train import TrainConfig, train   # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda-ot", type=float, default=0.2)
    ap.add_argument("--residual", action="store_true",
                    help="打开残差连接（非原文口径，仅对照）")
    ap.add_argument("--class-weighted", action="store_true",
                    help="类频加权 BCE（非原文口径，仅对照）")
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg = TrainConfig(epochs=args.epochs, hidden=args.hidden, layers=args.layers,
                      lr=args.lr, lambda_ot=args.lambda_ot,
                      residual=args.residual,
                      class_weighted_ce=args.class_weighted, seed=args.seed,
                      device=args.device, out_dir=args.out)
    res = train(args.graphs, cfg)
    print(json.dumps({k: v for k, v in res.items() if k != "history"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
