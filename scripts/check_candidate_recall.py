#!/usr/bin/env python3
"""检查式(26) 的候选边筛选是否损伤召回。

式(26) 用 OT 传输计划筛候选边（Γ ≥ θ_Γ 且 C ≤ θ_C）。若阈值过严，
真实关联会被提前排除，后续 GNN 再强也救不回来。本脚本量化：

  - 候选边数量相比纯 R_max 门控减少了多少（计算量收益）
  - **真实 move / division 边被保留的比例**（召回损失）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from celltracker.cost.features import CostConfig, build_cost, masses  # noqa: E402
from celltracker.data.ctc import read_man_track  # noqa: E402
from celltracker.graph.build import LABEL_DIV, LABEL_MOVE, build_pair_graph  # noqa: E402
from celltracker.track.base import Detections  # noqa: E402
from celltracker.graph import GraphConfig  # noqa: E402

import h5py  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--theta-gamma", default="0.0,0.02,0.05,0.1,0.2")
    ap.add_argument("--r-max", type=float, default=30.0)
    ap.add_argument("--frames", default="60,120,170")
    ap.add_argument("--eta", type=float, default=0.0,
                    help="式(9) FGW 结构项权重（>0 启用）")
    ap.add_argument("--tau", type=float, default=None,
                    help="式(14) 非平衡 OT 的 KL 惩罚（不给=平衡 OT）")
    ap.add_argument("--topk", type=int, default=3,
                    help="工程补充：每行额外保留质量前 k（0=关闭）")
    ap.add_argument("--eps-rel", type=float, default=0.1)
    args = ap.parse_args()

    dets = Detections.from_h5(args.h5)
    with h5py.File(args.h5, "r") as f:
        shape = np.asarray(f.attrs["shape"], dtype=float)
        gt = np.asarray(f["tracks"])
    gt_parent = {int(l): int(p) for l, p in zip(gt["label"], gt["parent"])}
    ts = dets.t_range
    probe = [int(x) for x in args.frames.split(",") if int(x) in ts[:-1]]

    print(f"{'theta_gamma':>12s} {'cand/对':>9s} {'move 保留':>10s} {'div 保留':>10s} "
          f"{'none 保留':>10s}")
    for tg in [float(x) for x in args.theta_gamma.split(",")]:
        cfg = GraphConfig(r_max=args.r_max, use_ot=True, cand_from_ot=True,
                          theta_gamma=tg, eta=args.eta, eps_rel=args.eps_rel,
                          cand_topk=args.topk)
        n_cand = keep = tot = 0
        per_class = {LABEL_MOVE: [0, 0], LABEL_DIV: [0, 0], 0: [0, 0]}
        for pos, t in enumerate(ts[:-1]):
            if t not in probe:
                # 只统计探测帧，但 label 需要真实 GT
                pass
            if t not in probe:
                continue
            g = build_pair_graph(dets, t, ts[pos + 1], cfg, gt_parent, shape, len(ts))
            if not g:
                continue
            n_cand += int(g["cand_edges"].shape[0])
            # 用"不筛"的参考集合统计召回
            g_full = build_pair_graph(dets, t, ts[pos + 1],
                                      GraphConfig(r_max=args.r_max, use_ot=True,
                                                  cand_from_ot=False),
                                      gt_parent, shape, len(ts))
            full_keys = {(int(i), int(j)) for i, j in g_full["cand_edges"]}
            kept_keys = {(int(i), int(j)) for i, j in g["cand_edges"]}
            for (i, j), lab in zip(g_full["cand_edges"], g_full["cand_label"]):
                key = (int(i), int(j))
                per_class[int(lab)][1] += 1
                if key in kept_keys:
                    per_class[int(lab)][0] += 1
        parts = []
        for lab in (LABEL_MOVE, LABEL_DIV, 0):
            k, n = per_class[lab]
            parts.append(f"{k / max(n, 1):10.3f}")
        print(f"{tg:12.3f} {n_cand / max(len(probe), 1):9.1f} " + " ".join(parts))
    print("\n（move/div = 真实边被保留的比例；none = 干扰边被保留的比例，越低越好）")


if __name__ == "__main__":
    main()
