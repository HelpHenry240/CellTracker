#!/usr/bin/env python3
"""A2 消融：η（式9 FGW）、τ（式14 非平衡）、α′（式22 运动先验）对 **OT 先验质量** 的影响。

为什么测这个而不是最终 TRA：候选边的**覆盖度**由 R_max 与 top-k 决定（实测 η/τ 不影响），
η/τ 真正改变的是**传输计划的数值**——也就是交给 GNN 的边特征（rnorm / log质量 /
argmax 指示）。所以这里直接量化"OT 先验把真实边与虚假边分开的能力"：

  * AUC(rnorm → 该边是否真实)：越接近 1 越好
  * 真实边的平均 rnorm vs 虚假边的平均 rnorm
  * 每行 argmax 命中真实边的比例（式23 的 argmax 规则本身有多准）

用法::

    python scripts/ablate_ot_prior.py --h5 data/interim/Fluo-N3DH-CE_01.h5 \
        --configs "eta=0,0.3,0.5" --tau 1.0 --frames 60,120,170
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from celltracker.cost.features import CostConfig, build_cost  # noqa: E402
from celltracker.pipeline.config import MeasureConfig, OTConfig  # noqa: E402
from celltracker.pipeline.ot_stage import compute_pairwise_plan  # noqa: E402
from celltracker.track.base import Detections  # noqa: E402


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann–Whitney U 统计量（正负样本分离度），无需 sklearn。"""
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, scores.size + 1)
    pos = labels.astype(bool)
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--configs", default="eta=0,0.3,0.5",
                    help="逗号分隔的 η 取值（其余参数用 --tau/--alpha-pred 固定）")
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--alpha-pred", type=float, default=0.0)
    ap.add_argument("--r-max", type=float, default=30.0)
    ap.add_argument("--eps-rel", type=float, default=0.1)
    ap.add_argument("--frames", default="60,90,120,150,170")
    args = ap.parse_args()

    dets = Detections.from_h5(args.h5)
    with h5py.File(args.h5, "r") as f:
        gt = np.asarray(f["tracks"])
    parent = {int(l): int(p) for l, p in zip(gt["label"], gt["parent"])}
    ts = dets.t_range
    probe = [t for t in (int(x) for x in args.frames.split(",")) if t in ts[:-1]]

    etas = [float(x.split("=")[-1]) for x in args.configs.split(",")]
    print(f"探测帧: {probe}    τ={args.tau}  α′={args.alpha_pred}  R_max={args.r_max}")
    print(f"{'η':>6s} {'AUC(rnorm)':>11s} {'真实边均值':>11s} {'虚假边均值':>11s} "
          f"{'argmax命中率':>12s} {'eps_eff':>8s}")
    for eta in etas:
        scores, labels, argmax_hit, argmax_tot, eps_list = [], [], 0, 0, []
        for pos, t in enumerate(ts[:-1]):
            if t not in probe:
                continue
            tn = ts[pos + 1]
            sx, dx = dets.centroid(t), dets.centroid(tn)
            sl, dl = dets.label(t), dets.label(tn)
            if sx.shape[0] == 0 or dx.shape[0] == 0:
                continue
            art = compute_pairwise_plan(
                sx, dx,
                OTConfig(r_max=args.r_max, eta=eta, tau_a=args.tau, tau_b=args.tau,
                         alpha_pred=args.alpha_pred, eps_rel=args.eps_rel),
                dets.volume(t), dets.volume(tn),
                measure=MeasureConfig(mass_mode="uniform", knn_k=6))
            eps_list.append(art.eps_eff)
            rn = art.rnorm
            C = art.cost
            finite = np.isfinite(C)
            # 真实边：同一轨迹延续（move）或父→子（division）
            true = np.zeros_like(finite)
            for i in range(sl.size):
                for j in range(dl.size):
                    if not finite[i, j]:
                        continue
                    if int(sl[i]) == int(dl[j]) or parent.get(int(dl[j]), 0) == int(sl[i]):
                        true[i, j] = True
            # 只看 R_max 可达的配对（否则被 +inf 门限掩盖）
            for i in range(sl.size):
                row = np.where(finite[i])[0]
                if row.size == 0:
                    continue
                scores.extend(rn[i, row].tolist())
                labels.extend(true[i, row].tolist())
                j_star = row[np.argmax(art.plan[i, row])]
                argmax_tot += 1
                argmax_hit += int(true[i, j_star])
        scores = np.asarray(scores); labels = np.asarray(labels, dtype=bool)
        print(f"{eta:6.2f} {auc(scores, labels):11.4f} "
              f"{scores[labels].mean() if labels.any() else float('nan'):11.4f} "
              f"{scores[~labels].mean() if (~labels).any() else float('nan'):11.4f} "
              f"{argmax_hit / max(argmax_tot, 1):12.4f} {np.mean(eps_list):8.1f}")


if __name__ == "__main__":
    main()
