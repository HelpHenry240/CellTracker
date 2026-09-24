#!/usr/bin/env python3
"""候选覆盖率标定：式(26) 的 θ_Γ/ε/R_max 到底能让多少**真实关联**进入候选集。

为什么必须先做这一步（AGENTS.md R5：不可逆筛选必须保守）：
  §2.0.1 的 GNN 只在式(26) 筛出的候选边里做决策 —— 候选集之外的边**永远找不回来**。
  所以 ε（熵正则强度）与 R_max（位移门限）不是"精度旋钮"，而是决定"上界"的旋钮：
  ε 太小 → 传输计划退化成硬分配，一行只给最优目标质量，分裂的第二个子目标拿 0 质量；
  R_max 太小 → 真实位移超门限的配对直接置 +∞。

判据（与被评估的失败模式匹配，R14）：
  * `cand_recall`        : 真实后继边中，进入候选集（有限代价 且 Γ ≥ θ_Γ·a_i）的比例
  * `div_child_recall`   : 真实分裂事件中，**两个**子目标都进入候选集的比例
  * `cand_density`       : 候选边数 / 全部可能配对数（候选集越小越快，但不能牺牲召回）

用法::

    python paperpipe/scripts/check_candidate_coverage.py \
        --h5 data/interim/Fluo-N3DH-CE_01_pred_v2.h5 \
        --gt-h5 data/interim/Fluo-N3DH-CE_01.h5 --pairs 30
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parent
sys.path.insert(0, str(PKG / "src"))

import papertrack  # noqa: E402,F401  —— 触发 _paths（vendor 优先）

from papertrack.config import load_config, override                 # noqa: E402
from papertrack.coupling import solve_coupling                      # noqa: E402
from papertrack.graph.build import _candidates                      # noqa: E402
from papertrack.representation.measure import resolve_spacing       # noqa: E402
from papertrack.runtime.pipeline import load_detections, load_gt_parent  # noqa: E402


def coverage_for(dets, ts, idx, cfg, gt_parent, spacing, r_max, eps_rel, eta,
                 theta_frac, topk) -> dict:
    cfg_c = override(cfg, [f"coupling.r_max={r_max}", f"coupling.eps_rel={eps_rel}",
                           f"coupling.eta={eta}"])
    gcfg = override(cfg, [f"graph.theta_gamma_frac={theta_frac}",
                          f"graph.cand_topk={topk}",
                          f"coupling.r_max={r_max}"])
    n_true = n_cov = n_possible = n_cand = 0
    n_div = n_div_cov = 0
    for pos in idx:
        t, tn = ts[pos], ts[pos + 1]
        if dets.n(t) == 0 or dets.n(tn) == 0:
            continue
        art = solve_coupling(dets.centroid(t), dets.centroid(tn), dets.volume(t),
                             dets.volume(tn), None, cfg_c.coupling, cfg_c.measure,
                             spacing)
        pairs, _g, _c = _candidates(art, gcfg.graph, r_max, 1)
        cand = {(int(i), int(j)) for i, j in pairs}
        n_possible += dets.n(t) * dets.n(tn)
        n_cand += len(cand)
        gl_s, gl_d = dets.gt_label(t), dets.gt_label(tn)
        for i in range(dets.n(t)):
            if int(gl_s[i]) == 0:
                continue
            succ = [j for j in range(dets.n(tn))
                    if int(gl_d[j]) != 0
                    and (int(gl_d[j]) == int(gl_s[i])
                         or int(gt_parent.get(int(gl_d[j]), 0)) == int(gl_s[i]))]
            if not succ:
                continue
            n_true += len(succ)
            n_cov += sum(1 for j in succ if (i, j) in cand)
            kids = [j for j in succ if int(gl_d[j]) != int(gl_s[i])]
            if len(kids) >= 2:
                n_div += 1
                if all((i, j) in cand for j in kids[:2]):
                    n_div_cov += 1
    return {"r_max": r_max, "eps_rel": eps_rel, "eta": eta,
            "theta_gamma_frac": theta_frac, "cand_topk": topk,
            "cand_recall": round(n_cov / max(n_true, 1), 4),
            "div_child_recall": round(n_div_cov / max(n_div, 1), 4),
            "cand_density": round(n_cand / max(n_possible, 1), 4),
            "n_true_edges": n_true, "n_divisions": n_div, "n_candidates": n_cand}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--gt-h5", required=True)
    ap.add_argument("--config", default=str(ROOT / "paperpipe" / "configs" /
                                            "paper_default.yaml"))
    ap.add_argument("--pairs", type=int, default=30)
    ap.add_argument("--r-max", default="3.0,4.3,6.0")
    ap.add_argument("--eps-rel", default="0.1,0.3,1.0,3.0")
    ap.add_argument("--eta", default="0.3")
    ap.add_argument("--theta-frac", default="0.01")
    ap.add_argument("--topk", default="0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    dets = load_detections(args.h5)
    ts = dets.t_range
    spacing = resolve_spacing(cfg.measure.spacing_zyx, args.h5)
    gt_parent = load_gt_parent(args.gt_h5)
    idx = sorted(set(int(i) for i in np.linspace(
        0, len(ts) - 2, min(args.pairs, len(ts) - 1)).astype(int)))

    rows = []
    for r_max, eps_rel, eta, theta, topk in itertools.product(
            [float(x) for x in args.r_max.split(",")],
            [float(x) for x in args.eps_rel.split(",")],
            [float(x) for x in args.eta.split(",")],
            [float(x) for x in args.theta_frac.split(",")],
            [int(x) for x in args.topk.split(",")]):
        rows.append(coverage_for(dets, ts, idx, cfg, gt_parent, spacing,
                                 r_max, eps_rel, eta, theta, topk))
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=2))
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
