#!/usr/bin/env python3
"""按物理量纲标定论文未给数值的参数（AGENTS.md R3）。

输出每一项的**经验分布**与建议取值，并把结果写成 json 存档：

  · spacing_zyx          体素物理间距（µm）
  · sigma_x / sigma_s    式(6)(8) 的尺度量（正距离中位数 / 体积中位数）
  · r_max                式(8) 的位移上限（相邻帧最近邻位移的 p99.9）
  · eps                  式(12) 的 ε（0.1 × median(C)）
  · theta_gamma_frac     式(24)(26) 的 θ_Γ（真实边质量占比的低分位）
  · eta_death/eta_birth  式(24) 的行和/列和占比分布
  · div_ratio/vol_tol    分裂判据（真实分裂事件的显著质量比例 / 体积守恒偏差）

用法::

    python paperpipe/scripts/calibrate_params.py \
        --h5 data/interim/Fluo-N3DH-CE_01_pred_v2.h5 \
        --gt-h5 data/interim/Fluo-N3DH-CE_01.h5 \
        --out paperpipe/calib/ce01.json --pairs 40
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "paperpipe" / "src"))
sys.path.insert(0, str(ROOT / "src"))

from papertrack.config import load_config                        # noqa: E402
from papertrack.coupling import solve_coupling                   # noqa: E402
from papertrack.measure import pairwise_distance, resolve_spacing  # noqa: E402
from papertrack.pipeline import load_detections, load_gt_parent   # noqa: E402


def pct(v: np.ndarray, qs=(1, 5, 50, 95, 99.9)) -> dict:
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {}
    return {f"p{q}": float(np.percentile(v, q)) for q in qs} | {
        "n": int(v.size), "mean": float(v.mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--gt-h5", default=None)
    ap.add_argument("--config", default=str(ROOT / "paperpipe" / "configs" /
                                            "paper_default.yaml"))
    ap.add_argument("--pairs", type=int, default=40, help="抽样的相邻帧对数")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    dets = load_detections(args.h5)
    ts = dets.t_range
    spacing = resolve_spacing(cfg.measure.spacing_zyx, args.h5)
    gt_parent = load_gt_parent(args.gt_h5) if args.gt_h5 else {}

    rep: dict = {"h5": args.h5, "gt_h5": args.gt_h5, "n_frames": len(ts),
                 "spacing_zyx": list(spacing) if spacing else None}
    if spacing is None:
        rep["warning"] = "未解析到 spacing：以下 r_max/位移量级按体素单位"

    counts = np.array([dets.n(t) for t in ts], dtype=float)
    vols = np.concatenate([dets.volume(t) for t in ts])
    rep["instances_per_frame"] = pct(counts)
    rep["volume_voxels"] = pct(vols)
    rep["suggest"] = {"sigma_s": float(np.median(vols)),
                      "mass_mode": "volume"}

    # ---- 最近邻位移（候选运动的量级）----
    nn_d = []
    for t, tn in zip(ts[:-1], ts[1:]):
        if dets.n(t) == 0 or dets.n(tn) == 0:
            continue
        D = pairwise_distance(dets.centroid(t), dets.centroid(tn), spacing)
        nn_d.append(D.min(axis=1))
    nn = np.concatenate(nn_d) if nn_d else np.zeros(0)
    rep["nearest_neighbor_displacement_um"] = pct(nn)
    rep["suggest"]["r_max"] = float(np.percentile(nn, 99.9)) if nn.size else None
    rep["suggest"]["sigma_x_src"] = float(np.median(nn[nn > 0])) if np.any(nn > 0) else None

    # ---- 抽样求解耦合，统计 C 与 Γ ----
    idx = np.linspace(0, len(ts) - 2, min(args.pairs, max(len(ts) - 1, 1))).astype(int)
    C_all, eps_all = [], []
    true_frac, false_frac = [], []
    row_true, row_false = [], []          # 行和/a_i：有真实后继 vs 无真实后继
    div_rn, vol_devs, pos_frac = [], [], []   # 分裂行内占比 / 体积守恒偏差 / 正样本占比
    for pos in sorted(set(int(i) for i in idx)):
        t, tn = ts[pos], ts[pos + 1]
        if dets.n(t) == 0 or dets.n(tn) == 0:
            continue
        art = solve_coupling(dets.centroid(t), dets.centroid(tn), dets.volume(t),
                             dets.volume(tn), None, cfg.coupling, cfg.measure,
                             spacing)
        fin = art.cost[np.isfinite(art.cost)]
        C_all.append(fin)
        eps_all.append(0.1 * float(np.median(fin)))
        if not gt_parent:
            continue
        gl_s, gl_d = dets.gt_label(t), dets.gt_label(tn)
        frac = art.plan / np.maximum(art.mass_a[:, None], 1e-12)
        row_ratio = art.row_sum / np.maximum(art.mass_a, 1e-12)
        for i in range(dets.n(t)):
            if int(gl_s[i]) == 0:             # 假阳性源不参与阈值标定
                continue
            s_i = float(dets.volume(t)[i])
            # "有真实后继" = 同一 GT 轨迹（移动）**或** 某个 GT 子轨迹（分裂）
            succ = [j for j in range(dets.n(tn))
                    if int(gl_d[j]) != 0
                    and (int(gl_d[j]) == int(gl_s[i])
                         or int(gt_parent.get(int(gl_d[j]), 0)) == int(gl_s[i]))]
            (row_true if succ else row_false).append(float(row_ratio[i]))
            for j in succ:                    # 式(24) 的接受判据用**原始 Γ**（÷a_i）
                pos_frac.append(float(frac[i, j]))
            kids = [j for j in succ if int(gl_d[j]) != int(gl_s[i])]
            if len(kids) >= 2:                # 真实分裂：标定 div_ratio（行内占比口径）
                row = float(art.plan[i].sum())
                if row > 0:
                    div_rn.append(float(min(art.plan[i, j] for j in kids[:2])) / row)
                    vol_devs.append(
                        abs(sum(float(dets.volume(tn)[j]) for j in kids[:2]) - s_i)
                        / max(s_i, 1e-9))
            for j in range(dets.n(tn)):
                same = (int(gl_d[j]) == int(gl_s[i]))
                is_child = (int(gl_d[j]) != 0
                            and int(gt_parent.get(int(gl_d[j]), 0)) == int(gl_s[i]))
                (true_frac if (same or is_child) else false_frac).append(float(frac[i, j]))

    if C_all:
        rep["cost_C"] = pct(np.concatenate(C_all))
        rep["suggest"]["eps_rel"] = 0.1
        rep["suggest"]["eps_abs"] = float(np.median(eps_all))
    if true_frac:
        tf, ff = np.array(true_frac), np.array(false_frac)
        rep["gamma_frac_true_edges"] = pct(tf)
        rep["gamma_frac_false_edges"] = pct(ff)
        rep["suggest"]["theta_gamma_frac"] = float(max(np.percentile(tf, 5), 0.01))
        rep["suggest"]["theta_gamma_abs_note"] = (
            "绝对阈值 = frac × a_i（a_i 随帧内细胞数变化，故推荐用 frac 口径）")
    if row_true:
        rep["row_sum_over_mass_has_true_successor"] = pct(np.array(row_true))
    if row_false:
        rep["row_sum_over_mass_no_true_successor"] = pct(np.array(row_false))
    if row_true and row_false:
        # η_death：把"确实没有后继"的源判为死亡，同时不误伤有后继的源。
        # 取两类分布的分界（有后继者 p5 与无后继者 p95 的中点），夹在 [0.1, 0.9]。
        lo = float(np.percentile(np.array(row_true), 5))
        hi = float(np.percentile(np.array(row_false), 95))
        rep["suggest"]["eta_death"] = float(min(max((lo + hi) / 2.0, 0.1), 0.9))
        rep["suggest"]["eta_death_note"] = "有后继者行和占比 p5 与无后继者 p95 的中点（论文未给数值）"
    if pos_frac:
        rep["gamma_frac_true_successor_edges"] = pct(np.array(pos_frac))
    if div_rn:
        rep["division_row_normalized_fraction"] = pct(np.array(div_rn))
        rep["suggest"]["div_ratio"] = float(max(np.percentile(np.array(div_rn), 5), 0.05))
        rep["suggest"]["div_ratio_note"] = "真实分裂中较小子目标的**行内占比** p5"
    if vol_devs:
        rep["division_volume_deviation"] = pct(np.array(vol_devs))
        rep["suggest"]["vol_tol"] = float(np.percentile(np.array(vol_devs), 90))

    text = json.dumps(rep, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
