#!/usr/bin/env python3
"""全链路误差漏斗：定位误差在 pipeline 的哪一环进入。

动机：逐模块调参只能回答"这个改动有没有用"，回答不了"误差主要在哪一环丢的"。
本脚本把每条**真实关联边**从 GT 追溯到最终轨迹，计算逐级留存率：

    S0  GT 真实边（R_max 可达）
    S1  ├─ 式(26) 候选集是否保留        ← 不可逆：丢掉就永远找不回
    S2  ├─ 决策（OT / GNN / 融合）是否选中正确目标
    S3  └─ 轨迹重建后是否仍属同一轨迹    ← 冲突消解、分裂判定的影响

输出每一级的留存率，一眼看出瓶颈在 S1（候选生成）、S2（模型判定）
还是 S3（后处理/重建）。

用法::

    python scripts/pipeline_funnel.py --h5 data/interim/Fluo-N3DH-CE_01.h5 \
        --graphs data/interim/graphs_ce01_otcand --seq 01 \
        --ckpt experiments/E4.8_gnn_otcand_ce01/artifacts/model/best.pt --lam 0.5
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from celltracker.cost.features import CostConfig, build_cost  # noqa: E402
from celltracker.gnn.infer import InferConfig, predict_pairs, reconstruct_tracks  # noqa: E402
from celltracker.gnn.fusion import move_division_scores  # noqa: E402
from celltracker.track import Detections  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--gt-h5", default=None,
                    help="GT 轨迹表所在的 h5（检测来自预测时必填；默认与 --h5 相同）")
    ap.add_argument("--graphs", required=True)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--ckpt", default=None, help="GNN 权重；不给则只统计 OT 规则")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--fusion", action="store_true",
                   help="消融：使用 λ 加权融合口径统计（默认论文口径）")
    ap.add_argument("--tau-move", type=float, default=0.5)
    ap.add_argument("--tau-div", type=float, default=0.5)
    ap.add_argument("--r-max", type=float, default=30.0)
    ap.add_argument("--spacing-zyx", default=None,
                    help="(z,y,x) µm 间距；给定时 r_max 的单位是 µm（与 pipeline 一致）")
    ap.add_argument("--limit-pairs", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    spacing = (tuple(float(x) for x in args.spacing_zyx.split(","))
               if args.spacing_zyx else None)
    dets = Detections.from_h5(args.h5)
    gt_h5 = args.gt_h5 or args.h5
    with h5py.File(gt_h5, "r") as f:
        gt = np.asarray(f["tracks"])
        shape = np.asarray([int(x) for x in f.attrs["shape"]], dtype=float)
    parent = {int(l): int(p) for l, p in zip(gt["label"], gt["parent"])}
    ts = dets.t_range

    # 预测（若给了 GNN 权重）
    preds = (predict_pairs(args.ckpt, args.graphs, device="cpu")
             if args.ckpt else None)
    result = reconstruct_tracks(dets, preds, InferConfig(
        fusion=args.fusion, lam=args.lam,
        tau_move=args.tau_move, tau_div=args.tau_div)) if preds else None

    tot = {0: 0, 1: 0, 2: 0}          # S0：move / div / none 总数
    cand = {0: 0, 1: 0, 2: 0}         # S1：候选集保留
    decided = {1: 0, 2: 0}            # S2：决策选中
    survived = {1: 0, 2: 0}           # S3：重建后仍在同一轨迹

    pair_files = sorted(Path(args.graphs).glob("pair_*.npz"))
    if args.limit_pairs:
        pair_files = pair_files[: args.limit_pairs]

    for pf in pair_files:
        d = np.load(pf)
        t = int(d["t"])
        tn = int(d["t_next"])
        if tn not in ts:
            continue
        # 真实身份用 gt_label（预测 h5 通过重叠映射提供，未匹配为 0；
        # GT h5 无该字段时自动回退到 label）——否则预测实例 id 会被当成轨迹 id。
        sl, dl = dets.gt_label(t), dets.gt_label(tn)
        n_src, n_dst = sl.size, dl.size
        if n_src == 0 or n_dst == 0:
            continue
        # S0：R_max 可达的全部真实边
        C, _ = build_cost(dets.centroid(t), dets.centroid(tn), None, None, None,
                          CostConfig(r_max=args.r_max, spacing_zyx=spacing))
        finite = np.isfinite(C)
        gt_true = {}
        for i in range(n_src):
            for j in range(n_dst):
                if not finite[i, j]:
                    continue
                if int(sl[i]) == int(dl[j]):
                    tot[1] += 1
                    gt_true[(i, j)] = 1
                elif parent.get(int(dl[j]), 0) == int(sl[i]):
                    tot[2] += 1
                    gt_true[(i, j)] = 2
        # S1：候选集
        cand_keys = {(int(i), int(j)): int(lab)
                     for (i, j), lab in zip(d["cand_edges"], d["cand_label"])}
        for key, lab in cand_keys.items():
            if lab in (1, 2):
                cand[lab] += 1
        # S2：决策是否选中正确目标（按该源的 argmax 规则）
        if preds is not None and t in preds:
            p = preds[t]
            feat, prob, pairs = p["feat"], p["prob"], p["pairs"]
            if args.fusion:
                e_link, e_div = move_division_scores(feat[:, 8], prob[:, 1], prob[:, 2],
                                                     args.lam)
            else:
                e_link, e_div = prob[:, 1], prob[:, 2]
            best_link = defaultdict(lambda: (-1, -1.0))
            div_by_src = defaultdict(list)
            for k, (i, j) in enumerate(pairs):
                i, j = int(i), int(j)
                if e_link[k] > best_link[i][1]:
                    best_link[i] = (j, float(e_link[k]))
                div_by_src[i].append((j, float(e_div[k])))
            for (i, j), lab in gt_true.items():
                if lab == 1 and best_link.get(i, (-1, -1))[0] == j:
                    decided[1] += 1
                if lab == 2:
                    # 分裂边：该源被判为分裂且选中了该目标
                    cands = sorted(div_by_src.get(i, []), key=lambda x: -x[1])[:2]
                    if len(cands) >= 2 and any(cj == j for cj, _ in cands):
                        decided[2] += 1
            # S3：重建后仍在同一轨迹
            if result is not None:
                a0 = result.assignment.get(t)
                a1 = result.assignment.get(tn)
                if a0 is not None and a1 is not None:
                    for (i, j), lab in gt_true.items():
                        if lab == 1 and int(a0[i]) == int(a1[j]):
                            survived[1] += 1
                        if lab == 2 and int(a1[j]) in result.tracks and \
                                result.tracks[int(a1[j])].parent == int(a0[i]):
                            survived[2] += 1

    def pct(a, b):
        return f"{a / b * 100:6.1f}%" if b else "   n/a"

    # ---- 事件级统计（分裂必须"两个子目标都对"才算成功）----
    # 逐条边的统计会高估分裂的表现：一次分裂有 2 条边，只命中 1 条也会记 50%，
    # 但对 AOGM 而言这仍是"缺失边"。事件级才是与最终指标对齐的口径。
    ev = {"gt": 0, "cand": 0, "decided": 0, "survived": 0}
    for pf in pair_files:
        d = np.load(pf)
        t, tn = int(d["t"]), int(d["t_next"])
        if tn not in ts:
            continue
        sl, dl = dets.gt_label(t), dets.gt_label(tn)
        C, _ = build_cost(dets.centroid(t), dets.centroid(tn), None, None, None,
                          CostConfig(r_max=args.r_max, spacing_zyx=spacing))
        finite = np.isfinite(C)
        events: dict[int, list[int]] = defaultdict(list)
        for i in range(sl.size):
            for j in range(dl.size):
                if finite[i, j] and parent.get(int(dl[j]), 0) == int(sl[i]):
                    events[i].append(j)
        cand_pairs = {(int(i), int(j)) for i, j in d["cand_edges"]}
        for i, js in events.items():
            if len(js) < 2:
                continue
            ev["gt"] += 1
            if all((i, j) in cand_pairs for j in js):
                ev["cand"] += 1
            if preds is not None and t in preds:
                p = preds[t]
                feat, prob, pairs = p["feat"], p["prob"], p["pairs"]
                if args.fusion:
                    _el, e_div = move_division_scores(feat[:, 8], prob[:, 1], prob[:, 2],
                                                      args.lam)
                else:
                    e_div = prob[:, 2]
                rows = {int(a): (int(b), float(e_div[k]))
                        for k, (a, b) in enumerate(pairs)}
                cand_list = sorted([(int(b), float(e_div[k]))
                                    for k, (a, b) in enumerate(pairs) if int(a) == i],
                                   key=lambda x: -x[1])[:2]
                if all(any(cj == j for cj, _ in cand_list) for j in js):
                    ev["decided"] += 1
                if result is not None and t in result.assignment and tn in result.assignment:
                    a0, a1 = result.assignment[t], result.assignment[tn]
                    if all(int(a1[j]) in result.tracks and
                           result.tracks[int(a1[j])].parent == int(a0[i]) for j in js):
                        ev["survived"] += 1

    rows = [
        ("S0 GT 真实边（R_max 可达）", tot[1], tot[2]),
        ("S1 式(26) 候选集保留", cand[1], cand[2]),
        ("S2 决策选中正确目标", decided[1], decided[2]),
        ("S3 重建后仍在同一轨迹", survived[1], survived[2]),
    ]
    print(f"序列 {args.seq}  GNN={'有' if preds else '无'}  λ={args.lam}"
          f"  τ_move={args.tau_move} τ_div={args.tau_div} R_max={args.r_max}")
    print(f"{'阶段':32s} {'move 边':>16s} {'division 边':>18s}")
    for name, m, dv in rows:
        print(f"{name:32s} {m:6d} ({pct(m, tot[1])}) {dv:6d} ({pct(dv, tot[2])})")
    if ev["gt"]:
        print(f"\n【事件级·分裂】共 {ev['gt']} 次真实分裂事件（要求两个子目标同时正确）")
        for key, label in (("cand", "S1 两个子目标都在候选集"),
                           ("decided", "S2 两个子目标都进 top-2"),
                           ("survived", "S3 两个子目标都连上父轨迹")):
            print(f"  {label:32s} {ev[key]:5d} ({pct(ev[key], ev['gt'])})")
    out_ev = ev

    out = {"seq": args.seq, "lam": args.lam, "tot": tot, "cand": cand,
           "decided": decided, "survived": survived, "events": out_ev}
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
