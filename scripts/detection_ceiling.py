#!/usr/bin/env python3
"""C4 误差归属的第一级漏斗：**检测层天花板**。

现有 `pipeline_funnel.py` 从"S0 = R_max 可达的真实边"起算，隐含假设
"检测就是 GT"。真实前端（nnU-Net 预测实例）下，这个假设不成立：
很多真实关联边的端点在检测层就不存在，**在候选生成之前就已经输掉了**。

本脚本量化三级天花板（不依赖图与 GNN，纯 h5 计算）：

    U0  GT 节点（marker）在检测集中出现的比例          ← 缺检/粘连的直接后果
    U1  GT 移动边的**两端**都被检测到的比例
    U2  两端不仅被检测到，且落在**同一个预测实例**上的比例   ← 实例跨帧不一致
    U3  分裂边的父+两子都被检测到的比例

用法::

    python scripts/detection_ceiling.py \
        --pred-h5 data/interim/Fluo-N3DH-CE_01_pred.h5 \
        --gt-h5 data/interim/Fluo-N3DH-CE_01.h5 \
        --out experiments/C4_pred_graphs/detection_ceiling.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np


def load_gt(gt_h5: Path):
    """GT：逐帧 marker → 轨迹 id，以及 tracks 表的父子关系。"""
    frames: dict[int, np.ndarray] = {}
    with h5py.File(gt_h5, "r") as f:
        for key in f["frames"]:
            t = int(key)
            frames[t] = np.asarray(f["frames"][key]["label"], dtype=np.int64)
        tr = np.asarray(f["tracks"])
    parent = {int(l): int(p) for l, p in zip(tr["label"], tr["parent"])}
    start = {int(l): int(s) for l, s in zip(tr["label"], tr["begin"])}
    end = {int(l): int(e) for l, e in zip(tr["label"], tr["end"])}
    children: dict[int, list[int]] = defaultdict(list)
    for l, p in parent.items():
        if p:
            children[p].append(l)
    return frames, parent, start, end, children


def load_pred(pred_h5: Path):
    """预测：逐帧 (预测实例 id → GT id) 的映射与反向映射。"""
    inst2gt: dict[int, dict[int, int]] = {}
    gt2inst: dict[int, dict[int, set[int]]] = {}
    n_pred: dict[int, int] = {}
    with h5py.File(pred_h5, "r") as f:
        for key in f["frames"]:
            t = int(key)
            g = f["frames"][key]
            labels = np.asarray(g["label"], dtype=np.int64)
            # GT 的 h5 没有 gt_label 字段（标签即轨迹 id），回退到 label
            gt = (np.asarray(g["gt_label"], dtype=np.int64)
                  if "gt_label" in g else np.asarray(g["label"], dtype=np.int64))
            inst2gt[t] = {int(a): int(b) for a, b in zip(labels, gt)}
            rev: dict[int, set[int]] = {}
            for a, b in zip(labels, gt):
                if b > 0:
                    rev.setdefault(int(b), set()).add(int(a))
            gt2inst[t] = rev
            n_pred[t] = int(labels.size)
    return inst2gt, gt2inst, n_pred


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-h5", required=True)
    ap.add_argument("--gt-h5", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    gt_frames, parent, start, end, children = load_gt(Path(args.gt_h5))
    inst2gt, gt2inst, n_pred = load_pred(Path(args.pred_h5))
    ts = sorted(gt_frames)

    # ---- U0：GT 节点（marker）被检测到的比例 ----
    n_nodes = n_nodes_det = 0
    nodes_detected_frac_per_frame: dict[int, float] = {}
    for t in ts:
        ids = gt_frames[t]
        if ids.size == 0:
            continue
        det = set(gt2inst.get(t, {}).keys())
        hit = sum(1 for i in ids if int(i) in det)
        n_nodes += int(ids.size)
        n_nodes_det += int(hit)
        nodes_detected_frac_per_frame[t] = hit / ids.size

    # ---- U1/U2：GT 移动边 ----
    # 注意 CTC 的 tracks 表在**每次分裂处断开**（父轨迹的 end 停在分裂前），
    # 所以"移动边"= 同一轨迹 id 在相邻两帧的连续出现，而不是父子对。
    n_move = n_move_det = n_move_same_inst = 0
    # ---- U3：分裂边（父 + 两个子都要在） ----
    n_div = n_div_det = n_div_disjoint = 0
    per_frame_move: dict[int, dict[str, int]] = defaultdict(
        lambda: {"n": 0, "det": 0, "same": 0})

    for l in parent:
        for t in range(start[l], end[l]):
            ids_t = set(int(x) for x in gt_frames.get(t, np.zeros(0, dtype=np.int64)))
            ids_n = set(int(x) for x in gt_frames.get(t + 1, np.zeros(0, dtype=np.int64)))
            if l not in ids_t or l not in ids_n:
                continue
            n_move += 1
            per_frame_move[t]["n"] += 1
            a = gt2inst.get(t, {}).get(l, set())
            b = gt2inst.get(t + 1, {}).get(l, set())
            if a and b:
                n_move_det += 1
                per_frame_move[t]["det"] += 1
                if a & b:                 # 同一预测实例在相邻两帧代表该 GT 细胞
                    n_move_same_inst += 1
                    per_frame_move[t]["same"] += 1

    for p, kids in children.items():
        if len(kids) != 2:
            continue
        t_p, t_c = end[p], start[kids[0]]
        if t_c != t_p + 1 or any(start[k] != t_c for k in kids):
            continue
        ids_p = set(int(x) for x in gt_frames.get(t_p, np.zeros(0, dtype=np.int64)))
        ids_c = set(int(x) for x in gt_frames.get(t_c, np.zeros(0, dtype=np.int64)))
        if p not in ids_p or not all(k in ids_c for k in kids):
            continue
        n_div += 1
        p_inst = gt2inst.get(t_p, {}).get(p, set())
        kids_inst = [gt2inst.get(t_c, {}).get(k, set()) for k in kids]
        if p_inst and all(kids_inst):
            n_div_det += 1
            # 两个子必须落在**不同的**实例上，分裂才在检测层可见
            if not (kids_inst[0] & kids_inst[1]):
                n_div_disjoint += 1

    out = {
        "pred_h5": str(args.pred_h5),
        "gt_h5": str(args.gt_h5),
        "seq_frames": len(ts),
        "pred_instances_per_frame_mean": float(np.mean([n_pred.get(t, 0) for t in ts])),
        "gt_nodes_per_frame_mean": float(np.mean([gt_frames[t].size for t in ts])),
        "U0_gt_nodes_detected": {
            "n_gt_nodes": n_nodes,
            "n_detected": n_nodes_det,
            "fraction": n_nodes_det / max(n_nodes, 1),
        },
        "U1_move_edges_both_detected": {
            "n_gt_move_edges": n_move,
            "n_both_detected": n_move_det,
            "fraction": n_move_det / max(n_move, 1),
        },
        "U2_move_edges_same_instance": {
            "n_same_instance": n_move_same_inst,
            "fraction_of_gt": n_move_same_inst / max(n_move, 1),
            "fraction_of_detected": n_move_same_inst / max(n_move_det, 1),
        },
        "U3_division_edges_parent_and_children_detected": {
            "n_gt_division_parents": n_div,
            "n_all_detected": n_div_det,
            "fraction": n_div_det / max(n_div, 1),
            "n_children_on_distinct_instances": n_div_disjoint,
            "fraction_children_distinct": n_div_disjoint / max(n_div, 1),
        },
        "per_frame_move_ceiling": {
            str(t): per_frame_move[t] for t in sorted(per_frame_move)
        },
        "nodes_detected_fraction_first_50": float(
            np.mean([v for t, v in nodes_detected_frac_per_frame.items() if t < 50])),
        "nodes_detected_fraction_last_50": float(
            np.mean([v for t, v in nodes_detected_frac_per_frame.items()
                     if t >= max(ts) - 49])),
    }
    text = json.dumps(out, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
