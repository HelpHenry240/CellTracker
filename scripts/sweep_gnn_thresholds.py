#!/usr/bin/env python3
"""GNN 决策阈值扫描：tau_move × tau_div。

动机：GNN 已把"误分裂"降了 79%，但预测轨迹数（913）仍多于 GT（720），
碎片化 301 是最小失分项。这里固定 GNN 权重（只做一次推理），
扫描"接受移动边/分裂边"的概率阈值，用本地指标快速选出折中点。
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_baseline import paint_and_write_stream  # noqa: E402

from celltracker.experiment import Experiment  # noqa: E402
from celltracker.gnn.infer import InferConfig, predict_pairs, reconstruct_tracks  # noqa: E402
from celltracker.track import Detections  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--h5", required=True)
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--tau-move", default="0.3,0.5,0.7")
    ap.add_argument("--tau-div", default="0.5,0.7,0.9")
    ap.add_argument("--dataset", default="Fluo-N3DH-CE")
    ap.add_argument("--seq", default="01")
    args = ap.parse_args()

    exp = Experiment(args.exp_id, purpose="GNN 决策阈值扫描",
                     params={"ckpt": args.ckpt, "graphs": args.graphs,
                             "tau_move": args.tau_move, "tau_div": args.tau_div})
    dets = Detections.from_h5(args.h5)
    ts = dets.t_range
    exp.log("推理一次，复用预测结果扫描阈值 ...")
    preds = predict_pairs(args.ckpt, args.graphs, device="cpu")

    rows = []
    for tm, td in itertools.product([float(x) for x in args.tau_move.split(",")],
                                    [float(x) for x in args.tau_div.split(",")]):
        res = reconstruct_tracks(dets, preds, InferConfig(tau_move=tm, tau_div=td))
        diag, _ = paint_and_write_stream(Path(args.h5), ts, dets, res.assignment,
                                         Path("/tmp/_gnn_sweep_res"), res.tracks,
                                         write_masks=False)
        row = {"tau_move": tm, "tau_div": td, "n_tracks": res.n_tracks(),
               "IDsw": diag["id_switches"], "frag": diag["fragmentation"],
               "div_P": round(diag.get("division_precision", float("nan")), 4),
               "div_R": round(diag.get("division_recall", float("nan")), 4),
               "div_pred": diag.get("division_pred"),
               "segments_split": res.meta.get("segments_split"),
               "parent_fixed": res.meta.get("parent_fixed")}
        rows.append(row)
        exp.log(f"tau_move={tm} tau_div={td}: tracks={row['n_tracks']} "
                f"IDsw={row['IDsw']} frag={row['frag']} "
                f"divP={row['div_P']} divR={row['div_R']}")

    keys = list(rows[0].keys())
    with (exp.dir / "sweep_results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    exp.save_metrics({"rows": rows})
    exp.finish(summary=f"{len(rows)} 组阈值")


if __name__ == "__main__":
    main()
