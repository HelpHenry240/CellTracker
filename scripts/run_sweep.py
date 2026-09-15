#!/usr/bin/env python3
"""参数扫描：在给定网格上批量运行 OT/FGW 追踪，汇总表格 + 热图。

示例::

    python scripts/run_sweep.py --h5 data/interim/Fluo-N3DH-CE_01.h5 \
        --dataset Fluo-N3DH-CE --seq 01 --exp-id E2.3_eta_eps_sweep \
        --x eta=0,0.2,0.5 --y eps=1,3,10 --official
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_baseline import load_gt_labels  # noqa: E402

from celltracker.cost.features import CostConfig  # noqa: E402
from celltracker.eval.ctc_io import write_result  # noqa: E402
from celltracker.eval.local_metrics import tracking_diagnostics  # noqa: E402
from celltracker.experiment import Experiment  # noqa: E402
from celltracker.track import Detections, paint_result  # noqa: E402
from celltracker.track.ot_tracker import OTTrackConfig, run_tracking_ot  # noqa: E402
from celltracker.viz.ot_plots import plot_metric_heatmap  # noqa: E402


def parse_axis(spec: str) -> tuple[str, list[str]]:
    name, vals = spec.split("=", 1)
    return name, vals.split(",")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--x", required=True, help="如 eta=0,0.2,0.5")
    ap.add_argument("--y", required=True, help="如 eps=1,3,10")
    ap.add_argument("--fixed", default="", help="额外固定参数, 如 tau=1.0,r_max=30")
    ap.add_argument("--limit-frames", type=int, default=None)
    ap.add_argument("--official", action="store_true")
    ap.add_argument("--cloud-gt-root", default=None)
    args = ap.parse_args()

    x_name, x_vals = parse_axis(args.x)
    y_name, y_vals = parse_axis(args.y)
    fixed = {}
    if args.fixed:
        for kv in args.fixed.split(","):
            k, v = kv.split("=")
            fixed[k] = v

    exp = Experiment(args.exp_id, purpose=f"参数扫描 {x_name}×{y_name}",
                     params={"dataset": args.dataset, "seq": args.seq,
                             "x": args.x, "y": args.y, "fixed": fixed,
                             "official_eval": args.official})

    dets_all = Detections.from_h5(args.h5)
    ts = dets_all.t_range
    if args.limit_frames:
        ts = ts[: args.limit_frames]
    dets = Detections({t: dets_all.frames[t] for t in ts})
    gt_labels = load_gt_labels(Path(args.h5), ts)

    rows = []
    values: dict[tuple, float] = {}
    for xv, yv in itertools.product(x_vals, y_vals):
        grid = {x_name: xv, y_name: yv, **fixed}
        cfg = OTTrackConfig(
            cost=CostConfig(alpha=float(grid.get("alpha", 1.0)),
                            beta=float(grid.get("beta", 0.0)),
                            r_max=float(grid.get("r_max", 30.0)),
                            mass_mode=grid.get("mass_mode", "uniform")),
            eta=float(grid.get("eta", 0.0)),
            eps=float(grid.get("eps", 1.0)),
            tau_a=(float(grid["tau"]) if "tau" in grid else None),
            tau_b=(float(grid["tau"]) if "tau" in grid else None),
            knn_k=int(float(grid.get("knn", 6))),
            theta_gamma=float(grid.get("theta_gamma", 0.2)),
            div_ratio=float(grid.get("div_ratio", 0.25)),
        )
        result = run_tracking_ot(dets, cfg)
        res_labels = {t: paint_result(gt_labels[t], dets.label(t), result.assignment[t])
                      for t in ts}
        diag = tracking_diagnostics(gt_labels, res_labels)
        row = {x_name: xv, y_name: yv, **fixed,
               "pred_tracks": result.n_tracks(), "FN": diag["fn"], "FP": diag["fp"],
               "id_switches": diag["id_switches"], "fragmentation": diag["fragmentation"],
               "det_recall": round(diag["detection_recall"], 4)}

        if args.official:
            import subprocess
            res_dir = exp.artifact_dir(f"sub_{x_name}{xv}_{y_name}{yv}") / f"{args.seq}_RES"
            write_result(res_labels, result.tracks, res_dir)
            cmd = [sys.executable, str(ROOT / "scripts" / "cloud_eval.py"),
                   "--res-dir", str(res_dir), "--dataset", args.dataset,
                   "--seq", args.seq,
                   "--out", str(exp.dir / "artifacts" / f"official_{x_name}{xv}_{y_name}{yv}.json")]
            if args.cloud_gt_root:
                cmd += ["--cloud-gt-root", args.cloud_gt_root]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            import json
            jpath = exp.dir / "artifacts" / f"official_{x_name}{xv}_{y_name}{yv}.json"
            tra = det = None
            if jpath.exists():
                jj = json.loads(jpath.read_text())
                tra, det = jj.get("TRA"), jj.get("DET")
            row["DET"] = det
            row["TRA"] = tra
            exp.log(f"{x_name}={xv} {y_name}={yv}: TRA={tra} DET={det} "
                    f"FN={diag['fn']} FP={diag['fp']} IDsw={diag['id_switches']}")
            if tra is not None:
                values[(xv, yv)] = float(tra)
        else:
            exp.log(f"{x_name}={xv} {y_name}={yv}: FN={diag['fn']} FP={diag['fp']} "
                    f"IDsw={diag['id_switches']} frag={diag['fragmentation']}")
            values[(xv, yv)] = -float(diag["id_switches"])
        rows.append(row)

    keys = list(rows[0].keys())
    csv_path = exp.dir / "sweep_results.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    exp.log(f"CSV -> {csv_path}")

    if values:
        plot_metric_heatmap({"x_label": x_name, "y_label": y_name}, values,
                            "TRA" if args.official else "ID switches (negated)",
                            exp.figure_path(f"heatmap_{x_name}_{y_name}"))

    exp.save_metrics({"rows": rows, "x": x_name, "y": y_name, "official": args.official})
    exp.finish(summary=f"{len(rows)} 组配置")


if __name__ == "__main__":
    main()
