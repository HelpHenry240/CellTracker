#!/usr/bin/env python3
"""跑完整 pipeline 并评测（本地诊断 + 可选云端官方指标）。

这是 Phase B 的主力脚本：同一份配置既可用于主实验，也可用于消融矩阵
（配合 `--ablate`）。产物：CTC 提交目录、本地指标、可选官方指标、图。

示例::

    # 论文口径（GNN 决策）
    python scripts/eval_pipeline.py --h5 data/interim/Fluo-N3DH-CE_01.h5 \
        --dataset Fluo-N3DH-CE --seq 01 --exp-id B1_gnn_ce01 \
        --ckpt experiments/B1_gnn_paper_ce01/artifacts/model/best.pt \
        --config configs/pipeline_default.yaml --official

    # 消融：关掉 FGW 与多尺度
    python scripts/eval_pipeline.py ... --ablate fgw,multiscale
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from celltracker.eval.ctc_io import ResultWriter  # noqa: E402
from celltracker.eval.local_metrics import StreamingDiagnostics, seg_measure  # noqa: E402
from celltracker.experiment import Experiment  # noqa: E402
from celltracker.gnn.adapter import make_gnn_runner  # noqa: E402
from celltracker.gnn.infer import InferConfig  # noqa: E402
from celltracker.pipeline import (PipelineConfig, apply_ablation, load_config,  # noqa: E402
                                  run_pipeline, save_config)
from celltracker.track.base import paint_result  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--config", default=str(ROOT / "configs" / "pipeline_default.yaml"))
    ap.add_argument("--ablate", default="")
    ap.add_argument("--ckpt", default=None, help="GNN 权重；不给则走 §1.6 OT 规则")
    ap.add_argument("--frames", default=None, help="帧范围，如 120:190（冒烟用）")
    ap.add_argument("--official", action="store_true", help="调用云端官方 DTI/SEG/TRA")
    ap.add_argument("--cloud-gt-root", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.ablate:
        cfg = apply_ablation(cfg, [s for s in args.ablate.split(",") if s])

    frames = None
    if args.frames:
        lo, _, hi = args.frames.partition(":")
        frames = list(range(int(lo), int(hi) + 1))

    exp = Experiment(args.exp_id, purpose="pipeline 评测"
                     + (f"（消融 {args.ablate}）" if args.ablate else ""),
                     params={"h5": args.h5, "ckpt": args.ckpt,
                             "ablate": args.ablate, "frames": args.frames})
    save_config(cfg, exp.dir / "pipeline_config.yaml")

    runner = None
    if args.ckpt:
        runner = make_gnn_runner(args.ckpt, InferConfig(), device="cpu",
                                 artifacts_dir=exp.artifact_dir("gnn"))
    run = run_pipeline(args.h5, cfg, frames=frames,
                       artifacts_dir=exp.artifact_dir("pipeline"), gnn=runner)
    exp.log(f"pipeline info: {run.info}")

    # ---- 流式写 CTC 提交并累积诊断（3D 数据必须逐帧）----
    dets = run.dets
    ts = dets.t_range
    res_dir = exp.artifact_dir("submission") / f"{args.seq}_RES"
    writer = ResultWriter(res_dir)
    with h5py.File(args.h5, "r") as f:
        gt_tracks = np.asarray(f["tracks"]) if "tracks" in f else None
        # 传入 GT 血缘，才能计算分裂事件的精确率/召回率
        diag = StreamingDiagnostics(gt_tracks=gt_tracks)
        for t in ts:
            gt = np.asarray(f[f"frames/{t:04d}/labels"])
            key = ts.index(t) if t != ts[0] else 0
            ids = run.track_result.assignment.get(t)
            if ids is None:
                continue
            res = paint_result(gt, dets.label(t), ids)
            writer.add(t, res)
            diag.add_frame(t, gt, res)
            del gt, res
    writer.close(run.track_result.tracks)

    diag_stats = diag.result()
    diag_stats.update(diag.division_pr(run.track_result.tracks))
    gt_seg_dir = ROOT / "data" / "raw" / args.dataset / f"{args.seq}_GT" / "SEG"
    sego = seg_measure(gt_seg_dir, res_dir) if gt_seg_dir.is_dir() else float("nan")
    metrics = {**diag_stats, "experiment": args.exp_id, "dataset": args.dataset,
               "seq": args.seq, "decision": run.info.get("decision"),
               "ablated": run.info.get("ablated"),
               "n_tracks_pred": run.track_result.n_tracks(),
               "local_SEG": sego, "pipeline_info": run.info}
    exp.save_metrics(metrics)
    exp.log(f"tracks={run.track_result.n_tracks()} FN={diag_stats['fn']} "
            f"FP={diag_stats['fp']} IDsw={diag_stats['id_switches']} "
            f"frag={diag_stats['fragmentation']} "
            f"divP={diag_stats.get('division_precision')} "
            f"divR={diag_stats.get('division_recall')}")

    if args.official:
        import subprocess
        cmd = [sys.executable, str(ROOT / "scripts" / "cloud_eval.py"),
               "--res-dir", str(res_dir), "--dataset", args.dataset, "--seq", args.seq,
               "--out", str(exp.dir / f"metrics_official_{args.seq}.json")]
        if args.cloud_gt_root:
            cmd += ["--cloud-gt-root", args.cloud_gt_root]
        exp.log("官方指标: " + " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        exp.log((proc.stdout or "")[-1500:])
        if proc.returncode != 0:
            exp.log("官方指标失败: " + (proc.stderr or "")[-800:])

    exp.finish(summary=f"tracks={run.track_result.n_tracks()} "
                       f"decision={run.info.get('decision')}")


if __name__ == "__main__":
    main()
