#!/usr/bin/env python3
"""P2：基于（非平衡）最优传输的追踪实验脚本。

示例::

    python scripts/run_ot.py --h5 data/interim/Fluo-N3DH-CHO_01.h5 \
        --dataset Fluo-N3DH-CHO --seq 01 --exp-id E2.1_ot_basic \
        --eps 1.0 --eta 0.0 --r-max 30 --official
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from run_baseline import _plot_diagnostics, _plot_tracks, paint_and_write_stream  # noqa: E402

from celltracker.cost.features import CostConfig  # noqa: E402
from celltracker.eval.local_metrics import seg_measure  # noqa: E402
from celltracker.experiment import Experiment  # noqa: E402
from celltracker.track import Detections, paint_result  # noqa: E402
from celltracker.track.ot_tracker import OTTrackConfig, run_tracking_ot  # noqa: E402
from celltracker.track.tracklets import TrackletMergeConfig, merge_tracklets  # noqa: E402

import h5py  # noqa: E402
import numpy as np  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--eps", type=float, default=1.0)
    ap.add_argument("--eta", type=float, default=0.0, help="结构项权重（0=纯 OT）")
    ap.add_argument("--tau", type=float, default=None, help="KL 松弛系数（越大越接近平衡）")
    ap.add_argument("--r-max", type=float, default=30.0)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--knn", type=int, default=6)
    ap.add_argument("--theta-gamma", type=float, default=0.2)
    ap.add_argument("--div-ratio", type=float, default=0.25)
    ap.add_argument("--div-sum-min", type=float, default=0.6)
    ap.add_argument("--velocity", action="store_true")
    ap.add_argument("--velocity-weight", type=float, default=1.0)
    ap.add_argument("--mass-mode", default="uniform", choices=["uniform", "volume"])
    ap.add_argument("--limit-frames", type=int, default=None)
    ap.add_argument("--official", action="store_true")
    ap.add_argument("--merge-tracklets", action="store_true",
                    help="启用二层 tracklet 合并（P3.4）")
    ap.add_argument("--merge-max-gap", type=int, default=3)
    ap.add_argument("--merge-rmax", type=float, default=30.0)
    ap.add_argument("--cloud-gt-root", default=None)
    args = ap.parse_args()

    cfg = OTTrackConfig(
        cost=CostConfig(alpha=args.alpha, beta=args.beta, r_max=args.r_max,
                        mass_mode=args.mass_mode),
        eta=args.eta, eps=args.eps, tau_a=args.tau, tau_b=args.tau,
        knn_k=args.knn, theta_gamma=args.theta_gamma, div_ratio=args.div_ratio,
        div_sum_min=args.div_sum_min,
        use_velocity=args.velocity, velocity_weight=args.velocity_weight,
        mass_mode=args.mass_mode,
    )
    exp = Experiment(args.exp_id, purpose=f"OT 追踪 eps={args.eps} eta={args.eta} "
                                          f"tau={args.tau}",
                     params={"dataset": args.dataset, "seq": args.seq,
                             **{k: v for k, v in cfg.__dict__.items() if k != "cost"},
                             "cost": cfg.cost.__dict__})

    dets = Detections.from_h5(args.h5)
    ts = dets.t_range
    if args.limit_frames:
        ts = ts[: args.limit_frames]
        dets = Detections({t: dets.frames[t] for t in ts})
    exp.log(f"frames={len(ts)} mean objects/frame="
            f"{np.mean([dets.n(t) for t in ts]):.1f}")

    result = run_tracking_ot(dets, cfg)
    exp.log(f"predicted tracks={result.n_tracks()}")
    if args.merge_tracklets:
        result = merge_tracklets(dets, result,
                                 TrackletMergeConfig(max_gap=args.merge_max_gap,
                                                     r_max=args.merge_rmax))
        exp.log(f"二层合并后 tracks={result.n_tracks()} "
                f"(merged_pairs={result.meta.get('merged_pairs')})")

    res_dir = exp.artifact_dir("submission") / f"{args.seq}_RES"
    mip_frame = ts[len(ts) // 2]
    diag, mip = paint_and_write_stream(Path(args.h5), ts, dets, result.assignment,
                                       res_dir, result.tracks, want_mip_frame=mip_frame)
    exp.log(f"提交目录 -> {res_dir}（{len(ts)} 帧）")

    gt_seg_dir = Path("data/raw") / args.dataset / f"{args.seq}_GT" / "SEG"
    sego = seg_measure(gt_seg_dir, res_dir) if gt_seg_dir.is_dir() else float("nan")
    with h5py.File(args.h5, "r") as f:
        gt_tracks = np.asarray(f["tracks"]) if "tracks" in f else np.zeros(0, dtype=[("parent", "i4")])
    metrics = {
        **diag,
        "experiment": args.exp_id, "dataset": args.dataset, "seq": args.seq,
        "n_frames": len(ts), "n_tracks_pred": result.n_tracks(),
        "gt_tracks": {"n_tracks": int(len(gt_tracks)),
                      "n_divisions": int(np.count_nonzero(gt_tracks["parent"] > 0))},
        "local_SEG": sego,
    }
    exp.save_metrics(metrics)
    exp.log(f"matched={diag['matched']} FN={diag['fn']} FP={diag['fp']} "
            f"IDsw={diag['id_switches']} frag={diag['fragmentation']} "
            f"detR={diag['detection_recall']:.3f} pred_tracks={result.n_tracks()}")

    _plot_diagnostics(exp, result, diag, args.dataset, args.seq)
    _plot_tracks(exp, dets, result.assignment, args.dataset, args.seq, mip=mip)

    if args.official:
        import subprocess
        cmd = [sys.executable, str(Path("scripts/cloud_eval.py").resolve()),
               "--res-dir", str(res_dir), "--dataset", args.dataset, "--seq", args.seq,
               "--out", str(exp.dir / f"metrics_official_{args.seq}.json")]
        if args.cloud_gt_root:
            cmd += ["--cloud-gt-root", args.cloud_gt_root]
        exp.log("官方指标: " + " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        exp.log((proc.stdout or "")[-1500:])
        if proc.returncode != 0:
            exp.log("官方指标失败: " + (proc.stderr or "")[-800:])

    exp.finish(summary=f"FN={diag['fn']} FP={diag['fp']} IDsw={diag['id_switches']}")


if __name__ == "__main__":
    main()
