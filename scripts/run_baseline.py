#!/usr/bin/env python3
"""P1 经典基线：贪心最近邻 / 匈牙利 / 匈牙利+匀速先验。

流程：内部 h5 检测 → 连接 → 画结果掩码 → 写 CTC 提交目录 → 本地诊断指标
（+ 可选云端官方 DET/SEG/TRA）→ 出图 + 实验记录。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from celltracker.eval.ctc_io import write_result  # noqa: E402
from celltracker.eval.local_metrics import seg_measure, tracking_diagnostics  # noqa: E402
from celltracker.experiment import Experiment  # noqa: E402
from celltracker.track import Detections, LinkerConfig, paint_result, run_tracking  # noqa: E402
from celltracker.viz import PALETTE, savefig  # noqa: E402


def load_gt_labels(h5_path: Path, frames: list[int]) -> dict[int, np.ndarray]:
    out = {}
    with h5py.File(h5_path, "r") as f:
        for t in frames:
            out[t] = np.asarray(f[f"frames/{t:04d}/labels"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--method", default="hungarian", choices=["greedy", "hungarian"])
    ap.add_argument("--max-dist", type=float, default=30.0)
    ap.add_argument("--velocity", action="store_true")
    ap.add_argument("--velocity-weight", type=float, default=1.0)
    ap.add_argument("--no-division", action="store_true")
    ap.add_argument("--division-max-dist", type=float, default=45.0)
    ap.add_argument("--limit-frames", type=int, default=None)
    ap.add_argument("--official", action="store_true", help="额外调用云端官方指标")
    ap.add_argument("--cloud-gt-root", default=None)
    args = ap.parse_args()

    cfg = LinkerConfig(method=args.method, max_dist=args.max_dist,
                       use_velocity=args.velocity, velocity_weight=args.velocity_weight,
                       detect_division=not args.no_division,
                       division_max_dist=args.division_max_dist)
    tag = args.method + ("+vel" if args.velocity else "")
    exp = Experiment(args.exp_id, purpose=f"经典基线 {tag}",
                     params={"dataset": args.dataset, "seq": args.seq, **cfg.__dict__,
                             "official_eval": args.official, "limit_frames": args.limit_frames})

    dets = Detections.from_h5(args.h5)
    ts = dets.t_range
    if args.limit_frames:
        ts = ts[: args.limit_frames]
        dets = Detections({t: dets.frames[t] for t in ts})
    exp.log(f"frames={len(ts)} mean objects/frame="
            f"{np.mean([dets.n(t) for t in ts]):.1f} cfg={cfg.__dict__}")

    result = run_tracking(dets, cfg)
    exp.log(f"predicted tracks={result.n_tracks()}")

    gt_labels = load_gt_labels(Path(args.h5), ts)
    res_labels = {t: paint_result(gt_labels[t], dets.label(t), result.assignment[t])
                  for t in ts}
    res_dir = exp.artifact_dir("submission") / f"{args.seq}_RES"
    write_result(res_labels, result.tracks, res_dir)

    gt_seg_dir = ROOT / "data" / "raw" / args.dataset / f"{args.seq}_GT" / "SEG"
    sego = seg_measure(gt_seg_dir, res_dir) if gt_seg_dir.is_dir() else float("nan")
    diag = tracking_diagnostics(gt_labels, res_labels)

    with h5py.File(args.h5, "r") as f:
        gt_tracks = np.asarray(f["tracks"]) if "tracks" in f else np.zeros(0, dtype=[("parent", "i4")])
    metrics = {
        **diag,
        "experiment": args.exp_id, "dataset": args.dataset, "seq": args.seq,
        "method": tag, "config": cfg.__dict__, "n_frames": len(ts),
        "n_tracks_pred": result.n_tracks(),
        "gt_tracks": {"n_tracks": int(len(gt_tracks)),
                      "n_divisions": int(np.count_nonzero(gt_tracks["parent"] > 0))},
        "local_SEG": sego,
    }
    exp.save_metrics(metrics)
    exp.log(f"SEG={sego:.4f} matched={diag['matched']} FN={diag['fn']} FP={diag['fp']} "
            f"IDsw={diag['id_switches']} frag={diag['fragmentation']} "
            f"detR={diag['detection_recall']:.3f} detP={diag['detection_precision']:.3f}")

    _plot_diagnostics(exp, res_labels, result, diag, args.dataset, args.seq)
    _plot_tracks(exp, gt_labels, res_labels, args.dataset, args.seq)

    if args.official:
        import subprocess
        cmd = [sys.executable, str(ROOT / "scripts" / "cloud_eval.py"),
               "--res-dir", str(res_dir), "--dataset", args.dataset, "--seq", args.seq,
               "--out", str(exp.dir / f"metrics_official_{args.seq}.json")]
        if args.cloud_gt_root:
            cmd += ["--cloud-gt-root", args.cloud_gt_root]
        exp.log("官方指标: " + " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        exp.log((proc.stdout or "")[-2000:])
        if proc.returncode != 0:
            exp.log("官方指标失败: " + (proc.stderr or "")[-1000:])

    exp.finish(summary=f"FN={diag['fn']} FP={diag['fp']} IDsw={diag['id_switches']} "
                       f"frag={diag['fragmentation']}")


def _plot_diagnostics(exp: Experiment, res_labels, result, diag, dataset, seq) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 2.9))
    axes[0].bar(["matched", "FN", "FP"],
                [diag["matched"], diag["fn"], diag["fp"]],
                color=[PALETTE["green"], PALETTE["red"], PALETTE["orange"]])
    axes[0].set_ylabel("objects (all frames)")
    axes[0].set_title(f"detection outcome | IDsw={diag['id_switches']} "
                      f"frag={diag['fragmentation']}")

    axes[1].bar(["GT tracks", "pred tracks"],
                [diag["gt_tracks"], len(result.tracks)],
                color=[PALETTE["blue"], PALETTE["purple"]])
    axes[1].set_title("track counts")
    fig.suptitle(f"{dataset} seq {seq} — {exp.exp_id}", y=1.03)
    savefig(fig, exp.figure_path("diagnostics"))


def _plot_tracks(exp: Experiment, gt_labels, res_labels, dataset, seq) -> None:
    """最大投影 + 轨迹连线（GT 与预测各一张）。"""
    import matplotlib.pyplot as plt

    ts = sorted(res_labels)
    mid = ts[len(ts) // 2]
    for name, labels in (("gt", gt_labels), ("pred", res_labels)):
        fig, ax = plt.subplots(figsize=(4.6, 4.0))
        ax.imshow(labels[mid].max(axis=0), cmap="gray_r")
        cents: dict[int, list[tuple[float, float]]] = {}
        for t in ts:
            v = labels[t]
            for lab in np.unique(v):
                if lab == 0:
                    continue
                idx = np.nonzero(v == lab)
                cents.setdefault(int(lab), []).append((idx[2].mean(), idx[1].mean()))
        for pts in cents.values():
            if len(pts) < 3:
                continue
            pts = np.array(pts)
            ax.plot(pts[:, 0], pts[:, 1], lw=0.8, alpha=0.85)
        ax.set_title(f"{dataset} {seq} — {name} (MIP frame {mid})")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        savefig(fig, exp.figure_path(f"tracks_{name}"))


if __name__ == "__main__":
    main()
