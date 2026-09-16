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

from celltracker.eval.ctc_io import ResultWriter  # noqa: E402
from celltracker.eval.local_metrics import StreamingDiagnostics, seg_measure  # noqa: E402
from celltracker.experiment import Experiment  # noqa: E402
from celltracker.track import Detections, LinkerConfig, paint_result, run_tracking  # noqa: E402
from celltracker.viz import PALETTE, savefig  # noqa: E402


def iter_gt_labels(h5_path: Path, frames: list[int]):
    """逐帧产出 (t, labels)，避免一次性把全部 3D 标注读入内存。"""
    with h5py.File(h5_path, "r") as f:
        for t in frames:
            yield t, np.asarray(f[f"frames/{t:04d}/labels"])


def paint_and_write_stream(h5_path: Path, frames: list[int], dets: Detections,
                           assignment: dict[int, np.ndarray], res_dir: Path,
                           tracks, want_mip_frame: int | None = None,
                           write_masks: bool = True):
    """流式：逐帧画结果掩码 → 写盘 → 累积诊断；返回 (诊断, MIP 背景图)。"""
    writer = ResultWriter(res_dir) if write_masks else None
    diag = StreamingDiagnostics()
    mip = None
    with h5py.File(h5_path, "r") as f:
        for t in frames:
            gt = np.asarray(f[f"frames/{t:04d}/labels"])
            res = paint_result(gt, dets.label(t), assignment[t])
            if writer is not None:
                writer.add(t, res)
            diag.add_frame(t, gt, res)
            if want_mip_frame is not None and t == want_mip_frame:
                mip = np.asarray(f[f"frames/{t:04d}/labels"]).max(axis=0).copy()
            del gt, res
    if writer is not None:
        writer.close(tracks)
    return diag.result(), mip


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

    res_dir = exp.artifact_dir("submission") / f"{args.seq}_RES"
    mip_frame = ts[len(ts) // 2]
    diag, mip = paint_and_write_stream(Path(args.h5), ts, dets, result.assignment,
                                       res_dir, result.tracks, want_mip_frame=mip_frame)
    exp.log(f"提交目录 -> {res_dir}（{len(ts)} 帧，MIP 参考帧 {mip_frame}）")

    gt_seg_dir = ROOT / "data" / "raw" / args.dataset / f"{args.seq}_GT" / "SEG"
    sego = seg_measure(gt_seg_dir, res_dir) if gt_seg_dir.is_dir() else float("nan")

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

    _plot_diagnostics(exp, result, diag, args.dataset, args.seq)
    _plot_tracks(exp, dets, result.assignment, args.dataset, args.seq, mip=mip)

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


def _plot_diagnostics(exp: Experiment, result, diag, dataset, seq) -> None:
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


def _plot_tracks(exp: Experiment, dets: Detections, assignment: dict[int, np.ndarray],
                 dataset, seq, mip=None) -> None:
    """最大投影背景 + 轨迹连线（用检测质心，不需要读整卷）。"""
    import matplotlib.pyplot as plt

    ts = sorted(assignment)
    cents: dict[int, list[tuple[float, float]]] = {}
    for t in ts:
        xy = dets.centroid(t)
        for k, tid in enumerate(assignment[t]):
            cents.setdefault(int(tid), []).append((xy[k, -1], xy[k, -2]))

    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    if mip is not None:
        ax.imshow(mip, cmap="gray_r")
    for tid, pts in cents.items():
        if len(pts) < 3:
            continue
        pts = np.array(pts)
        ax.plot(pts[:, 0], pts[:, 1], lw=0.7, alpha=0.8)
    ax.set_title(f"{dataset} {seq} — predicted tracks (n={len(cents)})")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    savefig(fig, exp.figure_path("tracks_pred"))


if __name__ == "__main__":
    main()
