#!/usr/bin/env python3
"""C2 标定：用 GT 标记点选择实例拆分参数（`min_distance` / `min_volume`）。

思路：GT 的 `_GT/TRA` 标记点给出"该帧有多少细胞、在哪"，
因此可以量化每个候选参数下的**检测召回/精确率**，选择最优点
（而不是靠肉眼看几张图）。

用法::

    python scripts/calibrate_instance_split.py \
        --pred-dir /tmp/preds_sample --img-root data/raw/Fluo-N3DH-CE --seq 01 \
        --gt-root data/raw/Fluo-N3DH-CE \
        --min-distance 2,3,4,5 --min-volume 20,50

留痕约定（C0）：扫描结果用 `--save-json` 落档到实验目录的 `metrics.json`；
帧抽样用 `--frame-stride`（早期帧稀疏、后期帧密集，只取前 N 帧会系统性高估精确率）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import tifffile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from celltracker.detect.instances import (InstanceSplitConfig,  # noqa: E402
                                          detection_recall_vs_markers,
                                          split_instances)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--img-root", required=True)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--gt-root", required=True)
    ap.add_argument("--min-distance", default="2,3,4,5")
    ap.add_argument("--min-volume", default="30")
    ap.add_argument("--h-frac", default="0.0",
                    help="h-maxima 相对高度（>0 启用尺度自适应种子）")
    ap.add_argument("--gaussian-sigma", default=str(InstanceSplitConfig.gaussian_sigma),
                    help="距离图平滑强度，单位 µm（按间距折算；可给多个值用逗号分隔）")
    ap.add_argument("--spacing-zyx", default=None,
                    help="体素物理间距 (z,y,x) µm；默认从预测文件头自动读取")
    ap.add_argument("--frame-stride", type=int, default=1,
                    help="帧抽样步长（>1 时按等间隔抽帧，覆盖稀疏与密集阶段）")
    ap.add_argument("--frames", default=None,
                    help="显式指定帧号，如 10,50,130,194（优先级高于 --frame-stride）")
    ap.add_argument("--save-json", default=None,
                    help="把扫描结果写入该路径（供实验目录留痕）")
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir)
    tra_dir = Path(args.gt_root) / f"{args.seq}_GT" / "TRA"
    files = sorted(pred_dir.glob(f"CE{args.seq}_f*.nii.gz"))
    if args.frames:
        want = [int(x) for x in args.frames.split(",")]
        by_t = {int(f.name.split("_f")[1][:3]): f for f in files}
        files = [by_t[t] for t in want]
    elif args.frame_stride > 1:
        files = files[:: args.frame_stride]
    if args.max_frames:
        files = files[: args.max_frames]
    print(f"用 {len(files)} 帧标定（seq {args.seq}，stride={args.frame_stride}）："
          f"t={files[0].name} .. {files[-1].name}")

    # 预读所有帧的分割结果与标记，避免重复 I/O
    if args.spacing_zyx:
        spacing_zyx = tuple(float(v) for v in args.spacing_zyx.split(","))
    else:
        import SimpleITK as _sitk
        spacing_zyx = tuple(float(v) for v in reversed(
            _sitk.ReadImage(str(files[0])).GetSpacing()))
    print(f"spacing(z,y,x)={spacing_zyx} µm")

    cache = []
    for pf in files:
        t = int(pf.name.split("_f")[1][:3])
        sem = sitk.GetArrayFromImage(sitk.ReadImage(str(pf))) > 0
        gt = tifffile.imread(tra_dir / f"man_track{t:03d}.tif")
        cache.append((t, sem, gt))

    print(f"{'h_frac':>7s} {'sig_um':>7s} {'min_dist':>9s} {'min_vol':>8s} {'实例/帧':>8s} {'标记/帧':>8s} "
          f"{'召回':>7s} {'精确':>7s} {'独立':>7s} {'F1':>7s}")
    best = None
    rows = []
    grid = [(hf, gs, md, mv)
            for hf in (float(x) for x in args.h_frac.split(","))
            for gs in (float(x) for x in str(args.gaussian_sigma).split(","))
            for md in (int(x) for x in args.min_distance.split(","))
            for mv in (int(x) for x in args.min_volume.split(","))]
    for hf, gs, md, mv in grid:
            cfg = InstanceSplitConfig(min_distance=md, min_volume=mv, h_frac=hf,
                                      gaussian_sigma=gs, spacing_zyx=spacing_zyx)
            recalls, precs, n_pred, n_gt, excl = [], [], [], [], []
            for _t, sem, gt in cache:
                lab = split_instances(sem, cfg)
                st = detection_recall_vs_markers(lab, gt)
                recalls.append(st["recall"]); precs.append(st["precision"])
                n_pred.append(st["n_pred"]); n_gt.append(st["n_gt"])
                excl.append(st.get("exclusive", 0.0))
            r, p = float(np.mean(recalls)), float(np.mean(precs))
            e = float(np.mean(excl))
            f1 = 2 * r * p / max(r + p, 1e-9)
            print(f"{hf:7.2f} {gs:7.2f} {md:9d} {mv:8d} {np.mean(n_pred):8.1f} {np.mean(n_gt):8.1f} "
                  f"{r:7.3f} {p:7.3f} {e:7.3f} {f1:7.3f}")
            rows.append({
                "h_frac": hf, "min_distance": md, "min_volume": mv,
                "gaussian_sigma": gs, "spacing_zyx": list(spacing_zyx),
                "instances_per_frame": float(np.mean(n_pred)),
                "markers_per_frame": float(np.mean(n_gt)),
                "recall": r, "precision": p, "exclusive": e, "f1": f1,
                "instances_per_marker": float(np.mean(n_pred) / max(np.mean(n_gt), 1e-9)),
            })
            if best is None or f1 > best[0]:
                best = (f1, md, mv, r, p, hf)

    print(f"\n最优: F1={best[0]:.3f}  h_frac={best[5]}  min_distance={best[1]}  "
          f"min_volume={best[2]} (召回 {best[3]:.3f}, 精确 {best[4]:.3f})")

    if args.save_json:
        out = Path(args.save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "seq": args.seq,
            "pred_dir": str(pred_dir),
            "frame_stride": args.frame_stride,
            "n_frames": len(files),
            "frames": [f.name for f in files],
            "best": {"f1": best[0], "min_distance": best[1], "min_volume": best[2],
                     "recall": best[3], "precision": best[4], "h_frac": best[5]},
            "sweep": rows,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"已写入 {out}")


if __name__ == "__main__":
    main()
