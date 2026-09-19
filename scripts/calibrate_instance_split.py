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
"""

from __future__ import annotations

import argparse
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
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir)
    tra_dir = Path(args.gt_root) / f"{args.seq}_GT" / "TRA"
    files = sorted(pred_dir.glob(f"CE{args.seq}_f*.nii.gz"))
    if args.max_frames:
        files = files[: args.max_frames]
    print(f"用 {len(files)} 帧标定（seq {args.seq}）")

    # 预读所有帧的分割结果与标记，避免重复 I/O
    cache = []
    for pf in files:
        t = int(pf.name.split("_f")[1][:3])
        sem = sitk.GetArrayFromImage(sitk.ReadImage(str(pf))) > 0
        gt = tifffile.imread(tra_dir / f"man_track{t:03d}.tif")
        cache.append((t, sem, gt))

    print(f"{'h_frac':>7s} {'min_dist':>9s} {'min_vol':>8s} {'实例/帧':>8s} {'标记/帧':>8s} "
          f"{'召回':>7s} {'精确':>7s} {'独立':>7s} {'F1':>7s}")
    best = None
    for hf in (float(x) for x in args.h_frac.split(",")):
      for md in (int(x) for x in args.min_distance.split(",")):
        for mv in (int(x) for x in args.min_volume.split(",")):
            cfg = InstanceSplitConfig(min_distance=md, min_volume=mv, h_frac=hf)
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
            print(f"{hf:7.2f} {md:9d} {mv:8d} {np.mean(n_pred):8.1f} {np.mean(n_gt):8.1f} "
                  f"{r:7.3f} {p:7.3f} {e:7.3f} {f1:7.3f}")
            if best is None or f1 > best[0]:
                best = (f1, md, mv, r, p, hf)

    print(f"\n最优: F1={best[0]:.3f}  h_frac={best[5]}  min_distance={best[1]}  "
          f"min_volume={best[2]} (召回 {best[3]:.3f}, 精确 {best[4]:.3f})")


if __name__ == "__main__":
    main()
