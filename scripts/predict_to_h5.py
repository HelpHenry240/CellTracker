#!/usr/bin/env python3
"""C3：把 nnU-Net 预测掩码 → 实例拆分 → 内部检测格式（h5）。

产物与 `data.build_dataset` 的输出**同构**（frames/{t}/label|centroid|volume|...），
因此下游的 OT / 图 / GNN / 重建全部无需改动，只要把 `--h5` 指到本文件即可。
唯一区别：本文件不写 `tracks`（没有 GT 血缘），评测时用 `--gt-h5` 单独提供。

用法::

    python scripts/predict_to_h5.py \
        --pred-dir preds_eval --img-root data/raw/Fluo-N3DH-CE \
        --seq 01 --out data/interim/Fluo-N3DH-CE_01_pred.h5 \
        --min-distance 3 --min-volume 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from celltracker.data.ctc import object_table_from_labels  # noqa: E402
from celltracker.detect.instances import (InstanceSplitConfig,  # noqa: E402
                                          detection_recall_vs_markers,
                                          split_instances)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True, help="nnU-Net 预测目录（*.nii.gz）")
    ap.add_argument("--img-root", required=True, help="原始图像根目录（算强度统计用）")
    ap.add_argument("--seq", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-distance", type=int, default=3)
    ap.add_argument("--min-volume", type=int, default=30)
    ap.add_argument("--no-watershed", action="store_true")
    ap.add_argument("--gt-root", default=None, help="GT 根目录（用于标定报告，可选）")
    ap.add_argument("--report", default=None, help="检测层面对比报告（json）")
    args = ap.parse_args()

    import SimpleITK as sitk
    import json

    pred_dir = Path(args.pred_dir)
    img_dir = Path(args.img_root) / args.seq
    cfg = InstanceSplitConfig(min_distance=args.min_distance,
                              min_volume=args.min_volume,
                              use_watershed=not args.no_watershed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(pred_dir.glob(f"CE{args.seq}_f*.nii.gz"))
    if not files:
        sys.exit(f"未找到预测文件: {pred_dir}/CE{args.seq}_f*.nii.gz")
    print(f"seq {args.seq}: {len(files)} 帧预测")

    gt_markers = None
    if args.gt_root:
        tra = Path(args.gt_root) / f"{args.seq}_GT" / "TRA"
        gt_markers = lambda t: tifffile.imread(tra / f"man_track{t:03d}.tif")  # noqa: E731

    stats_rows = []
    with h5py.File(out_path, "w") as f:
        first = sitk.ReadImage(str(files[0]))
        shape = tuple(int(x) for x in reversed(first.GetSize()))  # (z, y, x)
        f.attrs.update(name=f"Fluo-N3DH-CE-pred-{args.seq}", seq=args.seq,
                       ndim=3, shape=np.asarray(shape, dtype=np.int32),
                       source=str(pred_dir), min_distance=args.min_distance,
                       min_volume=args.min_volume,
                       watershed=not args.no_watershed)
        f.create_dataset("seg_frames", data=np.array([], dtype=np.int32))
        gf = f.create_group("frames")

        for i, pf in enumerate(files):
            t = int(pf.name.split("_f")[1][:3])
            sem = sitk.GetArrayFromImage(sitk.ReadImage(str(pf)))
            labels = split_instances(sem, cfg)
            img = tifffile.imread(img_dir / f"t{t:03d}.tif")
            tab = object_table_from_labels(labels, image=img)

            # ---- 检测 → GT 轨迹 的映射（真实检测设定下训练 GNN 必需）----
            # 每个预测实例与哪个 GT 标记重叠 >50%，就继承该标记的 track id；
            # 未匹配者记 0（即假阳性）。这样预测 h5 可以**直接替代 GT h5**
            # 进入训练路径（图构建用 gt_label 生成边标签），避免训练/测试分布不一致。
            gt_label = np.zeros(tab["label"].size, dtype=np.int32)
            if gt_markers is not None:
                gt = gt_markers(t)
                g_flat = gt.ravel()
                l_flat = labels.ravel()
                keep = (g_flat > 0) & (l_flat > 0)
                if keep.any():
                    gv, lv = g_flat[keep], l_flat[keep]
                    n_lab = int(labels.max()) + 1
                    key = lv.astype(np.int64) * (int(gt.max()) + 1) + gv
                    uniq, cnt = np.unique(key, return_counts=True)
                    pl = (uniq // (int(gt.max()) + 1)).astype(np.int64)
                    gl = (uniq % (int(gt.max()) + 1)).astype(np.int64)
                    g_areas = np.bincount(gv, minlength=int(gt.max()) + 1)
                    order = np.argsort(-cnt)          # 每个实例取重叠最大的标记
                    for idx in order:
                        if cnt[idx] > 0.5 * g_areas[gl[idx]] and gt_label[pl[idx] - 1] == 0:
                            gt_label[pl[idx] - 1] = gl[idx]
                    assert n_lab > 0

            g = gf.create_group(f"{t:04d}")
            g.create_dataset("labels", data=labels.astype(np.uint16, copy=False),
                             compression="gzip", compression_opts=4)
            g.create_dataset("label", data=tab["label"].astype(np.int32))
            g.create_dataset("gt_label", data=gt_label)
            g.create_dataset("centroid", data=tab["centroid"].astype(np.float32))
            g.create_dataset("volume", data=tab["volume"].astype(np.int32))
            g.create_dataset("bbox_min", data=tab["bbox_min"].astype(np.int16))
            g.create_dataset("bbox_max", data=tab["bbox_max"].astype(np.int16))
            g.create_dataset("intensity_mean", data=tab["intensity_mean"].astype(np.float32))
            g.create_dataset("intensity_std", data=tab["intensity_std"].astype(np.float32))

            row = {"t": t, "n_pred": int(tab["label"].size),
                   "fg_voxels": int((sem > 0).sum()),
                   "n_matched": int((gt_label > 0).sum())}
            if gt_markers is not None:
                gt = gt_markers(t)
                row.update(detection_recall_vs_markers(labels, gt))
            stats_rows.append(row)
            if (i + 1) % 25 == 0 or i == len(files) - 1:
                msg = f"  帧 {t:3d} [{i + 1}/{len(files)}] 实例 {row['n_pred']:4d}"
                if "recall" in row:
                    msg += f" 检测召回 {row['recall']:.3f} 精确 {row['precision']:.3f}"
                print(msg)

    print(f"写出: {out_path} ({out_path.stat().st_size / 1048576:.1f} MB)")
    if args.report:
        agg = {
            "n_frames": len(stats_rows),
            "mean_instances": float(np.mean([r["n_pred"] for r in stats_rows])),
            "config": {"min_distance": args.min_distance, "min_volume": args.min_volume,
                       "watershed": not args.no_watershed},
        }
        if "recall" in stats_rows[0]:
            agg["detection_recall"] = float(np.mean([r["recall"] for r in stats_rows]))
            agg["detection_precision"] = float(np.mean([r["precision"] for r in stats_rows]))
            agg["mean_gt_markers"] = float(np.mean([r["n_gt"] for r in stats_rows]))
        Path(args.report).write_text(json.dumps(
            {"aggregate": agg, "per_frame": stats_rows}, ensure_ascii=False, indent=2))
        print(f"报告: {args.report} -> {agg}")


if __name__ == "__main__":
    main()
