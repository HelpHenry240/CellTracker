#!/usr/bin/env python3
"""把 CTC Fluo-N3DH-CE 转成 nnU-Net v2 数据集（在云服务器上运行）。

要点：
  1. 训练标签用 **银标准 `_ST/SEG`**（195/190 帧全卷分割），而不是 `_GT/SEG`
     —— 后者在 CE 上每个序列只有 5 个切片，无法训练。
  2. nnU-Net 做的是**语义分割**，因此把实例标签二值化（所有细胞 = 1）。
     实例分割留到评测阶段用后处理（连通域 / watershed）解决。
  3. 体素间距必须显式写入，否则 SimpleITK 会当成各向同性 (1,1,1)，
     nnU-Net 的重采样决策会完全错误。CE 的间距为 (x=0.09, y=0.09, z=1.0) µm。

用法（云端）::

    /root/autodl-tmp/nnunet/venv/bin/python build_dataset.py \
        --src /root/autodl-tmp/ctc/Fluo-N3DH-CE \
        --out /root/autodl-tmp/nnunet/nnUNet_raw/Dataset501_CellTrackerCE
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import tifffile

SPACING_XYZ = (0.09, 0.09, 1.0)  # CE: 0.09 × 0.09 × 1.0 µm（x, y, z）


def convert_case(img_tif: Path, seg_tif: Path, out_img: Path, out_seg: Path) -> None:
    img = tifffile.imread(img_tif)          # (z, y, x) uint8
    seg = tifffile.imread(seg_tif)          # (z, y, x) uint16 实例标签
    if img.shape != seg.shape:
        raise ValueError(f"形状不一致: {img_tif.name} {img.shape} vs {seg_tif.name} {seg.shape}")

    itk_img = sitk.GetImageFromArray(np.ascontiguousarray(img))
    itk_img.SetSpacing(SPACING_XYZ)
    sitk.WriteImage(itk_img, str(out_img), useCompression=True)

    binary = (seg > 0).astype(np.uint8)     # 实例 → 语义（前/背景）
    itk_seg = sitk.GetImageFromArray(np.ascontiguousarray(binary))
    itk_seg.SetSpacing(SPACING_XYZ)
    sitk.WriteImage(itk_seg, str(out_seg), useCompression=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Fluo-N3DH-CE 根目录")
    ap.add_argument("--out", required=True, help="nnUNet_raw/DatasetXXX_xxx")
    ap.add_argument("--sequences", default="01,02")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    images_tr = out / "imagesTr"
    labels_tr = out / "labelsTr"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    seqs = args.sequences.split(",")
    n_cases = 0
    for seq in seqs:
        img_dir = src / seq
        seg_dir = src / f"{seq}_ST" / "SEG"
        segs = {int(p.stem[len("man_seg"):]): p for p in seg_dir.glob("man_seg*.tif")}
        frames = sorted(segs)
        print(f"seq {seq}: {len(frames)} 帧有银标准分割")
        for t in frames:
            img_tif = img_dir / f"t{t:03d}.tif"
            if not img_tif.exists():
                print(f"  跳过（无原图）: t{t:03d}")
                continue
            case = f"CE{seq}_f{t:03d}"
            convert_case(img_tif, segs[t],
                         images_tr / f"{case}_0000.nii.gz",
                         labels_tr / f"{case}.nii.gz")
            n_cases += 1
            if n_cases % 50 == 0:
                print(f"  已转换 {n_cases} 例")

    dataset = {
        "channel_names": {"0": "GFP"},
        "labels": {"background": 0, "cell": 1},
        "numTraining": n_cases,
        "file_ending": ".nii.gz",
        "description": "CTC Fluo-N3DH-CE nuclei, silver-truth segmentation, binary labels",
    }
    (out / "dataset.json").write_text(json.dumps(dataset, indent=2))
    print(f"完成：{n_cases} 例 -> {out}")


if __name__ == "__main__":
    main()
