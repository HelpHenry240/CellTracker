#!/usr/bin/env python3
"""把 CTC 的待测帧转成 nnU-Net 推理输入（`*_0000.nii.gz`）。

与训练时的转换保持一致：
  - 体素间距显式写为 (0.09, 0.09, 1.0) µm（x, y, z）；
  *   间距必须与训练一致，否则 nnU-Net 的重采样会与大模型不匹配
  - 只转换**有 GT 标注的帧**（评测只需要这些帧）

用法（云端）::

    python make_infer_input.py --src /root/autodl-tmp/ctc/Fluo-N3DH-CE \
        --out /root/autodl-tmp/nnunet/nnUNet_raw/Dataset501_CellTrackerCE/imagesTs_eval
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import tifffile

SPACING_XYZ = (0.09, 0.09, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gt-root", default=None,
                    help="GT 所在根目录（默认与 --src 相同；云端数据分散时用）")
    ap.add_argument("--sequences", default="01,02")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    gt_root = Path(args.gt_root) if args.gt_root else src
    out.mkdir(parents=True, exist_ok=True)

    n = 0
    for seq in args.sequences.split(","):
        img_dir = src / seq
        tra_dir = gt_root / f"{seq}_GT" / "TRA"
        # 只做有 GT 标注的帧（评测用）
        frames = sorted(int(p.stem[len("man_track"):]) for p in tra_dir.glob("man_track*.tif"))
        print(f"seq {seq}: {len(frames)} 帧需要推理")
        for t in frames:
            img_path = img_dir / f"t{t:03d}.tif"
            if not img_path.exists():
                continue
            case = f"CE{seq}_f{t:03d}"
            img = tifffile.imread(img_path)
            itk = sitk.GetImageFromArray(np.ascontiguousarray(img))
            itk.SetSpacing(SPACING_XYZ)
            sitk.WriteImage(itk, str(out / f"{case}_0000.nii.gz"), useCompression=True)
            n += 1
            if n % 50 == 0:
                print(f"  已转换 {n} 帧")
    print(f"完成：{n} 个推理输入 -> {out}")


if __name__ == "__main__":
    main()
