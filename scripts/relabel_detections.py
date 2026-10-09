#!/usr/bin/env python3
"""为已有预测掩码重建 GT 身份映射；保留源文件，逐帧读取体数据。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from celltracker.detect.labels import match_gt_labels


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--gt-h5", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument('--frames',help='可选闭区间，例如 120:125；只生成小样本文件')
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.frames:
        low,high=map(int,args.frames.split(':'))
        if high < low:
            raise ValueError('帧范围起点不能大于终点')
        with h5py.File(args.input,'r') as source,h5py.File(out,'w') as destination:
            for name,value in source.attrs.items():
                destination.attrs[name]=value
            destination.attrs['frame_subset']=args.frames
            target=destination.create_group('frames')
            for t in range(low,high+1):
                key=f'{t:04d}'
                source.copy(source['frames'][key],target,name=key)
            for key in source:
                if key!='frames':
                    source.copy(source[key],destination,name=key)
    else:
        shutil.copy2(args.input, out)
    stats = {"frames": 0, "instances": 0, "changed_primary": 0, "multiple_gt": 0}
    with h5py.File(out, "r+") as pred, h5py.File(args.gt_h5, "r") as truth:
        for key in sorted(pred["frames"]):
            group = pred["frames"][key]
            best, all_ids = match_gt_labels(group["labels"][:], truth["frames"][key]["labels"][:], group["label"][:])
            previous = group["gt_label"][:] if "gt_label" in group else np.zeros(len(best))
            stats["changed_primary"] += int(np.sum(previous != best))
            stats["multiple_gt"] += int(np.sum((all_ids > 0).sum(1) > 1))
            stats["instances"] += len(best)
            stats["frames"] += 1
            for name, values in [("gt_label", best), ("gt_ids", all_ids)]:
                if name in group:
                    del group[name]
                group.create_dataset(name, data=values)
        pred.attrs["label_mapping"] = "ctc_full_marker_majority_v2"
        pred.attrs["detection_source"] = "nnunet_pred"
        pred.attrs["mapping_source"] = str(args.input)
    out.with_suffix(".mapping.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
