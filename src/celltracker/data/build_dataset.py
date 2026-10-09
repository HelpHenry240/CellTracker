"""把 CTC 原始目录转成内部中间表示（HDF5）。

产物结构::

    /                       attrs: name, seq, ndim, shape, n_frames, source, created
    /tracks                 (n,) 结构化: label, begin, end, parent
    /seg_frames             (m,) SEG 参考帧号（稀疏 GT）
    /frames/{t:04d}/labels  追踪 GT 标签体数据（uint16/uint32，gzip）
    /frames/{t:04d}/label   (n,) 实例标签
    /frames/{t:04d}/centroid (n, ndim) float32
    /frames/{t:04d}/volume  (n,) int32
    /frames/{t:04d}/bbox_min, bbox_max  (n, ndim) int16
    /frames/{t:04d}/intensity_mean, intensity_std (n,) float32

用法::

    python -m celltracker.data.build_dataset \
        --dataset data/raw/Fluo-N3DH-CE --seq 01 \
        --out data/interim/Fluo-N3DH-CE_01.h5
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py
import numpy as np

from .ctc import CTCSequence


def build(dataset_dir: str | Path, seq: str, out_path: str | Path,
          limit_frames: int | None = None, quiet: bool = False,
          annotated_only: bool = True) -> Path:
    dataset_dir = Path(dataset_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    s = CTCSequence(dataset_dir, seq)

    frames = s.frames
    if annotated_only and s.has_tra:
        # CTC 的 TRA 标注不一定覆盖每一帧（如 Fluo-N3DH-CE 只标注 195/250 帧）
        annotated = set(s.tra_frames)
        frames = [t for t in frames if t in annotated]
    if limit_frames is not None:
        frames = frames[:limit_frames]
    seg_frames = [t for t in s.seg_frames if t in set(frames)]

    if not quiet:
        print(f"[{dataset_dir.name}/{seq}] 空间维度 {s.shape} (ndim={s.ndim}), "
              f"帧数 {s.n_frames}（本次处理 {len(frames)}）, TRA={s.has_tra}, SEG={s.has_seg}")

    t0 = time.time()
    with h5py.File(out_path, "w") as f:
        f.attrs.update(
            name=dataset_dir.name,
            seq=str(seq),
            ndim=s.ndim,
            shape=np.asarray(s.shape, dtype=np.int32),
            n_frames=len(frames),
            n_frames_total=s.n_frames,
            source=str(dataset_dir.resolve()),
            created=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )

        tracks = s.tracks
        if tracks:
            labels = np.array(sorted(tracks), dtype=np.int32)
            arr = np.zeros(len(labels), dtype=[("label", "i4"), ("begin", "i4"),
                                               ("end", "i4"), ("parent", "i4")])
            arr["label"] = labels
            arr["begin"] = [tracks[int(l)].begin for l in labels]
            arr["end"] = [tracks[int(l)].end for l in labels]
            arr["parent"] = [tracks[int(l)].parent for l in labels]
            f.create_dataset("tracks", data=arr)
        f.create_dataset("seg_frames", data=np.asarray(seg_frames, dtype=np.int32))

        max_label = max(tracks) if tracks else 1
        label_dtype = np.uint16 if max_label < 65535 else np.uint32
        grp_frames = f.create_group("frames")

        for i, t in enumerate(frames):
            g = grp_frames.create_group(f"{t:04d}")
            labels_vol = s.tra_labels(t) if s.has_tra else s.seg_labels(t)
            g.create_dataset("labels", data=labels_vol.astype(label_dtype, copy=False),
                             compression="gzip", compression_opts=4)
            tab = s.object_table(t, kind="tra" if s.has_tra else "seg")
            g.create_dataset("label", data=tab["label"].astype(np.int32))
            g.create_dataset("centroid", data=tab["centroid"].astype(np.float32))
            g.create_dataset("volume", data=tab["volume"].astype(np.int32))
            g.create_dataset("bbox_min", data=tab["bbox_min"].astype(np.int16))
            g.create_dataset("bbox_max", data=tab["bbox_max"].astype(np.int16))
            g.create_dataset("intensity_mean", data=tab["intensity_mean"].astype(np.float32))
            g.create_dataset("intensity_std", data=tab["intensity_std"].astype(np.float32))

            if not quiet and (i % 10 == 0 or i == len(frames) - 1):
                print(f"  帧 {t:4d} [{i + 1}/{len(frames)}] 实例数 {tab['label'].size:5d} "
                      f"耗时 {time.time() - t0:6.1f}s")

    if not quiet:
        print(f"写出: {out_path} ({out_path.stat().st_size / 1048576:.1f} MB, "
              f"总耗时 {time.time() - t0:.1f}s)")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="构建 CTC 内部 HDF5 表示")
    ap.add_argument("--dataset", required=True, help="数据集目录, 如 data/raw/Fluo-N3DH-CE")
    ap.add_argument("--seq", default="01", help="序列号, 如 01")
    ap.add_argument("--out", default=None, help="输出 h5 路径")
    ap.add_argument("--limit-frames", type=int, default=None, help="只处理前 N 帧（调试用）")
    ap.add_argument("--all-frames", action="store_true",
                    help="处理所有图像帧（默认只处理有 TRA 标注的帧）")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset)
    out = args.out or f"data/interim/{dataset_dir.name}_{args.seq}.h5"
    build(dataset_dir, args.seq, out, limit_frames=args.limit_frames,
          annotated_only=not args.all_frames)


if __name__ == "__main__":
    main()
