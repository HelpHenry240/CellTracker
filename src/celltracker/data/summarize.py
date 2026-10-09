"""从内部 HDF5 统计数据集结构，并出图（实验 E0.2）。

用法::

    python -m celltracker.data.summarize --h5 data/interim/Fluo-N3DH-CHO_01.h5 \
        --figdir experiments/E0.2_data_stats/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from ..viz import PALETTE, savefig


def summarize(h5_path: str | Path, figdir: str | Path, out_json: str | Path | None = None
              ) -> dict:
    h5_path = Path(h5_path)
    figdir = Path(figdir)
    figdir.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        name = f.attrs["name"]
        seq = f.attrs["seq"]
        shape = tuple(int(x) for x in f.attrs["shape"])
        frames = sorted(f["frames"].keys())
        n_objects, volumes, centroids, labels = [], [], [], []
        for key in frames:
            g = f["frames"][key]
            n_objects.append(int(g["centroid"].shape[0]))
            volumes.append(np.asarray(g["volume"], dtype=float))
            centroids.append(np.asarray(g["centroid"], dtype=float))
            labels.append(np.asarray(g["label"], dtype=np.int64))
        tracks = np.asarray(f["tracks"]) if "tracks" in f else np.zeros(0, dtype=[("label", "i4")])
        seg_frames = np.asarray(f["seg_frames"])

    n_frames = len(frames)
    track_labels = tracks["label"]
    lengths = tracks["end"] - tracks["begin"] + 1
    n_divisions = int(np.count_nonzero(tracks["parent"] > 0))

    # 相邻帧位移：GT 标签即轨迹 id，可直接配对
    disp: list[float] = []
    for t in range(n_frames - 1):
        la, lb = labels[t], labels[t + 1]
        common, ia, ib = np.intersect1d(la, lb, return_indices=True)
        if common.size:
            d = np.linalg.norm(centroids[t][ia] - centroids[t + 1][ib], axis=1)
            disp.extend(d.tolist())

    volumes_all = np.concatenate(volumes) if volumes else np.zeros(1)
    stats = {
        "dataset": str(name),
        "seq": str(seq),
        "shape": list(shape),
        "n_frames": n_frames,
        "n_tracks": int(track_labels.size),
        "n_divisions": n_divisions,
        "n_seg_ref_frames": int(seg_frames.size),
        "objects_per_frame": {
            "min": int(np.min(n_objects)), "max": int(np.max(n_objects)),
            "mean": float(np.mean(n_objects)),
        },
        "objects_total": int(np.sum(n_objects)),
        "track_length": {
            "min": int(lengths.min()) if lengths.size else 0,
            "max": int(lengths.max()) if lengths.size else 0,
            "mean": float(lengths.mean()) if lengths.size else 0.0,
            "median": float(np.median(lengths)) if lengths.size else 0.0,
        },
        "volume_voxels": {
            "min": float(volumes_all.min()), "max": float(volumes_all.max()),
            "mean": float(volumes_all.mean()),
        },
        "frame_displacement_voxels": {
            "mean": float(np.mean(disp)) if disp else 0.0,
            "p95": float(np.percentile(disp, 95)) if disp else 0.0,
            "max": float(np.max(disp)) if disp else 0.0,
        },
    }

    # ---- 图 1: 细胞数随时间 ----
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    ax.plot(np.arange(n_frames), n_objects, color=PALETTE["blue"], lw=1.2)
    ax.set_xlabel("frame index")
    ax.set_ylabel("objects per frame")
    ax.set_title(f"{name} seq {seq}: object count over time")
    savefig(fig, figdir / "object_count_over_time")

    # ---- 图 2: 轨迹长度分布 + 体积分布 + 位移分布 ----
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.8))
    axes[0].hist(lengths, bins=min(30, max(5, int(lengths.max()))), color=PALETTE["green"])
    axes[0].set_xlabel("track length (frames)")
    axes[0].set_ylabel("count")
    axes[0].set_title(f"track lengths (n={track_labels.size}, div={n_divisions})")

    axes[1].hist(volumes_all, bins=40, color=PALETTE["orange"])
    axes[1].set_xlabel("volume (voxels)")
    axes[1].set_ylabel("count")
    axes[1].set_title("object volume")

    if disp:
        axes[2].hist(disp, bins=40, color=PALETTE["purple"])
    axes[2].set_xlabel("frame-to-frame displacement (voxels)")
    axes[2].set_ylabel("count")
    axes[2].set_title("motion magnitude")
    savefig(fig, figdir / "distributions")

    if out_json:
        out_json = Path(out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="CTC 数据集统计与可视化")
    ap.add_argument("--h5", required=True)
    ap.add_argument("--figdir", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    stats = summarize(args.h5, args.figdir, args.out)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
