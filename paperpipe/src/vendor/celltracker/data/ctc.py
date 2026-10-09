"""CTC (Cell Tracking Challenge) 数据集读取。

目录约定（2D 与 3D 通用）::

    <dataset_dir>/<seq>/t000.tif ...                 原始图像（3D 为多页 TIFF 栈）
    <dataset_dir>/<seq>_GT/TRA/man_track000.tif ...  追踪 GT，像素值 = track id（整个序列唯一）
    <dataset_dir>/<seq>_GT/TRA/man_track.txt         每行: L B E P
    <dataset_dir>/<seq>_GT/SEG/man_seg000.tif ...    分割 GT，稀疏参考帧（SEG 指标用）

关键性质：TRA 掩码里的像素值就是 track id，因此"细胞实例 → 轨迹身份 → 血缘"
全部可由 `labels` + `man_track.txt` 推出，无需额外标注。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile

__all__ = ["Track", "CTCSequence", "list_sequences", "read_man_track"]


@dataclass(frozen=True)
class Track:
    """一条 CTC 轨迹记录。"""

    label: int
    begin: int
    end: int
    parent: int = 0

    @property
    def length(self) -> int:
        return self.end - self.begin + 1

    @property
    def is_birth_by_division(self) -> bool:
        return self.parent != 0


def read_man_track(path: str | Path) -> dict[int, Track]:
    """读取 `man_track.txt` / `res_track.txt`：每行 `L B E P`。"""
    tracks: dict[int, Track] = {}
    path = Path(path)
    if not path.exists():
        return tracks
    with path.open("r") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 4:
                continue
            label, begin, end, parent = (int(x) for x in parts[:4])
            tracks[label] = Track(label, begin, end, parent)
    return tracks


def _frame_index(path: Path, prefix: str) -> int | None:
    m = re.fullmatch(rf"{re.escape(prefix)}(\d+)", path.stem)
    return int(m.group(1)) if m else None


def list_sequences(dataset_dir: str | Path) -> list[str]:
    """列出数据集下的序列目录（如 `01`, `02`）。"""
    dataset_dir = Path(dataset_dir)
    seqs = []
    for child in sorted(dataset_dir.iterdir()):
        if child.is_dir() and re.fullmatch(r"\d+", child.name):
            seqs.append(child.name)
    return seqs


class CTCSequence:
    """单个 CTC 序列的惰性读取器（带单帧缓存）。"""

    def __init__(self, dataset_dir: str | Path, seq: str, cache_frames: int = 2):
        self.dataset_dir = Path(dataset_dir)
        self.seq = seq
        self.dir = self.dataset_dir / seq
        self.gt_dir = self.dataset_dir / f"{seq}_GT"
        self.tra_dir = self.gt_dir / "TRA"
        self.seg_dir = self.gt_dir / "SEG"
        self.cache_frames = cache_frames
        self._cache: dict[tuple[str, int], np.ndarray] = {}

        self.frames = self._discover(self.dir, "t", ".tif")
        if not self.frames:
            raise FileNotFoundError(f"未在 {self.dir} 找到 t###.tif 图像")
        self.n_frames = len(self.frames)
        self._shape: tuple[int, ...] | None = None

    # ---------- 发现 ----------

    @staticmethod
    def _discover(folder: Path, prefix: str, suffix: str) -> list[int]:
        if not folder.is_dir():
            return []
        idx = []
        for p in folder.iterdir():
            if p.suffix.lower() != suffix:
                continue
            i = _frame_index(p, prefix)
            if i is not None:
                idx.append(i)
        return sorted(idx)

    @property
    def tracks(self) -> dict[int, Track]:
        return read_man_track(self.tra_dir / "man_track.txt")

    @property
    def tra_frames(self) -> list[int]:
        return self._discover(self.tra_dir, "man_track", ".tif")

    @property
    def seg_frames(self) -> list[int]:
        """SEG 金标准覆盖的帧号（去重排序）。

        兼容两种命名：`man_segT.tif`（2D 或 3D 整体标注）与
        `man_seg_T_Z.tif`（3D 逐切片标注）。
        """
        return sorted({t for t, _, _ in self.seg_entries})

    @property
    def seg_entries(self) -> list[tuple[int, int | None, Path]]:
        """[(帧号, 切片号或 None, 路径), ...]，按文件名排序。"""
        if not self.seg_dir.is_dir():
            return []
        out: list[tuple[int, int | None, Path]] = []
        for p in sorted(self.seg_dir.iterdir()):
            if p.suffix.lower() != ".tif":
                continue
            if m := re.fullmatch(r"man_seg_(\d+)_(\d+)", p.stem):
                out.append((int(m.group(1)), int(m.group(2)), p))
            elif m := re.fullmatch(r"man_seg(\d+)", p.stem):
                out.append((int(m.group(1)), None, p))
        return out

    @property
    def has_tra(self) -> bool:
        return bool(self.tra_frames)

    @property
    def has_seg(self) -> bool:
        return bool(self.seg_frames)

    # ---------- 读取 ----------

    def _load(self, kind: str, t: int) -> np.ndarray:
        key = (kind, t)
        if key in self._cache:
            return self._cache[key]
        if kind == "image":
            path = self.dir / f"t{t:03d}.tif"
        elif kind == "tra":
            path = self.tra_dir / f"man_track{t:03d}.tif"
        elif kind == "seg":
            path = self.seg_dir / f"man_seg{t:03d}.tif"
        else:  # pragma: no cover
            raise ValueError(kind)
        if not path.exists():
            raise FileNotFoundError(path)
        arr = tifffile.imread(path)
        if self.cache_frames > 0:
            if len(self._cache) >= self.cache_frames:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = arr
        return arr

    def image(self, t: int) -> np.ndarray:
        return self._load("image", t)

    def tra_labels(self, t: int) -> np.ndarray:
        return self._load("tra", t)

    def seg_labels(self, t: int) -> np.ndarray:
        return self._load("seg", t)

    @property
    def shape(self) -> tuple[int, ...]:
        """单帧空间维度（不含时间）。"""
        if self._shape is None:
            t0 = self.frames[0]
            self._shape = tuple(int(s) for s in self.image(t0).shape)
        return self._shape

    @property
    def ndim(self) -> int:
        return len(self.shape)

    # ---------- 派生量 ----------

    def object_table(self, t: int, kind: str = "tra") -> dict[str, np.ndarray]:
        """返回第 t 帧的实例表（质心/体积/包围盒/强度统计）。

        `kind="tra"` 用追踪 GT（像素值 = track id），`kind="seg"` 用分割 GT
        （像素值在该帧内为 1..n 的局部标签）。
        """
        labels = self.tra_labels(t) if kind == "tra" else self.seg_labels(t)
        return object_table_from_labels(labels, image=self.image(t))


def object_table_from_labels(
    labels: np.ndarray, image: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    """从标签体数据提取实例表。

    返回字典包含：
      - `label`: 实例标签（TRA 时为 track id）
      - `centroid`: (n, ndim) 质心坐标
      - `volume`: (n,) 体素数
      - `bbox_min` / `bbox_max`: (n, ndim) 包围盒（含端点，Python 索引语义）
      - `intensity_mean` / `intensity_std`: (n,) 仅当提供 image 时
    """
    labels = np.asarray(labels)
    ndim = labels.ndim
    flat = labels.reshape(-1)
    nz = flat > 0
    present = flat[nz]
    uniq = np.unique(present)
    n = uniq.size

    out: dict[str, np.ndarray] = {
        "label": uniq.astype(np.int64),
        "centroid": np.zeros((n, ndim), dtype=np.float64),
        "volume": np.zeros(n, dtype=np.int64),
        "bbox_min": np.zeros((n, ndim), dtype=np.int64),
        "bbox_max": np.zeros((n, ndim), dtype=np.int64),
    }
    if n == 0:
        if image is not None:
            out["intensity_mean"] = np.zeros(0)
            out["intensity_std"] = np.zeros(0)
        return out

    # 用 searchsorted 把标签映射到 0..n-1，随后全程 bincount（向量化，O(体素数)）
    g = np.searchsorted(uniq, present)

    coords = np.nonzero(labels)
    volume = np.bincount(g, minlength=n).astype(np.int64)
    out["volume"] = volume
    for d in range(ndim):
        out["centroid"][:, d] = np.bincount(g, weights=coords[d], minlength=n) / volume

    if image is not None:
        vals = np.asarray(image).reshape(-1)[nz].astype(np.float64)
        s1 = np.bincount(g, weights=vals, minlength=n)
        s2 = np.bincount(g, weights=vals * vals, minlength=n)
        mean = s1 / volume
        var = np.maximum(s2 / volume - mean * mean, 0.0)
        out["intensity_mean"] = mean
        out["intensity_std"] = np.sqrt(var)

    # 包围盒：scipy 的 find_objects 返回按 label-1 索引的列表
    from scipy import ndimage

    objs = ndimage.find_objects(labels.astype(np.int32, copy=False))
    for k, lab in enumerate(uniq):
        sl = objs[int(lab) - 1]
        if sl is None:
            continue
        for d in range(ndim):
            out["bbox_min"][k, d] = sl[d].start
            out["bbox_max"][k, d] = sl[d].stop - 1
    return out
