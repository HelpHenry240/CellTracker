"""C2：把分割网络的**语义掩码**拆成**单细胞实例**（§1.6 的检测前端）。

nnU-Net 输出的是二值前景掩码（细胞 vs 背景），但后续的测度/OT/GNN 都需要
"每个细胞一个实例 + 质心 + 体积"。CE 的胚胎细胞密集且常相互接触，
纯连通域会把多个细胞粘成一个；因此采用标准做法：

  1. 距离变换（distance transform）
  2. 在距离图上取局部极大值作为**种子**（`min_distance` 控制最小细胞间距）
  3. 以种子为标记做**分水岭**，把粘连区域切开

`min_distance` 是唯一的关键参数，可用 GT 标记点**标定**：
比较拆分出的实例数与 GT 标记数，并计算"GT 标记被预测实例覆盖 >50% 的比例"
（即检测召回率）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

__all__ = ["InstanceSplitConfig", "split_instances", "detection_recall_vs_markers"]


@dataclass
class InstanceSplitConfig:
    min_distance: int = 3        # 种子间最小距离（体素）；h_frac>0 时作为兜底
    gaussian_sigma: float = 1.0  # 距离图平滑，抑制噪声极值
    min_volume: int = 30         # 过滤过小的实例（体素），抑制假阳性
    use_watershed: bool = True   # False = 纯连通域（消融对照）
    # **尺度自适应**：用 h-maxima 提取种子，h = h_frac × 距离图的 90 分位数。
    # 为什么需要它：胚胎发育过程中细胞大小变化数倍（实测 CE 上"每细胞核体素"
    # 从早期 ~2.4e4 降到晚期 ~4e3），任何绝对的 min_distance 都无法同时适配
    # 稀疏帧与密集帧——密集帧会欠分割 4–6 倍（实测 46–75 个实例 vs 289 个标记）。
    h_frac: float = 0.35


def split_instances(mask: np.ndarray, cfg: InstanceSplitConfig | None = None) -> np.ndarray:
    """把二值语义掩码拆成实例标签图（0 = 背景，1..n = 实例）。"""
    cfg = cfg or InstanceSplitConfig()
    binary = np.asarray(mask) > 0
    if not binary.any():
        return np.zeros(binary.shape, dtype=np.int32)

    if not cfg.use_watershed:
        labels, n = ndimage.label(binary)
        return _filter_small(labels, n, cfg.min_volume)

    dist = ndimage.distance_transform_edt(binary)
    if cfg.gaussian_sigma > 0:
        dist = ndimage.gaussian_filter(dist, sigma=cfg.gaussian_sigma)

    if cfg.h_frac > 0:
        # 尺度自适应种子：h-maxima（相对高度阈值），自动适配细胞大小
        from skimage.morphology import h_maxima

        h = cfg.h_frac * float(np.percentile(dist[binary], 90))
        local_max = h_maxima(dist, h) & binary
    else:
        # 固定尺度兜底：局部极大值（用 maximum_filter 实现）
        footprint = np.ones((2 * cfg.min_distance + 1,) * binary.ndim, dtype=bool)
        local_max = (dist == ndimage.maximum_filter(dist, footprint=footprint)) & binary
    seeds, n_seeds = ndimage.label(local_max)
    if n_seeds == 0:
        labels, n = ndimage.label(binary)
        return _filter_small(labels, n, cfg.min_volume)

    # 以 -dist 为"高度"做分水岭（flooding from markers），等价于把粘连处切开
    from skimage.segmentation import watershed

    labels = watershed(-dist, seeds, mask=binary)
    return _filter_small(labels, int(labels.max()), cfg.min_volume)


def _filter_small(labels: np.ndarray, n: int, min_volume: int) -> np.ndarray:
    if n == 0 or min_volume <= 1:
        return labels.astype(np.int32, copy=False)
    counts = np.bincount(labels.ravel(), minlength=n + 1)
    keep = counts >= min_volume
    keep[0] = False
    remap = np.zeros(n + 1, dtype=np.int32)
    remap[keep] = np.arange(1, int(keep.sum()) + 1)
    return remap[labels]


def detection_recall_vs_markers(pred_labels: np.ndarray, gt_markers: np.ndarray,
                                min_overlap: float = 0.5) -> dict:
    """用 GT 标记点评估实例拆分质量（检测层面）。

    注意两侧目标的**尺度差异**：CTC 的标记点（marker）远小于真实细胞核
    （实测 CE 上约为 1:23），而分割网络输出的是整个细胞核。因此"相对量"必须
    各按各自的面积算，否则精确率会恒为 0：

    - `recall`    ：GT 标记被某个预测实例覆盖 > `min_overlap`（按**标记**面积）
    - `precision` ：预测实例内含某个 GT 标记 > `min_overlap`（按**标记**面积，
                    即该实例里确实有一个真实细胞）
    - `exclusive` ：标记的"最佳实例"互不冲突的比例（衡量是否把细胞核切碎）
    - `n_pred` / `n_gt`：实例数与标记数（过分割时 n_pred ≫ n_gt）
    """
    gt = np.asarray(gt_markers).ravel()
    pr = np.asarray(pred_labels).ravel()
    keep = (gt > 0) & (pr > 0)
    if not keep.any():
        return {"recall": 0.0, "precision": 0.0, "n_pred": 0, "n_gt": int((gt > 0).sum())}
    g, p = gt[keep], pr[keep]
    key = g.astype(np.int64) * (pr.max() + 1) + p
    uniq, counts = np.unique(key, return_counts=True)
    g_u = (uniq // (pr.max() + 1)).astype(np.int64)
    p_u = (uniq % (pr.max() + 1)).astype(np.int64)

    gt_areas = np.bincount(gt[gt > 0], minlength=int(gt.max()) + 1)
    pr_areas = np.bincount(pr[pr > 0], minlength=int(pr.max()) + 1)

    # 两个方向都按**标记面积**判定（标记远小于细胞核，按实例面积判会恒为 0）
    hit = counts > min_overlap * gt_areas[g_u]
    n_gt = int((gt_areas > 0).sum())
    n_pr = int((pr_areas > 0).sum())
    # 每个标记取重叠最大的实例，统计"多标记抢同一实例"的比例
    best_pr: dict[int, tuple[int, int]] = {}
    for g_id, p_id, c in zip(g_u, p_u, counts):
        cur = best_pr.get(int(g_id))
        if cur is None or c > cur[1]:
            best_pr[int(g_id)] = (int(p_id), int(c))
    used: dict[int, int] = {}
    for g_id, (p_id, _c) in best_pr.items():
        used[p_id] = used.get(p_id, 0) + 1
    exclusive = sum(1 for v in used.values() if v == 1) / max(len(best_pr), 1)
    return {
        "recall": float(np.unique(g_u[hit]).size / max(n_gt, 1)),
        # 精确率：被"某个标记覆盖>50%"的实例占比（按标记面积算）
        "precision": float(np.unique(p_u[hit]).size / max(n_pr, 1)),
        "exclusive": float(exclusive),
        "n_pred": n_pr, "n_gt": n_gt,
    }
