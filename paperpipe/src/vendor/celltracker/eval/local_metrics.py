"""本地快速指标：SEG（与官方定义一致）+ 追踪诊断量。

权威的 DET/TRA 由 CTC 官方二进制在云服务器上计算（见 scripts/cloud_eval.py）；
这里提供**逐帧可迭代**的快速反馈，用于实验阶段的方向判断与误差归因。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile


def _frame_jaccard_pairs(gt: np.ndarray, res: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (gt_label, res_label, overlap) 三元组（仅非零重叠）。"""
    # 先做布尔裁剪再转 int64：避免在 3D 大体积上做整卷 int64 拷贝
    gt = np.asarray(gt)
    res = np.asarray(res)
    keep = (gt > 0) & (res > 0)
    gt = gt[keep].astype(np.int64, copy=False)
    res = res[keep].astype(np.int64, copy=False)
    if gt.size == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty
    gres = np.maximum(res.max(), 1) + 1
    key = gt * gres + res
    uniq, counts = np.unique(key, return_counts=True)
    return uniq // gres, uniq % gres, counts


_SEG_STACK_RE = r"man_seg(\d+)$"
_SEG_SLICE_RE = r"man_seg_(\d+)_(\d+)$"


def list_seg_gt(gt_dir: str | Path) -> list[tuple[int, int | None, Path]]:
    """列出 SEG 金标准，返回 (帧号 t, 切片号 z 或 None, 路径)。

    约定（见 CTC "Naming and file content conventions"）：
      - 2D 或 3D 整体标注: `man_segT.tif`      → z=None
      - 3D 逐切片标注:      `man_seg_T_Z.tif`  → z=Z（单张 2D 切片）
    """
    import re

    gt_dir = Path(gt_dir)
    out: list[tuple[int, int | None, Path]] = []
    for p in sorted(gt_dir.glob("man_seg*.tif")):
        if m := re.fullmatch(_SEG_SLICE_RE, p.stem):
            out.append((int(m.group(1)), int(m.group(2)), p))
        elif m := re.fullmatch(_SEG_STACK_RE, p.stem):
            out.append((int(m.group(1)), None, p))
    return out


def _load_res_slice(res_dir: Path, t: int, z: int | None, num_digits: int,
                    gt_shape: tuple[int, ...]) -> np.ndarray | None:
    rp = res_dir / f"mask{t:0{num_digits}d}.tif"
    if not rp.exists():
        return None
    res = tifffile.imread(rp)
    if z is not None and res.ndim > len(gt_shape):
        res = res[z]
    return res


def seg_measure(gt_dir: str | Path, res_dir: str | Path, num_digits: int = 3,
                frames: list[int] | None = None) -> float:
    """CTC SEG 指标（与官方实现一致）。

    对每个 GT 目标 R：找重叠最大的结果目标 S；若 |R∩S| > 0.5·|R| 则记
    J = |R∩S|/|R∪S|，否则记 0。最终取所有 GT 目标的平均。
    支持 3D 数据集的逐切片金标准（`man_seg_T_Z.tif`）。
    """
    res_dir = Path(res_dir)
    entries = list_seg_gt(gt_dir)
    if frames is not None:
        wanted = set(frames)
        entries = [e for e in entries if e[0] in wanted]

    jaccards: list[float] = []
    for t, z, gp in entries:
        gt = tifffile.imread(gp)
        res = _load_res_slice(res_dir, t, z, num_digits, gt.shape)
        if res is None:
            continue
        if gt.shape != res.shape:
            raise ValueError(f"形状不一致: {gp.name} {gt.shape} vs mask{t:03d} {res.shape}")

        gt_labels, res_labels, overlap = _frame_jaccard_pairs(gt, res)
        res_area = _label_area(res)
        gt_area = _label_area(gt)

        # 每个 GT 目标选重叠最大的结果目标
        best: dict[int, tuple[int, int]] = {}
        for gl, rl, ov in zip(gt_labels, res_labels, overlap):
            cur = best.get(int(gl))
            if cur is None or ov > cur[1]:
                best[int(gl)] = (int(rl), int(ov))

        for gl, area in gt_area.items():
            cand = best.get(int(gl))
            if cand is None or cand[1] <= 0.5 * area:
                jaccards.append(0.0)
                continue
            rl, inter = cand
            union = area + res_area.get(rl, 0) - inter
            jaccards.append(inter / union if union > 0 else 0.0)

    return float(np.mean(jaccards)) if jaccards else float("nan")


def _label_area(labels: np.ndarray) -> dict[int, int]:
    flat = labels.astype(np.int64, copy=False).ravel()
    flat = flat[flat > 0]
    if flat.size == 0:
        return {}
    uniq, counts = np.unique(flat, return_counts=True)
    return {int(u): int(c) for u, c in zip(uniq, counts)}


def match_gt_res(gt: np.ndarray, res: np.ndarray, min_overlap: float = 0.5
                 ) -> dict[int, int]:
    """帧内 GT→RES 的匹配（结果目标需覆盖 GT 目标 > min_overlap 比例）。"""
    gt_labels, res_labels, overlap = _frame_jaccard_pairs(gt, res)
    gt_area = _label_area(gt)
    best: dict[int, tuple[int, int]] = {}
    for gl, rl, ov in zip(gt_labels, res_labels, overlap):
        cur = best.get(int(gl))
        if cur is None or ov > cur[1]:
            best[int(gl)] = (int(rl), int(ov))
    out: dict[int, int] = {}
    for gl, (rl, ov) in best.items():
        if ov > min_overlap * gt_area[gl]:
            out[gl] = rl
    return out


def tracking_diagnostics(gt_frames: dict[int, np.ndarray], res_frames: dict[int, np.ndarray],
                         gt_tracks: dict[int, "object"] | None = None,
                         ) -> dict[str, float]:
    """追踪诊断量：FN/FP、ID switch、轨迹碎片、分裂召回等。

    `gt_frames` / `res_frames`: 帧号 → 标签体数据（TRA 掩码，像素值为轨迹 id）。
    """
    from collections import defaultdict

    n_fn = n_fp = n_matched = 0
    gt_total = res_total = 0
    # res_track → 依时间为序匹配到的 gt_track 序列
    res_to_gt: dict[int, list[int]] = defaultdict(list)
    frames = sorted(set(gt_frames) & set(res_frames))

    for t in frames:
        gt, res = gt_frames[t], res_frames[t]
        mapping = match_gt_res(gt, res)
        gt_area = _label_area(gt)
        res_area = _label_area(res)
        gt_total += len(gt_area)
        res_total += len(res_area)
        n_matched += len(mapping)
        n_fn += len(gt_area) - len(mapping)
        n_fp += len(res_area) - len(set(mapping.values()))
        for gl, rl in mapping.items():
            res_to_gt[rl].append((t, gl))

    id_switches = 0
    for rl, seq in res_to_gt.items():
        seq.sort()
        prev = None
        for _, gl in seq:
            if prev is not None and gl != prev:
                id_switches += 1
            prev = gl

    # 每条 GT 轨迹被多少条 res 轨迹覆盖（>1 说明碎片化）
    gt_to_res: dict[int, set[int]] = defaultdict(set)
    for rl, seq in res_to_gt.items():
        for _, gl in seq:
            gt_to_res[gl].add(rl)
    fragmentation = int(sum(max(0, len(v) - 1) for v in gt_to_res.values()))

    gt_track_seconds = 0
    if gt_tracks is not None:
        gt_track_seconds = len(gt_tracks)

    return {
        "gt_objects": gt_total,
        "res_objects": res_total,
        "matched": n_matched,
        "fn": n_fn,
        "fp": n_fp,
        "detection_recall": n_matched / gt_total if gt_total else float("nan"),
        # 欠分割时多个 GT 可匹配一个结果；精确率的分子必须数结果身份。
        "detection_precision": (res_total-n_fp) / res_total if res_total else float("nan"),
        "matched_results": res_total-n_fp,
        "merged_gt_excess": n_matched-(res_total-n_fp),
        "id_switches": id_switches,
        "fragmentation": fragmentation,
        "gt_tracks": gt_track_seconds,
        "frames_evaluated": len(frames),
    }


class StreamingDiagnostics:
    """逐帧累积诊断量（3D 数据必须用流式，否则内存爆炸）。"""

    def __init__(self, gt_tracks: np.ndarray | None = None) -> None:
        self.n_fn = self.n_fp = self.n_matched = 0
        self.gt_total = self.res_total = 0
        self.res_to_gt: dict[int, list[tuple[int, int]]] = {}
        self.n_frames = 0
        self.mapping: dict[int, dict[int, int]] = {}   # t -> {gt_label: pred_track}
        self.gt_tracks = gt_tracks

    def add_frame(self, t: int, gt: np.ndarray, res: np.ndarray) -> None:
        mapping = match_gt_res(gt, res)
        gt_area = _label_area(gt)
        res_area = _label_area(res)
        self.gt_total += len(gt_area)
        self.res_total += len(res_area)
        self.n_matched += len(mapping)
        self.n_fn += len(gt_area) - len(mapping)
        self.n_fp += len(res_area) - len(set(mapping.values()))
        for gl, rl in mapping.items():
            self.res_to_gt.setdefault(rl, []).append((t, gl))
        self.mapping[t] = dict(mapping)
        self.n_frames += 1

    # ---------- 分裂事件 P/R ----------

    def _gt_divisions(self) -> dict[tuple[int, int], set[int]]:
        """{(父标签, 事件帧): {子标签, ...}}，事件帧 = 子轨迹起始帧。"""
        if self.gt_tracks is None or len(self.gt_tracks) == 0:
            return {}
        out: dict[tuple[int, int], set[int]] = {}
        for lab, begin, _end, parent in zip(self.gt_tracks["label"], self.gt_tracks["begin"],
                                            self.gt_tracks["end"], self.gt_tracks["parent"]):
            if parent:
                out.setdefault((int(parent), int(begin)), set()).add(int(lab))
        return out

    def division_pr(self, pred_tracks: dict[int, "object"] | None = None) -> dict[str, float]:
        """分裂事件的召回率/精确率（与 GT 血缘逐事件比对）。"""
        if self.gt_tracks is None or pred_tracks is None:
            return {}

        gt_div = self._gt_divisions()
        pred_div: dict[tuple[int, int], set[int]] = {}
        for tid, tr in pred_tracks.items():
            if getattr(tr, "parent", 0):
                pred_div.setdefault((int(tr.parent), int(tr.begin)), set()).add(int(tid))

        # GT 分裂是否被"检出"：父在 t-1 的预测轨迹 与 两个子在 t 的预测轨迹 互不相同
        tp_recall = 0
        for (gp, gb), gchildren in gt_div.items():
            prev = self.mapping.get(gb - 1, {}).get(gp)
            kids = {self.mapping.get(gb, {}).get(c) for c in gchildren}
            kids.discard(None)
            if prev is not None and len(kids) >= 2 and prev not in kids:
                tp_recall += 1

        # 预测分裂是否正确：其父/子在 GT 归属上确实构成一次分裂
        tp_prec = 0
        for (pp, pb), pchildren in pred_div.items():
            # 预测的父轨迹在 t-1 覆盖了哪个 GT 标签
            g_parent = None
            for gl, rl in self.mapping.get(pb - 1, {}).items():
                if rl == pp:
                    g_parent = gl
                    break
            ok = False
            if g_parent is not None:
                g_kids = gt_div.get((g_parent, pb))
                if g_kids and g_kids.issubset(set(self.mapping.get(pb, {}).keys())):
                    ok = True
            if ok:
                tp_prec += 1

        return {
            "division_gt": len(gt_div),
            "division_pred": len(pred_div),
            "division_recall": tp_recall / len(gt_div) if gt_div else float("nan"),
            "division_precision": tp_prec / len(pred_div) if pred_div else float("nan"),
        }

    def result(self) -> dict[str, float]:
        id_switches = 0
        for seq in self.res_to_gt.values():
            seq.sort()
            prev = None
            for _, gl in seq:
                if prev is not None and gl != prev:
                    id_switches += 1
                prev = gl
        gt_to_res: dict[int, set[int]] = {}
        for rl, seq in self.res_to_gt.items():
            for _, gl in seq:
                gt_to_res.setdefault(gl, set()).add(rl)
        fragmentation = int(sum(max(0, len(v) - 1) for v in gt_to_res.values()))
        return {
            "gt_objects": self.gt_total,
            "res_objects": self.res_total,
            "matched": self.n_matched,
            "fn": self.n_fn,
            "fp": self.n_fp,
            "detection_recall": self.n_matched / self.gt_total if self.gt_total else float("nan"),
            "detection_precision": (self.res_total-self.n_fp) / self.res_total if self.res_total else float("nan"),
            "matched_results": self.res_total-self.n_fp,
            "merged_gt_excess": self.n_matched-(self.res_total-self.n_fp),
            "id_switches": id_switches,
            "fragmentation": fragmentation,
            "gt_tracks": len(gt_to_res),
            "frames_evaluated": self.n_frames,
        }
