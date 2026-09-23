"""CTC 结果导出：流式写 mask + res_track.txt，并处理论文 vs CTC 格式的空洞冲突。

原文（§2.0.1）："此时虽然帧 t 的 mask 仍为空，但整体轨迹不会被截断"。
CTC 官方格式却要求轨迹在 [begin,end] 内每帧都出现 —— 这是**论文与评测的唯一正面冲突**，
处理方式（`hole_policy`）见 `export_ctc` 的 docstring（ENG_SUPP，FORMULA_MAP 的 E-2）。

本模块只负责"把重建结果写成合法提交"；轨迹的规范化在
`reconstruction/tracks.py`，编排在 `runtime/pipeline.py`。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from celltracker.eval.ctc_io import ResultWriter
from celltracker.eval.local_metrics import StreamingDiagnostics, seg_measure
from celltracker.track.base import Detections, paint_result

if TYPE_CHECKING:  # 避免与 runtime.pipeline 循环导入
    from ..runtime.pipeline import PipelineRun

__all__ = ["export_ctc"]


# ---------------------------------------------------------------------------
# CTC 结果导出（含空洞帧补画）
# ---------------------------------------------------------------------------

def _hole_plan(assignment: dict[int, np.ndarray], dets: Detections
               ) -> dict[int, list[tuple[int, int, int]]]:
    """把空洞展开成"逐帧要补画的实例"：frame → [(tid, src_frame, tgt_frame)]。"""
    plan: dict[int, list[tuple[int, int, int]]] = {}
    for tid, fr in _frames_of(assignment).items():
        for a, b in zip(fr[:-1], fr[1:]):
            for f in range(a + 1, b):
                plan.setdefault(f, []).append((tid, a, b))
    return plan


def _frames_of(assignment: dict[int, np.ndarray]) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for t in sorted(assignment):
        for tid in np.unique(assignment[t]):
            out.setdefault(int(tid), []).append(int(t))
    return {k: sorted(v) for k, v in out.items()}


def _det_index_of(dets: Detections, t: int, tid: int,
                  assignment: dict[int, np.ndarray]) -> int | None:
    idx = np.where(np.asarray(assignment[t]) == tid)[0]
    return int(idx[0]) if idx.size else None


def export_ctc(run: PipelineRun, h5_path: str | Path, out_dir: str | Path,
               seq: str = "01", num_digits: int = 3,
               gt_h5: str | Path | None = None, gt_seg_dir: str | Path | None = None):
    """流式写出 CTC 结果（mask###.tif + res_track.txt）并累计本地诊断。

    **空洞帧补画（ENG_SUPP，需用户确认的口径点）**：CTC 官方格式要求轨迹在
    `[begin,end]` 内每帧都出现，而论文 §2.0.1 明确要求"帧 t 的 mask 仍为空，
    但整体轨迹不会被截断"。二者冲突时本函数按 `hole_policy="fill"` 把源帧实例的
    体素按质心插值位置**平移复制**到空洞帧（只写在当前为 0 的位置），
    使轨迹在 CTC 里合法且不被截断；`hole_policy="split"` 时不做补画。
    """
    import h5py

    from celltracker.eval.ctc_io import ResultWriter
    from celltracker.eval.local_metrics import StreamingDiagnostics, seg_measure

    dets, result = run.dets, run.result
    ts = dets.t_range
    out_dir = Path(out_dir)
    writer = ResultWriter(out_dir)
    plan = (_hole_plan(result.assignment, dets)
            if run.config.reconstruct.hole_policy == "fill" else {})
    gt_path = Path(gt_h5) if gt_h5 else Path(h5_path)

    filled = 0
    overwritten = 0
    unfilled = 0
    assignment = result.assignment
    with h5py.File(h5_path, "r") as f_det, h5py.File(gt_path, "r") as f_gt:
        gt_tracks = np.asarray(f_gt["tracks"]) if "tracks" in f_gt else None
        diag = StreamingDiagnostics(gt_tracks=gt_tracks)
        for t in ts:
            ids = result.assignment.get(t)
            if ids is None:
                continue
            canvas = np.asarray(f_det[f"frames/{t:04d}/labels"])
            res = paint_result(canvas, dets.label(t), ids)
            for tid, a, b in plan.get(t, []):
                status = _stamp_hole(res, f_det, dets, assignment, t, a, b, tid)
                if status == "filled":
                    filled += 1
                elif status == "overwritten":
                    filled += 1
                    overwritten += 1
                else:
                    unfilled += 1
            gt = np.asarray(f_gt[f"frames/{t:04d}/labels"]) if "frames" in f_gt else res
            writer.add(t, res)
            diag.add_frame(t, gt, res)
            del canvas, res
    writer.close(result.tracks)

    stats = diag.result()
    stats.update(diag.division_pr(result.tracks))
    stats.update({"n_tracks_pred": result.n_tracks(), "hole_frames_filled": filled,
                  "hole_frames_overwritten": overwritten,
                  "hole_frames_unfilled": unfilled,
                  "hole_policy": run.config.reconstruct.hole_policy,
                  "seq": seq})
    if gt_seg_dir is not None and Path(gt_seg_dir).is_dir():
        stats["local_SEG"] = seg_measure(str(gt_seg_dir), out_dir)
    (out_dir / "local_metrics.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2, default=str))
    return stats


def _stamp_hole(res: np.ndarray, f_det, dets: Detections,
                assignment: dict[int, np.ndarray],
                t: int, a: int, b: int, tid: int) -> str:
    """把源帧 (a) 里该轨迹实例的体素按插值质心平移复制到空洞帧 (t)。

    返回 `"filled"`（只写空闲体素）/ `"overwritten"`（无处可放，回退为覆盖）/
    `"failed"`（源实例缺体素）。

    ENG_SUPP 说明：CTC 官方格式要求轨迹在 [begin,end] 内每帧出现，而论文 §2.0.1
    要求"轨迹不被截断"；当平移后的体素全被别的检测占满时（密集帧常见），
    若坚持不覆盖就只能把轨迹拆断（退回 `hole_policy="split"`）。本实现选择
    **覆盖**并把次数记进 `hole_frames_overwritten`，让提交保持格式合法；
    该取舍需要用户确认（见 FORMULA_MAP.md 的 E-2）。
    """
    k_src = _det_index_of(dets, a, tid, assignment)
    k_tgt = _det_index_of(dets, b, tid, assignment)
    if k_src is None or k_tgt is None:
        return "failed"
    c_src = dets.centroid(a)[k_src]
    c_tgt = dets.centroid(b)[k_tgt]
    frac = (t - a) / max(b - a, 1)
    c_int = c_src + frac * (c_tgt - c_src)
    shift = np.round(c_int - c_src).astype(int)
    lab_src = np.asarray(f_det[f"frames/{a:04d}/labels"])
    src_label = int(dets.label(a)[k_src])
    coords = np.nonzero(lab_src == src_label)
    if not coords or coords[0].size == 0:
        return "failed"
    dst = [np.clip(c + s, 0, res.shape[k] - 1)
           for k, (c, s) in enumerate(zip(coords, shift))]
    free = res[tuple(dst)] == 0
    if np.any(free):
        res[tuple(d[free] for d in dst)] = tid
        return "filled"
    res[tuple(dst)] = tid          # 回退：覆盖（见 docstring 的取舍说明）
    return "overwritten"
