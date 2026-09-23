"""严格论文口径的完整 pipeline（ideas.pdf §1.2 → §2.0.1 的顺序串联）。

阶段顺序（与原文一致）
---------------------
§1.2 测度与帧内图（式1-7） → 检测表 `Detections`
§1.5 运动先验（式20-22）  → 两遍式：第1遍 α′=0 估速 → 第2遍 α′>0
§1.3 相邻帧 OT（式8-14）  → {Γ^t}
§1.4 多尺度时间正则（式17-19） → 精炼 {Γ^t}，同时产出跳帧耦合 Γ^{t,t+k}_direct
§1.6 轨迹重建（式23-24）  → 顶点/边约束下的硬关联（OT 规则路径）
§2.0.1 时间展开图 + GNN（式25-35）→ 边分类决策（推荐路径）
§1.5/§1.6 第二层 tracklet OT → 长程串联
导出 CTC 结果（含空洞帧补画，见 `export_ctc`）

产物与留痕
----------
`run_pipeline` 返回 `PipelineRun`：配置、检测表、{Γ^t}、跳帧耦合、轨迹结果、
运行信息（含各项诊断计数）。可选落盘：`artifacts_dir`（耦合 + info）、
`dump_graphs`（图数据集，供训练 GNN；**训练与推理用同一检测来源**，R11）。
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from celltracker.track.base import Detections, TrackResult, paint_result

from .config import PipelineConfig
from .coupling import PairCoupling, solve_coupling, solve_jump_coupling
from .graph import build_dataset
from .measure import resolve_spacing
from .motion import attach_velocity, estimate_velocity, velocity_report
from .multiscale import refine_couplings
from .reconstruct import reconstruct_from_edges, reconstruct_from_ot
from .tracklet import link_tracklets
from .tracks import holes_of, normalize_tracks

__all__ = ["PipelineRun", "run_pipeline", "load_detections", "export_ctc"]


# ---------------------------------------------------------------------------
# 检测表载入（含原仓库 `from_h5` 的一个已知缺口的修补）
# ---------------------------------------------------------------------------

def load_detections(h5_path: str | Path) -> Detections:
    """读检测表（复用 `Detections.from_h5`）。

    ⚠️ 修补一个已核实的缺口（审计 B2）：原 `Detections.from_h5` 只读
    `intensity_mean`，**没有读 `intensity_std`**，导致式(25) 节点特征里强度标准差
    那一维恒为 0（实测 std=0.00000），同时 h5 里其实存了该字段
    （`data/build_dataset.py` / `scripts/predict_to_h5.py` 都写）。
    本函数在载入后补齐这一列，使式(25) 的 f_i 两个分量都真实生效。
    """
    import h5py

    dets = Detections.from_h5(h5_path)
    with h5py.File(h5_path, "r") as f:
        for key in f["frames"].keys():
            t = int(key)
            if t not in dets.frames:
                continue
            g = f["frames"][key]
            if "intensity_std" in g:
                dets.frames[t]["intensity_std"] = np.asarray(g["intensity_std"],
                                                            dtype=float)
            if "gt_label" in g:
                dets.frames[t]["gt_label"] = np.asarray(g["gt_label"], dtype=np.int64)
    return dets


def load_gt_parent(gt_h5: str | Path) -> dict[int, int]:
    """读 GT 血缘（式29 的边标签需要；预测检测的 h5 没有 tracks 表）。"""
    import h5py

    with h5py.File(gt_h5, "r") as f:
        if "tracks" not in f:
            return {}
        tr = np.asarray(f["tracks"])
    return {int(l): int(p) for l, p in zip(tr["label"], tr["parent"])}


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

@dataclass
class PipelineRun:
    config: PipelineConfig
    dets: Detections
    couplings: dict[int, PairCoupling]
    jump: dict[int, PairCoupling]
    result: TrackResult
    spacing: tuple[float, float, float] | None = None
    holes: dict[int, list[int]] = field(default_factory=dict)
    info: dict = field(default_factory=dict)
    artifacts_dir: Path | None = None


def run_pipeline(h5_path: str | Path, cfg: PipelineConfig,
                 gt_h5: str | Path | None = None,
                 ckpt: str | Path | None = None,
                 frames: list[int] | None = None,
                 artifacts_dir: str | Path | None = None,
                 dump_graphs: str | Path | None = None,
                 device: str = "cpu",
                 raw_image: str | Path | None = None,
                 verbose: bool = True) -> PipelineRun:
    """按论文顺序跑完整链路。

    决策路径：`ckpt` 给出时走 §2.0.1 的 GNN（式33）；否则走 §1.6 的 OT 规则
    （式23/24），后者可用于"未训练模型时的端到端冒烟"。
    """
    h5_path = Path(h5_path)
    t_start = time.time()
    stage_seconds: dict[str, float] = {}

    def _say(msg: str, key: str | None = None) -> None:
        """阶段进度（E11：长任务要能看到进度，长跑时每阶段一行）。"""
        nonlocal t_start
        now = time.time()
        if key is not None:
            stage_seconds[key] = round(now - t_start, 1)
            t_start = now
            if verbose:
                print(f"[{time.strftime('%H:%M:%S')}] {key}: {msg} "
                      f"({stage_seconds[key]:.1f}s)", flush=True)
        elif verbose:
            print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    dets = load_detections(h5_path)
    if frames is not None:
        keep = set(frames)
        dets = Detections({t: d for t, d in dets.frames.items() if t in keep},
                          meta=dets.meta)
    ts = dets.t_range
    if len(ts) < 2:
        raise ValueError("至少需要 2 帧")
    _say(f"载入检测：{len(ts)} 帧", "load")

    mcfg = cfg.measure
    # 式(22) 的 α′ 由 §1.5 的配置给出；它进入 C_feat，故覆盖到 coupling 配置
    ccfg = replace(cfg.coupling,
                   alpha_pred=(cfg.motion.alpha_pred if cfg.motion.enabled else 0.0))
    spacing = resolve_spacing(mcfg.spacing_zyx, h5_path, raw_image)
    info: dict = {"n_frames": len(ts), "decision": "gnn" if ckpt else "ot_rule",
                  "detection_source": cfg.detection_source,
                  "spacing_zyx": list(spacing) if spacing else None,
                  "spacing_source": ("config" if mcfg.spacing_zyx else
                                     ("h5" if spacing else "none(voxel units!)")),
                  "gt_h5": str(gt_h5) if gt_h5 else None,
                  "mass_mode": mcfg.mass_mode}
    if spacing is None:
        info["warning"] = ("未解析到体素间距 → 距离按体素单位计算，"
                           "违反 R3（物理量必须按 spacing 换算）")
    art_dir = Path(artifacts_dir) if artifacts_dir else None
    if art_dir:
        art_dir.mkdir(parents=True, exist_ok=True)

    # ---- §1.5 运动先验：两遍式（第 1 遍 α′=0 建立硬关联 → 估速 → 第 2 遍）----
    pred_xy: dict[int, np.ndarray] = {}
    if cfg.motion.enabled and cfg.motion.alpha_pred > 0:
        coarse = _solve_all(dets, replace(ccfg, alpha_pred=0.0), mcfg, spacing)
        result_coarse = reconstruct_from_ot(dets, coarse, cfg.reconstruct, ccfg.r_max,
                                            bridge=False)
        vel, valid = estimate_velocity(dets, result_coarse)
        attach_velocity(dets, vel, valid)
        for t in ts:
            pred_xy[t] = dets.centroid(t) + np.asarray(vel[t], dtype=float)
        info["motion"] = {"pass1_tracks": result_coarse.n_tracks(),
                          "alpha_pred": cfg.motion.alpha_pred,
                          "velocity": velocity_report(dets)}
        _say(f"第 1 遍（α′=0）估速完成：{result_coarse.n_tracks()} 条轨迹", "motion_pass1")

    # ---- §1.3 相邻帧 OT（式8-14）----
    couplings = _solve_all(dets, ccfg, mcfg, spacing, pred_xy=pred_xy)
    info["n_couplings"] = len(couplings)
    info["coupling_eps"] = sorted({round(c.eps_eff, 6) for c in couplings.values()})
    _say(f"第 2 遍 OT 完成：{len(couplings)} 对，ε={info['coupling_eps'][:3]}", "couplings")

    # ---- §1.4 多尺度时间正则（式17-19）+ 跳帧耦合（式18/19 的 Γ^{t,t+k}_direct）----
    frame_xy = {t: dets.centroid(t) for t in ts}
    frame_vol = {t: dets.volume(t) for t in ts}
    ms = refine_couplings(couplings, ts, frame_xy, frame_vol, cfg.multiscale,
                          ccfg, mcfg, spacing, pred_xy=pred_xy)
    couplings = ms.couplings
    info["multiscale"] = {k: v for k, v in ms.info.items() if k != "history"}
    if ms.history:
        info["multiscale"]["history"] = ms.history

    jump: dict[int, PairCoupling] = {}
    need_gap = int(cfg.graph.bridge_gap)
    if cfg.graph.bridge or cfg.multiscale.enabled:
        for pos, t in enumerate(ts):
            if pos + need_gap <= len(ts) - 1:
                cached = ms.direct.get((pos, need_gap))
                if cached is not None:
                    jump[t] = cached
                else:
                    jump[t] = solve_jump_coupling(
                        frame_xy[t], frame_xy[ts[pos + need_gap]], need_gap, ccfg,
                        mcfg, frame_vol[t], frame_vol[ts[pos + need_gap]], spacing,
                        eps=couplings[pos].eps_eff)
    info["n_jump_couplings"] = len(jump)
    _say(f"多尺度精炼完成：跳帧耦合 {len(jump)} 个", "multiscale")

    # ---- §2.0.1 决策 ----
    encoder_feats = _load_encoder_features(cfg, dets, info)
    if ckpt:
        from .infer import make_edge_decider

        decider = make_edge_decider(ckpt, cfg, device=device, mcfg=mcfg,
                                    spacing=spacing)
        decisions = decider(dets, couplings, jump, encoder_feats=encoder_feats,
                            r_max=ccfg.r_max)
        info["n_decision_pairs"] = len(decisions)
        result = reconstruct_from_edges(dets, decisions, cfg.reconstruct)
    else:
        result = reconstruct_from_ot(dets, couplings, cfg.reconstruct, ccfg.r_max,
                                     jump=jump, bridge=cfg.graph.bridge)
    _say(f"决策（{info['decision']}）完成：{result.n_tracks()} 条轨迹", "decision")
    info["tracks_after_decision"] = result.n_tracks()

    # ---- §1.5 末段 / §1.6 第二层 tracklet OT ----
    if cfg.tracklet.enabled:
        tl = link_tracklets(dets, result, cfg.tracklet, ccfg, mcfg, spacing)
        result = tl.result
        info["tracklet"] = tl.info
    info["tracks_final"] = result.n_tracks()

    holes = holes_of(result.assignment)
    info["holes"] = {"n_tracks_with_holes": len(holes),
                     "n_hole_frames": int(sum(len(v) for v in holes.values())),
                     "hole_policy": cfg.reconstruct.hole_policy}
    info["reconstruct"] = dict(result.meta)

    # ---- 可选落盘 ----
    if art_dir:
        np.savez_compressed(art_dir / "couplings.npz",
                            **{f"plan_{i}": c.plan for i, c in couplings.items()})
        (art_dir / "run_info.json").write_text(
            json.dumps(info, ensure_ascii=False, indent=2, default=str))
    if dump_graphs is not None:
        gt_parent = load_gt_parent(gt_h5) if gt_h5 else {}
        if not gt_parent:
            print("警告：GT 血缘为空 → 式(29) 的边标签无法生成，"
                  "请用 --gt-h5 指定 GT 文件（R11：训练/推理检测来源必须同分布）")
        stats = build_dataset(dets, couplings, jump, cfg.graph, mcfg, dump_graphs,
                             gt_parent=gt_parent or None, encoder_feats=encoder_feats,
                             spacing=spacing, r_max=ccfg.r_max)
        info["graph_dataset"] = stats
        (Path(dump_graphs) / "run_info.json").write_text(
            json.dumps(info, ensure_ascii=False, indent=2, default=str))
        _say(f"图数据集落盘：{stats['n_graphs']} 张", "dump_graphs")

    info["stage_seconds"] = stage_seconds
    return PipelineRun(config=cfg, dets=dets, couplings=couplings, jump=jump,
                       result=result, spacing=spacing, holes=holes, info=info,
                       artifacts_dir=art_dir)


def _solve_all(dets: Detections, ccfg, mcfg, spacing, pred_xy=None
               ) -> dict[int, PairCoupling]:
    ts = dets.t_range
    out: dict[int, PairCoupling] = {}
    for pos, (t, t_next) in enumerate(zip(ts[:-1], ts[1:])):
        out[pos] = solve_coupling(
            dets.centroid(t), dets.centroid(t_next), dets.volume(t),
            dets.volume(t_next), None if pred_xy is None else pred_xy.get(t),
            ccfg, mcfg, spacing)
    return out


def _load_encoder_features(cfg: PipelineConfig, dets: Detections, info: dict):
    """式(25) 的 f_i：走论文口径（encoder 特征侧车文件）时载入，否则返回 None。"""
    if cfg.node.f_source != "encoder_npz" or not cfg.node.encoder_feat_path:
        info["f_source"] = "intensity(工程近似：实例内强度均值/标准差)"
        return None
    from .graph import load_encoder_features

    feats = load_encoder_features(cfg.node.encoder_feat_path, dets)
    info["f_source"] = f"encoder_npz:{cfg.node.encoder_feat_path}"
    info["encoder_feature_frames"] = len(feats)
    return feats


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
