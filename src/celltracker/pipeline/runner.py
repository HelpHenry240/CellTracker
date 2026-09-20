"""统一 pipeline runner：按论文顺序串起各阶段，支持配置与消融开关。

阶段顺序（对应 ideas.pdf）::

  §1.2 测度与帧内图   →  Detections（来自 GT 标记点或 nnU-Net 分割）
  §1.3 相邻帧 OT      →  {Γ_t}      （式8/9/11/12/14，η/τ/ε 可切换）
  §1.5 运动先验       →  两遍式（α′>0 时先估速再重跑，式20-22）
  §1.4 多尺度时间正则 →  精炼后的 {Γ_t}（式17-19）
  §2.0.1 时间展开图   →  每对帧的图（式25-28，候选边由式26 筛）
  §2.0.1 GNN          →  边分类概率 → 轨迹（式30-35）
      或 §1.6 OT 规则 →  由 Γ 直接重建（GNN 关闭时的消融对照）
  §1.6 tracklet 二层 OT → 合并碎片/长程关联
  评测                →  本地诊断 + CTC 官方指标（云端）

所有阶段的中间产物都可落盘（`artifacts_dir`），便于复用与调试；
每个阶段都可经 `--ablate <name>` 关闭，形成论文所需的消融表。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..data.ctc import Track
from ..track.base import Detections, TrackResult, finalize_tracks
from .config import MeasureConfig, OTConfig, PipelineConfig
from .multiscale_stage import MultiscaleResult, refine_couplings
from .ot_stage import CouplingArtifacts, compute_pairwise_plan, save_coupling
from .tracklet_stage import TrackletResult, link_tracklets

__all__ = ["PipelineRun", "run_pipeline", "ot_rule_reconstruct"]


@dataclass
class PipelineRun:
    """一次 pipeline 运行的完整结果与中间产物。"""

    config: PipelineConfig
    dets: Detections
    couplings: dict[int, CouplingArtifacts]
    track_result: TrackResult
    ms_result: MultiscaleResult | None = None
    tracklet_result: TrackletResult | None = None
    graph_dir: Path | None = None
    artifacts_dir: Path | None = None
    info: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# §1.6（式23/24）OT 规则重建：GNN 关闭时的对照路径
# ---------------------------------------------------------------------------

def ot_rule_reconstruct(dets: Detections, couplings: dict[int, CouplingArtifacts],
                        cfg: PipelineConfig) -> TrackResult:
    """按式(23)(24) 直接从传输计划重建轨迹（不含 GNN）。

    - 关联：`j* = argmax_j Γ_ij`，且 `Γ_ij*/行和 ≥ θ_Γ`、`C_ij* ≤ θ_C`
    - 分裂：一行中出现 ≥2 个质量占比 ≥ `div_ratio` 的目标，则判为二分裂
    """
    ts = dets.t_range
    theta_g = cfg.graph.theta_gamma
    theta_c = cfg.ot.r_max
    out = TrackResult(meta={"reconstruct": "ot_rule"})
    assignment: dict[int, np.ndarray] = {}
    next_id = 1

    first = ts[0]
    ids0 = np.arange(next_id, next_id + dets.n(first))
    next_id += dets.n(first)
    assignment[first] = ids0

    for pos, (t, t_next) in enumerate(zip(ts[:-1], ts[1:])):
        art = couplings.get(pos)
        if art is None:
            assignment[t_next] = np.arange(next_id, next_id + dets.n(t_next))
            next_id += dets.n(t_next)
            continue
        plan, cost = art.plan, art.cost
        rn = art.rnorm
        src_ids = assignment[t]
        n_dst = dets.n(t_next)
        out_ids = np.zeros(n_dst, dtype=np.int64)
        claimed: set[int] = set()

        # 分裂优先（式1.6：一行对多个目标显著传输）
        # 阈值来自 ReconstructConfig.div_ratio —— 它与 ε 强耦合（A2 结论），
        # 不能复用 tracklet 阶段的字段（语义不同）。
        div_ratio = cfg.reconstruct.div_ratio
        for i in range(len(src_ids)):
            cands = [j for j in np.argsort(-rn[i]) if j not in claimed]
            sig = [j for j in cands if rn[i, j] >= div_ratio]
            if len(sig) < 2:
                continue
            for j in sig[:2]:
                out_ids[j] = next_id
                claimed.add(int(j))
                out.tracks[next_id] = Track(next_id, t_next, t_next, int(src_ids[i]))
                next_id += 1

        # 单目标关联
        for i in range(len(src_ids)):
            if rn[i].size == 0:
                continue
            j = int(np.argmax(plan[i]))
            if j in claimed or not np.isfinite(cost[i, j]):
                continue
            if rn[i, j] < theta_g or cost[i, j] > theta_c ** 2:
                continue
            out_ids[j] = int(src_ids[i])
            claimed.add(j)

        # 其余：新生
        for j in range(n_dst):
            if out_ids[j] == 0:
                out_ids[j] = next_id
                out.tracks[next_id] = Track(next_id, t_next, t_next, 0)
                next_id += 1
        assignment[t_next] = out_ids

    assignment, tracks, fix = finalize_tracks(assignment, out.tracks)
    out.assignment, out.tracks = assignment, tracks
    out.meta.update(fix)
    return out


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def run_pipeline(h5_path: str | Path, cfg: PipelineConfig,
                 frames: list[int] | None = None,
                 artifacts_dir: str | Path | None = None,
                 gnn=None,
                 dump_graphs: str | Path | None = None,
                 gt_h5: str | Path | None = None) -> PipelineRun:
    """按论文顺序执行 pipeline。

    `gnn`：为 None 时走 §1.6 的 OT 规则重建（消融对照）；
           否则应为可调用对象 `gnn(dets, couplings, cfg) -> TrackResult`，
           由 `gnn/` 侧提供（Phase B 接入训练好的模型）。

    `dump_graphs`：给定目录时，把**本次耦合**导出的图数据集写盘
    （格式与 `graph.build_dataset` 一致，可直接喂给 `run_gnn.py train`）。

    `gt_h5`：GT 血缘所在文件（`tracks` 表）。检测来自预测 h5 时**必须**提供，
    否则 `gt_parent` 为空 → 分裂边无法标注（历史上还会因此产生错误标签）。
    消融实验必须走这条路径，才能保证"训练用的图"与"评测用的耦合"同源。
    """
    h5_path = Path(h5_path)
    dets = Detections.from_h5(h5_path)
    if frames is not None:
        keep = set(frames)
        dets = Detections({t: d for t, d in dets.frames.items() if t in keep})
    ts = dets.t_range

    art_dir = Path(artifacts_dir) if artifacts_dir else None
    if art_dir:
        art_dir.mkdir(parents=True, exist_ok=True)

    # ---- §1.5 运动先验：α′>0 时先做一遍不带先验的粗追踪来估速 ----
    info: dict = {"n_frames": len(ts), "ablated": list(cfg.ablated)}
    if cfg.ot.alpha_pred > 0 and "motion" not in cfg.ablated:
        coarse = _ot_only_run(dets, cfg, alpha_pred=0.0)
        from .motion import attach_velocity, estimate_velocity, velocity_feature_is_live
        vel, valid = estimate_velocity(dets, coarse)
        attach_velocity(dets, vel, valid)
        info["velocity_live"] = velocity_feature_is_live(dets)
        info["n_with_predecessor"] = int(sum(int(v.sum()) for v in valid.values()))

    # ---- §1.3 相邻帧 OT（含 FGW η、非平衡 τ、自适应 ε）----
    couplings: dict[int, CouplingArtifacts] = {}
    for pos, (t, t_next) in enumerate(zip(ts[:-1], ts[1:])):
        pred_xy = None
        v = dets.frames[t].get("velocity")
        if v is not None and cfg.ot.alpha_pred > 0:
            pred_xy = dets.centroid(t) + np.asarray(v, dtype=float)
        couplings[pos] = compute_pairwise_plan(
            dets.centroid(t), dets.centroid(t_next), cfg.ot,
            dets.volume(t), dets.volume(t_next), pred_xy=pred_xy,
            measure=cfg.measure)
    info["n_couplings"] = len(couplings)
    if art_dir:
        np.savez_compressed(art_dir / "couplings.npz",
                            **{f"plan_{i}": c.plan for i, c in couplings.items()})

    # ---- §1.4 多尺度时间正则（式17-19）----
    ms_result = refine_couplings(dets, couplings, cfg.multiscale, cfg.ot, cfg.measure) \
        if cfg.multiscale.enabled else None
    if ms_result is not None:
        couplings = ms_result.couplings
        info["multiscale"] = ms_result.info

    # ---- §2.0.1 决策：GNN 或 OT 规则 ----
    graph_dir = None
    if gnn is not None and "gnn" not in cfg.ablated:
        track_result, graph_dir = gnn(dets, couplings, cfg)
        info["decision"] = "gnn"
    else:
        track_result = ot_rule_reconstruct(dets, couplings, cfg)
        info["decision"] = "ot_rule"

    # ---- 可选：导出一致的图数据集（供 GNN 训练；消融实验必需）----
    if dump_graphs is not None:
        graph_dir = Path(dump_graphs)
        graph_dir.mkdir(parents=True, exist_ok=True)
        n_written = _dump_graphs(dets, couplings, cfg, graph_dir, h5_path, gt_h5)
        info["dumped_graphs"] = {"dir": str(graph_dir), "n_pairs": n_written}
    info["tracks_after_decision"] = track_result.n_tracks()

    # ---- §1.6 第二层 tracklet OT ----
    tracklet_result = None
    if cfg.tracklet.enabled and "tracklet" not in cfg.ablated:
        tracklet_result = link_tracklets(dets, track_result, cfg.tracklet)
        track_result = tracklet_result.result
        info["tracklet"] = tracklet_result.info
        info["tracks_final"] = track_result.n_tracks()
    else:
        info["tracks_final"] = track_result.n_tracks()

    if art_dir:
        (art_dir / "run_info.json").write_text(
            json.dumps(info, ensure_ascii=False, indent=2, default=str))
    return PipelineRun(config=cfg, dets=dets, couplings=couplings,
                       track_result=track_result, ms_result=ms_result,
                       tracklet_result=tracklet_result, graph_dir=graph_dir,
                       artifacts_dir=art_dir, info=info)


def _ot_only_run(dets: Detections, cfg: PipelineConfig, alpha_pred: float) -> TrackResult:
    """粗追踪（仅用于给两遍式估速）：不启用多尺度/tracklet。"""
    from dataclasses import replace

    cfg1 = replace(cfg, ot=replace(cfg.ot, alpha_pred=alpha_pred))
    ts = dets.t_range
    couplings = {
        pos: compute_pairwise_plan(dets.centroid(t), dets.centroid(t_next), cfg1.ot,
                                   dets.volume(t), dets.volume(t_next),
                                   measure=cfg1.measure)
        for pos, (t, t_next) in enumerate(zip(ts[:-1], ts[1:]))}
    return ot_rule_reconstruct(dets, couplings, cfg1)


def _dump_graphs(dets: Detections, couplings, cfg: PipelineConfig,
                 graph_dir: Path, h5_path: Path,
                 gt_h5: str | Path | None = None) -> int:
    """把当前耦合导成图数据集（与 `graph.build_dataset` 同格式，可直接训练）。

    需要的 GT 血缘（边标签）从 h5 读；图构建配置由 pipeline 配置段映射
    （`gnn.adapter.graph_cfg_from_pipeline`，单一配置源）。
    """
    import h5py

    from ..graph.build import build_pair_graph
    from ..gnn.adapter import graph_cfg_from_pipeline

    graph_cfg = graph_cfg_from_pipeline(cfg)
    ts = dets.t_range
    with h5py.File(gt_h5 or h5_path, "r") as f:
        gt = np.asarray(f["tracks"]) if "tracks" in f else np.zeros(
            0, dtype=[("label", "i4"), ("begin", "i4"), ("end", "i4"), ("parent", "i4")])
    gt_parent = {int(l): int(p) for l, p in zip(gt["label"], gt["parent"])} if len(gt) else {}
    if not gt_parent:
        print("警告：GT lineage(tracks 表) 为空 → 分裂边无法标注，"
              "只能用 --gt-h5 指定 GT 文件（预测检测训练时必须提供）")
    shape = dets.meta.get("shape")

    n = 0
    for pos in sorted(couplings):
        if pos + 1 >= len(ts):
            continue
        t, t_next = ts[pos], ts[pos + 1]
        g = build_pair_graph(dets, t, t_next, graph_cfg, gt_parent, shape, len(ts),
                             coupling=couplings[pos])
        if not g:
            continue
        np.savez_compressed(graph_dir / f"pair_{t:04d}.npz", **g)
        n += 1
    import json
    (graph_dir / "meta.json").write_text(json.dumps(
        {"n_pairs": n, "config": {"graph": vars(cfg.graph), "ot": vars(cfg.ot),
                                  "ablated": list(cfg.ablated)},
         "source_h5": str(h5_path)}, ensure_ascii=False, indent=2))
    return n
