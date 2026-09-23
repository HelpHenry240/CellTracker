"""时间展开图构建（ideas.pdf §2.0.1）。

每对相邻帧构造成一张小图：
  - **节点**：两帧的检测（位置 / 体积 / 强度 / 归一化时间）
  - **跨帧候选边**：`‖x_i - x_{t+1,j}‖ ≤ R_max` 的全部配对，边特征编码
    OT 成本、OT 传输质量、位移、尺寸比、运动先验残差、是否为行内 argmax
  - **帧内边**：kNN 邻接（提供局部拓扑上下文）
  - **标签（3 分类）**：0=无关联，1=同一细胞继续（movement），2=分裂（division）

关键：CTC 的 AOGM 区分"移动边"与"分裂边"语义，所以把分裂当作**独立类别**，
而不是让模型只做"是否相连"的二分类 —— 这样正好对应 §2.0.1 的论断。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..cost.features import CostConfig, build_cost, gaussian_knn_graph, masses
from ..ot.fgw import fused_gw
from ..ot.sinkhorn import sinkhorn_log
from ..track.base import Detections

__all__ = ["GraphConfig", "build_pair_graph", "build_dataset"]

LABEL_NONE, LABEL_MOVE, LABEL_DIV = 0, 1, 2


@dataclass
class GraphConfig:
    r_max: float = 30.0            # 候选位移门限（体素）
    knn: int = 4                   # 帧内 kNN 度数
    use_ot: bool = True            # 是否用 OT 计划作为边特征
    # ---- 式 (26)：候选边由 OT 传输计划筛选 ----
    cand_from_ot: bool = True      # True = 按式(26)筛；False = 纯 R_max 几何门控（消融对照）
    theta_gamma: float = 0.05      # 式(26)：行归一化传输质量下限 Γ_ij ≥ θ_Γ
    theta_c: float | None = None   # 式(26)：代价上限 C_ij ≤ θ_C；None 时取 r_max²
    eps_rel: float | None = 0.1    # 熵正则按代价尺度自适应：eps = eps_rel × median(C)
                                   # （代价是平方距离、量级 ~10²；直接给 eps=1 会让
                                   #   传输计划退化成硬分配，失去 §1.3 的"软匹配"前提）
    cand_topk: int = 3             # 式(26) 之外的保底：每行额外保留传输质量前 k 的目标。
                                   # 漏斗诊断显示：仅靠 Γ ≥ θ_Γ 会让约 21% 的真实分裂
                                   # 事件永久丢失一个子目标（GNN 无法挽回），
                                   # 加入 top-k 后该比例降到 ~16%（k=3）。
    window: int = 0                # 时间上下文窗口：额外纳入前后各 window 帧
    eps: float = 1.0
    eta: float = 0.0               # >0 时用 FGW（结构项进入 OT 特征）
    beta: float = 0.0
    node_feat_dim: int = 8
    edge_feat_dim: int = 10
    time_scale: float = 1.0
    # 代价与 R_max 的单位：给定 (z,y,x) µm 间距则按物理长度（r_max 单位随之变 µm）；
    # None = 体素单位（历史行为）。见 `cost/features.py::CostConfig.spacing_zyx`。
    spacing_zyx: tuple[float, float, float] | None = None


def _node_features(xy: np.ndarray, vol: np.ndarray, imean: np.ndarray,
                   istd: np.ndarray, t: int, shape: np.ndarray,
                   t_total: int) -> np.ndarray:
    """节点特征：(n, 8)。坐标按体数据尺寸归一化，体积取 log。"""
    xy_n = xy / np.maximum(shape[None, :], 1.0)
    feats = [xy_n,
             np.log1p(vol)[:, None],
             (imean / 255.0)[:, None] if imean is not None else np.zeros((xy.shape[0], 1)),
             (istd / 255.0)[:, None] if istd is not None else np.zeros((xy.shape[0], 1))]
    out = np.concatenate(feats, axis=1)
    tcol = np.full((xy.shape[0], 1), t / max(t_total - 1, 1))
    return np.concatenate([out, tcol], axis=1).astype(np.float32)


def _edge_feature_matrix(src_xy: np.ndarray, dst_xy: np.ndarray, pairs: np.ndarray,
                         src_vol: np.ndarray, dst_vol: np.ndarray,
                         cost: np.ndarray, plan: np.ndarray | None,
                         src_vel: np.ndarray | None) -> np.ndarray:
    """跨帧候选边的特征 (E, 10)。"""
    i, j = pairs[:, 0], pairs[:, 1]
    disp = dst_xy[j] - src_xy[i]
    d = np.linalg.norm(disp, axis=1)
    size_ratio = np.log1p(dst_vol[j]) - np.log1p(src_vol[i])
    c = cost[i, j]
    feats = [disp, d[:, None], size_ratio[:, None], np.nan_to_num(c)[:, None]]
    if src_vel is not None:
        pred = src_xy[i] + src_vel[i]
        d_pred = np.linalg.norm(pred - dst_xy[j], axis=1)
        feats.append(d_pred[:, None])
    else:
        feats.append(np.zeros((pairs.shape[0], 1)))
    if plan is not None:
        row_sum = plan.sum(axis=1, keepdims=True) + 1e-12
        rn = plan / row_sum
        mass = plan[i, j][:, None]
        rnorm = rn[i, j][:, None]
        # 是否为该行的 argmax
        argmax_j = np.argmax(plan, axis=1)
        is_argmax = (argmax_j[i] == j).astype(np.float64)[:, None]
        feats.extend([np.log1p(mass * 1000), rnorm, is_argmax])
    else:
        zeros = np.zeros((pairs.shape[0], 1))
        feats.extend([zeros, zeros, zeros])
    return np.concatenate(feats, axis=1).astype(np.float32)


def build_pair_graph(dets: Detections, t: int, t_next: int,
                     cfg: GraphConfig | None = None,
                     gt_parent: dict[int, int] | None = None,
                     shape: np.ndarray | None = None,
                     t_total: int | None = None,
                     coupling=None) -> dict:
    """构造单对相邻帧的图（含 3 分类边标签）。

    `cfg.window > 0` 时额外纳入前后各 window 帧作为**上下文**（其跨帧边不参与
    损失，只提供多步时间信息）——这正是 ideas.pdf §2.0.1 强调的"多步时间上下文"。

    `coupling`：可传入 `pipeline.ot_stage.CouplingArtifacts`，复用上游 OT 阶段
    已求好的传输计划（Γ）与代价矩阵（C），避免重复求解；为 None 时按 cfg 现场求解。
    """
    cfg = cfg or GraphConfig()
    all_ts = dets.t_range
    pos = all_ts.index(t)
    shape = np.asarray(shape if shape is not None else np.ones(3) * 1000.0, dtype=float)
    t_total = int(t_total or (t_next + 1))

    # 参与构图的时间帧（按时间排序）：[t-k ... t, t+1, ... t+1+k]
    win = max(cfg.window, 0)
    lo = max(0, pos - win)
    hi = min(len(all_ts) - 1, pos + 1 + win)
    win_frames = all_ts[lo: hi + 1]

    src_xy, dst_xy = dets.centroid(t), dets.centroid(t_next)
    src_vol, dst_vol = dets.volume(t), dets.volume(t_next)
    # 训练 GNN 时用 **GT 轨迹 id** 生成边标签（预测检测的 h5 用 `gt_label` 提供，
    # 未匹配的检测为 0 → 相关边自然成为负样本，对应假阳性）
    src_lab, dst_lab = dets.gt_label(t), dets.gt_label(t_next)
    imean_s = dets.frames[t].get("intensity_mean")
    istd_s = dets.frames[t].get("intensity_std")
    imean_d = dets.frames[t_next].get("intensity_mean")
    istd_d = dets.frames[t_next].get("intensity_std")

    n_src, n_dst = src_xy.shape[0], dst_xy.shape[0]
    if n_src == 0 or n_dst == 0:
        return {}

    # ---- §1.3 OT：优先复用上游 coupling，否则现场求解 ----
    if coupling is not None:
        C = np.asarray(coupling.cost, dtype=np.float64)
        plan = np.asarray(coupling.plan, dtype=np.float64) if cfg.use_ot else None
        eps_eff = float(getattr(coupling, "eps_eff", cfg.eps))
        info = dict(getattr(coupling, "info", {}) or {})
    else:
        from ..pipeline.config import MeasureConfig, OTConfig
        from ..pipeline.ot_stage import compute_pairwise_plan
        art = compute_pairwise_plan(
            src_xy, dst_xy,
            OTConfig(alpha=1.0, beta=cfg.beta, r_max=cfg.r_max, eta=cfg.eta,
                     eps=cfg.eps, eps_rel=cfg.eps_rel,
                     spacing_zyx=cfg.spacing_zyx),
            src_vol, dst_vol,
            measure=MeasureConfig(mass_mode="uniform", knn_k=cfg.knn))
        C, plan, eps_eff, info = art.cost, art.plan, art.eps_eff, art.info

    # ---- 候选边：式(26) 用 OT 传输计划筛选 ----
    #     E^t_time = { (i→j) | Γ_ij ≥ θ_Γ 且 C_ij ≤ θ_C }
    # OT 在此处是"决定候选边的人"（而不是仅提供特征），GNN 只在候选集内做判定。
    finite = np.isfinite(C)
    if cfg.cand_from_ot and plan is not None:
        row_sum = plan.sum(axis=1, keepdims=True) + 1e-12
        rnorm_full = plan / row_sum
        mask = (rnorm_full >= cfg.theta_gamma) & finite
        if cfg.theta_c is not None:
            mask &= (C <= cfg.theta_c)
        # 保底：每行至少保留传输质量最大的目标（式(23) 的 argmax）。
        # 否则源细胞可能一个候选都没有，必然产生缺失边（AOGM 罚 1.5）。
        # 同时保留每行质量前 k 的目标（漏斗诊断：分裂的第二子目标常因质量低被误滤）。
        #
        # 注意 k_eff：必须用**实际可用的列数** min(k, n_dst)。
        # 若按 k 直接构造行索引，在 n_dst < k 的稀疏帧上会与列索引长度不匹配
        # （实测：seq02 早期帧只有 2 个目标时抛 IndexError；seq01 每帧目标数 ≥3 未触发）。
        k = max(int(cfg.cand_topk), 1)
        order = np.argsort(-np.where(finite, plan, -1.0), axis=1)[:, :k]
        k_eff = order.shape[1]
        rows = np.repeat(np.arange(n_src), k_eff)
        cols = order.reshape(-1)
        ok = finite[rows, cols]
        mask[rows[ok], cols[ok]] = True
        ii, jj = np.where(mask)
    else:
        ii, jj = np.where(finite)
    pairs = np.stack([ii, jj], axis=1)

    # ---- 运动先验（用上一帧位移近似，若无则用 0）----
    src_vel = dets.frames[t].get("velocity")
    if src_vel is None:
        src_vel = np.zeros_like(src_xy)

    edge_feat = _edge_feature_matrix(src_xy, dst_xy, pairs, src_vol, dst_vol,
                                     C, plan, src_vel)

    # ---- 标签 ----
    labels = np.zeros(pairs.shape[0], dtype=np.int64)
    if gt_parent is not None:
        for k, (i, j) in enumerate(pairs):
            gl_s, gl_d = int(src_lab[i]), int(dst_lab[j])
            # **任何一端没有 GT 身份（gt_label=0，来自未匹配的检测/假阳性）时，
            # 这条边一律保持 NONE**：它既不是移动也不是分裂，而是"不该连"的负样本。
            # 曾经的写法 `gt_parent.get(gl_d, 0) == gl_s` 在 gt_parent 为空（预测 h5
            # 没有 tracks 表）且 gl_s=0 时退化成 `0 == 0`，把**所有假阳性源边标成分裂**，
            # 导致分裂头在垃圾标签上训练（C5.0 漏斗里分裂事件 S2 只有 13.8% 的元凶之一）。
            if gl_s == 0 or gl_d == 0:
                continue
            if gl_s == gl_d:
                labels[k] = LABEL_MOVE
            elif gt_parent.get(gl_d, 0) == gl_s:
                labels[k] = LABEL_DIV

    # ---- 节点（时间窗内全部帧）与帧内边 ----
    node_feats, node_frames, node_dets = [], [], []
    intra_src, intra_dst = [], []
    offset_of_frame: dict[int, int] = {}
    offset = 0
    for f in win_frames:
        xy = dets.centroid(f)
        vol = dets.volume(f)
        nf = _node_features(xy, vol, dets.frames[f].get("intensity_mean"),
                            dets.frames[f].get("intensity_std"), f, shape, t_total)
        node_feats.append(nf)
        node_frames.append(np.full(xy.shape[0], f))
        node_dets.append(np.arange(xy.shape[0]))
        offset_of_frame[f] = offset
        D, _ = gaussian_knn_graph(xy, k=min(cfg.knn, max(xy.shape[0] - 1, 1)))
        si, di = np.where(D > 0)
        intra_src.append(si + offset)
        intra_dst.append(di + offset)
        offset += xy.shape[0]
    node_feat = np.concatenate(node_feats, axis=0)
    node_frame = np.concatenate(node_frames)
    node_det = np.concatenate(node_dets)
    intra = np.stack([np.concatenate(intra_src), np.concatenate(intra_dst)], axis=1) \
        if intra_src and intra_src[0].size else np.zeros((0, 2), dtype=np.int64)

    # ---- 上下文帧之间的跨帧边（不参与损失，只提供多步信息）----
    ctx_edges: list[np.ndarray] = []
    ctx_feats: list[np.ndarray] = []
    # 回归修复（2026-09-23）：`cost_cfg` 此前**从未定义**，`window>0`（多步上下文）
    # 一进入下面的循环就 NameError —— 即 E4.4 之后那条消融路径其实是坏的。
    cost_cfg = CostConfig(r_max=cfg.r_max, beta=cfg.beta,
                          spacing_zyx=cfg.spacing_zyx)
    for f, f_next in zip(win_frames[:-1], win_frames[1:]):
        if (f, f_next) == (t, t_next):
            continue
        xy_a, xy_b = dets.centroid(f), dets.centroid(f_next)
        if xy_a.shape[0] == 0 or xy_b.shape[0] == 0:
            continue
        Cc, _ = build_cost(xy_a, xy_b, None, None, None, cost_cfg)
        ia, ib = np.where(np.isfinite(Cc))
        if ia.size == 0:
            continue
        pc = np.stack([ia, ib], axis=1)
        ef = _edge_feature_matrix(xy_a, xy_b, pc, dets.volume(f), dets.volume(f_next),
                                  Cc, None, None)
        ctx_edges.append(pc + np.array([offset_of_frame[f], offset_of_frame[f_next]]))
        ctx_feats.append(ef)

    # 目标边（当前对）在统一节点索引下的编号
    target_edges = pairs + np.array([offset_of_frame[t], offset_of_frame[t_next]])
    edge_index = np.concatenate([intra, *ctx_edges, target_edges], axis=0)
    cand_feat_p = np.concatenate(
        [edge_feat, np.zeros((edge_feat.shape[0], 1), dtype=np.float32)], axis=1)
    intra_feat = np.zeros((intra.shape[0], cand_feat_p.shape[1]), dtype=np.float32)
    if ctx_feats:
        # 上下文边：末位标志 = 2（区分帧内 1 / 目标 0）
        ctx_feat_m = np.concatenate(
            [np.concatenate(ctx_feats, axis=0),
             np.full((sum(f.shape[0] for f in ctx_feats), 1), 2.0, dtype=np.float32)],
            axis=1)
    else:
        ctx_feat_m = np.zeros((0, cand_feat_p.shape[1]), dtype=np.float32)
    if intra.shape[0]:
        intra_feat[:, -1] = 1.0
    edge_feat_all = np.concatenate([intra_feat, ctx_feat_m, cand_feat_p], axis=0)
    is_target = np.concatenate([
        np.zeros(intra.shape[0] + ctx_feat_m.shape[0], dtype=bool),
        np.ones(cand_feat_p.shape[0], dtype=bool)])

    return {
        "node_feat": node_feat,
        "node_frame": node_frame.astype(np.int64),
        "node_det": node_det.astype(np.int64),
        "edge_index": edge_index.T.astype(np.int64),   # (2, E)
        "edge_feat": edge_feat_all.astype(np.float32),
        "is_target": is_target,
        "cand_edges": pairs.astype(np.int64),          # (E_t, 2) 目标对内的局部下标
        "cand_feat": cand_feat_p,
        "cand_label": labels,
        "intra_edges": intra.astype(np.int64),
        "eps_eff": np.float64(eps_eff),
        "t": np.int64(t),
        "t_next": np.int64(t_next),
        "n_src": np.int64(n_src),
        "n_dst": np.int64(n_dst),
        "src_label": src_lab.astype(np.int64),
        "dst_label": dst_lab.astype(np.int64),
    }


def build_dataset(h5_path: str | Path, out_dir: str | Path,
                  cfg: GraphConfig | None = None,
                  frames: list[int] | None = None,
                  verbose: bool = True) -> Path:
    """把整个序列的所有相邻帧对构造成图并存成 `pair_XXXX.npz`。"""
    import h5py

    cfg = cfg or GraphConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dets = Detections.from_h5(h5_path)
    ts = dets.t_range if frames is None else [t for t in dets.t_range if t in set(frames)]
    with h5py.File(h5_path, "r") as f:
        shape = np.asarray(f.attrs["shape"], dtype=float)
        gt_tracks = np.asarray(f["tracks"])
    gt_parent = {int(l): int(p) for l, p in zip(gt_tracks["label"], gt_tracks["parent"])}

    n_written = 0
    for t, t_next in zip(ts[:-1], ts[1:]):
        g = build_pair_graph(dets, t, t_next, cfg, gt_parent, shape, len(ts))
        if not g:
            continue
        np.savez_compressed(out_dir / f"pair_{t:04d}.npz", **g)
        n_written += 1
        if verbose and n_written % 25 == 0:
            print(f"  已写出 {n_written} 对（当前 t={t}）")

    meta = {
        "n_pairs": n_written,
        "config": cfg.__dict__,
        "h5": str(h5_path),
    }
    import json
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    if verbose:
        print(f"图数据集 -> {out_dir}（{n_written} 对）")
    return out_dir
