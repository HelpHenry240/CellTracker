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
    eps: float = 1.0
    eta: float = 0.0               # >0 时用 FGW（结构项进入 OT 特征）
    beta: float = 0.0
    node_feat_dim: int = 8
    edge_feat_dim: int = 10
    time_scale: float = 1.0


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
                     t_total: int | None = None) -> dict:
    """构造单对相邻帧的图（含 3 分类边标签）。"""
    cfg = cfg or GraphConfig()
    shape = np.asarray(shape if shape is not None else np.ones(3) * 1000.0, dtype=float)
    t_total = int(t_total or (t_next + 1))

    src_xy, dst_xy = dets.centroid(t), dets.centroid(t_next)
    src_vol, dst_vol = dets.volume(t), dets.volume(t_next)
    src_lab, dst_lab = dets.label(t), dets.label(t_next)
    imean_s = dets.frames[t].get("intensity_mean")
    istd_s = dets.frames[t].get("intensity_std")
    imean_d = dets.frames[t_next].get("intensity_mean")
    istd_d = dets.frames[t_next].get("intensity_std")

    n_src, n_dst = src_xy.shape[0], dst_xy.shape[0]
    if n_src == 0 or n_dst == 0:
        return {}

    # ---- OT 计划（作为边特征）----
    cost_cfg = CostConfig(r_max=cfg.r_max, beta=cfg.beta)
    C, info = build_cost(src_xy, dst_xy, src_vol, dst_vol, None, cost_cfg)
    a = masses(src_vol, n_src, "uniform")
    b = masses(dst_vol, n_dst, "uniform")
    plan = None
    if cfg.use_ot:
        if cfg.eta > 0:
            D, _ = gaussian_knn_graph(src_xy, k=cfg.knn)
            Dp, _ = gaussian_knn_graph(dst_xy, k=cfg.knn)
            plan, _ = fused_gw(C, a, b, D, Dp, eta=cfg.eta, eps=cfg.eps, n_outer=20)
        else:
            plan = sinkhorn_log(C, a, b, eps=cfg.eps)

    # ---- 候选边（R_max 门限内）----
    ii, jj = np.where(np.isfinite(C))
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
            if gl_s == gl_d:
                labels[k] = LABEL_MOVE
            elif gt_parent.get(gl_d, 0) == gl_s:
                labels[k] = LABEL_DIV

    # ---- 节点与帧内边 ----
    node_feat = np.concatenate([
        _node_features(src_xy, src_vol, imean_s, istd_s, t, shape, t_total),
        _node_features(dst_xy, dst_vol, imean_d, istd_d, t_next, shape, t_total),
    ], axis=0)
    node_frame = np.concatenate([np.full(n_src, t), np.full(n_dst, t_next)])
    node_det = np.concatenate([np.arange(n_src), np.arange(n_dst)])

    intra_src, intra_dst = [], []
    for offset, xy in ((0, src_xy), (n_src, dst_xy)):
        D, _ = gaussian_knn_graph(xy, k=min(cfg.knn, max(xy.shape[0] - 1, 1)))
        si, di = np.where(D > 0)
        intra_src.append(si + offset)
        intra_dst.append(di + offset)
    intra = np.stack([np.concatenate(intra_src), np.concatenate(intra_dst)], axis=1) \
        if intra_src and intra_src[0].size else np.zeros((0, 2), dtype=np.int64)

    return {
        "node_feat": node_feat,
        "node_frame": node_frame.astype(np.int64),
        "node_det": node_det.astype(np.int64),
        "cand_edges": pairs.astype(np.int64),        # (E,2) 指向节点下标（目标 +n_src）
        "cand_feat": edge_feat,
        "cand_label": labels,
        "intra_edges": intra.astype(np.int64),
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
