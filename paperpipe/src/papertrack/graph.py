"""§2.0.1 时间展开动态图（ideas.pdf 式 25–29）。

原文公式（R1 抄录）
------------------
式(25)  h^(0)_{(i,t)} = [ x_i^t, s_i^t, f_i^t, t̃ ] ∈ R^{d_x+1+d_f+1}
        "x 为质心坐标，s 为细胞尺寸（面积/体积），f 为由分割网络（如 nnU-Net
         encoder）提取的外观特征向量，t̃ 为归一化的时间索引（如 t/T）"
式(26)  E^t_time = { ((i,t)→(j,t+1)) | Γ^t_ij ≥ θ_Γ, C^t_ij ≤ θ_C }
式(27)  e^(0)_e = [ C^t_ij, Γ^t_ij, ‖x_i^t−x_j^{t+1}‖₂, s_i^t, s_j^{t+1}, Δt ], Δt=1
式(28)  E^t_intra = { ((i,t)↔(k,t)) | x_k^t ∈ kNN(x_i^t) }
        "并为帧内边定义基于空间和特征差异的边特征（例如 ‖x_i^t−x_k^t‖₂、
         ‖f_i^t−f_k^t‖₂ 等）"
式(29)  y_e = 1, 若 c_i^t 为 c_j^{t+1} 的真实前驱细胞；0, 否则

§2.0.1 的桥接边（本包实现，原仓库缺失）
--------------------------------------
原文："若在 OT构图或后续轨迹拼接中允许跨帧候选，例如从 (i,t−1) 直接连到
(k,t+1)，则 GNN 可在训练过程中学习到'跨一帧的平滑连接'模式，从而在图中通过边
(i,t−1)→(k,t+1) 对分割空洞进行桥接，保持轨迹在时间上的连续性。"
→ 本模块把 `Δt=2` 的桥接边也放进 E_time（式27 的 Δt 分量在此处才有非 1 取值），
  其候选来自**帧 t 与 t+2 之间的直接 OT 耦合**（式18/19 的 Γ^{t,t+2}_direct，
  与多尺度正则共用同一个量）。

节点/边特征的实际布局（论文未逐维规定预处理，凡变换均在 FORMULA_MAP.md 登记）
--------------------------------------------------------------------------
节点 `node_feat`（式25）：[x/shape (3), log1p(s) (1), f (d_f), t̃ (1)]
  * `f_source="intensity"` 时 f = [强度均值/255, 强度标准差/255]（d_f=2）——**工程近似**；
  * `f_source="encoder_npz"` 时 f 取自侧车 npz（论文口径）。
边 `edge_feat`（式27/28 的统一 10 维布局）：
  0 C_ij | 1 Γ_ij | 2 ‖Δx‖₂ | 3 log1p(s_i) | 4 log1p(s_j) | 5 Δt
  6 ‖f_i−f_j‖₂ | 7 W_ik（式6，仅帧内边非零）
  8 intra 标志 | 9 bridge 标志
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from celltracker.track.base import Detections

from .config import GraphConfig, MeasureConfig
from .coupling import PairCoupling
from .measure import knn_structure, pairwise_distance, point_features

__all__ = ["NODE_COLS", "EDGE_COLS", "build_graph", "build_dataset",
           "load_encoder_features", "node_dim_for"]

NODE_COLS = ["x0", "x1", "x2", "log1p_s", "f0", "f1", "t_norm"]
EDGE_COLS = ["C_ij", "Gamma_ij", "disp", "log1p_s_i", "log1p_s_j", "dt",
             "f_dist", "W_ik", "is_intra", "is_bridge"]
EDGE_DIM = len(EDGE_COLS)


def node_dim_for(f_source: str, encoder_dim: int = 0) -> int:
    f_dim = 2 if f_source == "intensity" else int(encoder_dim)
    return 3 + 1 + f_dim + 1


@dataclass
class PairGraph:
    """一张时间展开图（源帧 t 的决策单元）。"""

    data: dict


def load_encoder_features(path: str | Path, dets: Detections) -> dict[int, np.ndarray]:
    """读取"分割网络 encoder 外观特征"侧车文件（式25 的 f_i，论文口径）。

    文件格式（npz）：`frame_<T>` = (n, d) 特征，按**质心最近邻**对齐到该帧检测。
    导出脚本需在云端跑 nnU-Net encoder 前向 + hook（见 README 的"待补"清单）。
    """
    z = np.load(Path(path))
    out: dict[int, np.ndarray] = {}
    for key in z.files:
        if not key.startswith("frame_"):
            continue
        t = int(key.split("_", 1)[1])
        feats = np.asarray(z[key], dtype=np.float32)
        if t not in dets.frames:
            continue
        cents = np.asarray(z.get(f"centroid_{t}", dets.centroid(t)), dtype=float)
        xy = dets.centroid(t)
        if cents.shape[0] == xy.shape[0]:
            out[t] = feats
            continue
        # 数目不一致 → 最近邻对齐（记录在调用方 info 里）
        d = pairwise_distance(cents, xy)
        idx = np.argmin(d, axis=0)
        out[t] = feats[idx]
    return out


def _candidates(art: PairCoupling, cfg: GraphConfig, r_max: float,
                gap: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """式(26)：由 OT 传输计划筛候选边。返回 (pairs, gamma, cost)。

    `theta_gamma_frac` 给出时 θ_Γ(i) = frac × a_i^t（CALIBRATED）；
    给出 `theta_gamma` 时按原文的单一绝对阈值。
    `cand_topk>0` 时额外保留每行质量前 k（**ENG_SUPP，默认 0 = 原文口径**）。
    `cfg.use_ot_candidates=False` 时退化为"只用 R_max 几何门控"的消融口径
    （此时仍沿用同一张 C 与 Γ，保证边特征口径不变）。
    """
    plan, cost = art.plan, art.cost
    finite = np.isfinite(cost)
    thr = art.theta_gamma(cfg.theta_gamma_frac, cfg.theta_gamma)
    theta_c = art.theta_c(cfg.theta_c, r_max)
    mask = finite & (cost <= theta_c)
    if cfg.use_ot_candidates:
        mask &= (plan >= thr[:, None]) & (plan > 0)
    if cfg.use_ot_candidates and int(cfg.cand_topk) > 0 and plan.size:
        k = min(int(cfg.cand_topk), plan.shape[1])
        order = np.argsort(-np.where(finite, plan, -1.0), axis=1)[:, :k]
        rows = np.repeat(np.arange(plan.shape[0]), order.shape[1])
        cols = order.reshape(-1)
        ok = finite[rows, cols]
        mask[rows[ok], cols[ok]] = True
    ii, jj = np.where(mask)
    return (np.stack([ii, jj], axis=1) if ii.size else np.zeros((0, 2), dtype=np.int64),
            plan[ii, jj] if ii.size else np.zeros(0),
            cost[ii, jj] if ii.size else np.zeros(0))


def _edge_features(disp: np.ndarray, cost: np.ndarray, gamma: np.ndarray,
                   s_src: np.ndarray, s_dst: np.ndarray, f_dist: np.ndarray,
                   dt: np.ndarray, w_ik: np.ndarray, kind: str) -> np.ndarray:
    """按式(27)/(28) 组装边特征（统一 10 维布局，见模块 docstring）。"""
    n = disp.shape[0]
    feats = [
        np.asarray(cost, dtype=np.float32).reshape(-1, 1),
        np.asarray(gamma, dtype=np.float32).reshape(-1, 1),
        np.asarray(disp, dtype=np.float32).reshape(-1, 1),
        np.log1p(np.asarray(s_src, dtype=np.float32)).reshape(-1, 1),
        np.log1p(np.asarray(s_dst, dtype=np.float32)).reshape(-1, 1),
        np.asarray(dt, dtype=np.float32).reshape(-1, 1),
        np.asarray(f_dist, dtype=np.float32).reshape(-1, 1),
        np.asarray(w_ik, dtype=np.float32).reshape(-1, 1),
        np.full((n, 1), 1.0 if kind == "intra" else 0.0, dtype=np.float32),
        np.full((n, 1), 1.0 if kind == "bridge" else 0.0, dtype=np.float32),
    ]
    return np.concatenate(feats, axis=1).astype(np.float32)


def _row_norm(a: np.ndarray, b: np.ndarray,
              spacing: tuple[float, ...] | None = None) -> np.ndarray:
    """逐对距离 ‖a_k − b_k‖₂（按 spacing 折算物理长度）——式(27)(28) 的位移列。"""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if spacing is not None:
        diff = diff * np.asarray(spacing, dtype=float)[None, :]
    return np.linalg.norm(diff, axis=1)


def build_graph(dets: Detections, t: int, ts: list[int],
                coupling_direct: PairCoupling | None,
                coupling_bridge: PairCoupling | None,
                cfg: GraphConfig, mcfg: MeasureConfig,
                gt_parent: dict[int, int] | None = None,
                encoder_feats: dict[int, np.ndarray] | None = None,
                spacing: tuple[float, ...] | None = None,
                r_max: float = 3.0) -> dict:
    """构造源帧 t 的时间展开图（式25-29 + §2.0.1 桥接边）。

    `coupling_direct`：帧对 (t, t+1) 的 Γ；`coupling_bridge`：帧对 (t, t+2) 的 Γ（式18/19）。
    """
    pos = ts.index(t)
    shape = np.asarray(dets.meta.get("shape", np.ones(3)), dtype=float)
    t_total = len(ts)
    win = max(int(cfg.ctx_window), 0)
    hi = min(pos + int(cfg.bridge_gap), len(ts) - 1)
    frames = ts[max(0, pos - win): hi + 1]
    if coupling_direct is None and coupling_bridge is None:
        return {}

    # ---- 节点（式25）与帧内边（式28）----
    node_feats, node_frame, node_det = [], [], []
    intra_src, intra_dst, intra_feat = [], [], []
    offset_of: dict[int, int] = {}
    offset = 0
    for f in frames:
        xy = dets.centroid(f)
        vol = dets.volume(f)
        n_f = xy.shape[0]
        if encoder_feats is not None and f in encoder_feats:
            fv = np.asarray(encoder_feats[f], dtype=np.float32)
        else:
            imean = dets.frames[f].get("intensity_mean")
            istd = dets.frames[f].get("intensity_std")
            fv = np.stack([
                np.asarray(imean, dtype=np.float32) / 255.0 if imean is not None
                else np.zeros(n_f, dtype=np.float32),
                np.asarray(istd, dtype=np.float32) / 255.0 if istd is not None
                else np.zeros(n_f, dtype=np.float32)], axis=1)
        nf = np.concatenate([
            xy / np.maximum(shape[None, :], 1.0),
            np.log1p(vol)[:, None].astype(np.float32),
            fv,
            np.full((n_f, 1), f / max(t_total - 1, 1), dtype=np.float32)], axis=1)
        node_feats.append(nf.astype(np.float32))
        node_frame.append(np.full(n_f, f))
        node_det.append(np.arange(n_f))
        D, W, adj = knn_structure(xy, vol, mcfg, spacing)
        si, di = np.where(adj)
        if si.size:
            disp = _row_norm(xy[si], xy[di], spacing)
            f_all = point_features(xy, None, spacing)
            f_dist = _row_norm(f_all[si], f_all[di])
            intra_src.append(si + offset)
            intra_dst.append(di + offset)
            intra_feat.append(_edge_features(
                disp=disp, cost=np.zeros(si.size), gamma=np.zeros(si.size),
                s_src=vol[si], s_dst=vol[di], f_dist=f_dist,
                dt=np.zeros(si.size), w_ik=W[si, di], kind="intra"))
        offset_of[f] = offset
        offset += n_f

    node_feat = np.concatenate(node_feats, axis=0) if node_feats else np.zeros((0, 7))
    node_fr = np.concatenate(node_frame) if node_frame else np.zeros(0, dtype=np.int64)
    node_dt = np.concatenate(node_det) if node_det else np.zeros(0, dtype=np.int64)
    intra = (np.stack([np.concatenate(intra_src), np.concatenate(intra_dst)], axis=1)
             if intra_src else np.zeros((0, 2), dtype=np.int64))
    intra_e = (np.concatenate(intra_feat, axis=0) if intra_feat
               else np.zeros((0, EDGE_DIM), dtype=np.float32))

    # ---- 目标候选边（式26：直接边 + 桥接边）----
    src_xy, src_vol = dets.centroid(t), dets.volume(t)
    direct_rows = None
    if coupling_direct is not None:
        pairs_d, _g, _c = _candidates(coupling_direct, cfg, r_max, 1)
        direct_rows = np.zeros(dets.n(t), dtype=bool)
        if pairs_d.shape[0]:
            direct_rows[pairs_d[:, 0]] = True
    tgt_all: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
    for gap, art in ((1, coupling_direct), (int(cfg.bridge_gap), coupling_bridge)):
        if art is None or (gap > 1 and not cfg.bridge):
            continue
        if dets.n(t + gap) == 0:
            continue
        pairs, gamma, cost = _candidates(art, cfg, r_max * gap, gap)
        if gap > 1 and cfg.bridge_scope == "missing_only" and pairs.shape[0]:
            # 原文的前置条件：帧 t+1 的**节点缺失**（连一个 R_max 内的候选都没有）
            # 才允许跨帧桥接；否则任何未被接受的边都会变成"空洞"，实测会产生
            # 大量隔帧轨迹（在真实数据上验证过）。
            missing = ~direct_rows if direct_rows is not None \
                else np.ones(dets.n(t), dtype=bool)
            keep = missing[pairs[:, 0]]
            pairs, gamma, cost = pairs[keep], gamma[keep], cost[keep]
        if pairs.shape[0] == 0:
            continue
        tgt_all.append((gap, pairs, gamma, cost))
    if not tgt_all:
        return {}

    tgt_xy_list, tgt_feat_list, tgt_pairs_list, tgt_gap_list = [], [], [], []
    for gap, pairs, gamma, cost in tgt_all:
        tn = t + gap
        xy_d, vol_d = dets.centroid(tn), dets.volume(tn)
        i, j = pairs[:, 0], pairs[:, 1]
        disp = _row_norm(src_xy[i], xy_d[j], spacing)
        f_all = point_features(src_xy, None, spacing)
        f_all_d = point_features(xy_d, None, spacing)
        f_dist = _row_norm(f_all[i], f_all_d[j])
        tgt_feat_list.append(_edge_features(
            disp=disp, cost=np.nan_to_num(cost), gamma=gamma, s_src=src_vol[i],
            s_dst=vol_d[j], f_dist=f_dist, dt=np.full(len(i), gap),
            w_ik=np.zeros(len(i)), kind="bridge" if gap > 1 else "target"))
        tgt_pairs_list.append(pairs)
        tgt_gap_list.append(np.full(len(i), gap, dtype=np.int64))
        tgt_xy_list.append((disp, i, j, tn))

    tgt_feat = np.concatenate(tgt_feat_list, axis=0)
    tgt_pairs = np.concatenate(tgt_pairs_list, axis=0)
    tgt_gap = np.concatenate(tgt_gap_list)

    # ---- 标签（式29）：源与目标是否为"真实前驱关系" ----
    n_cand = tgt_pairs.shape[0]
    labels = np.zeros(n_cand, dtype=np.int64)
    label_div = np.zeros(n_cand, dtype=np.int64)
    if gt_parent is not None and n_cand:
        src_gt = dets.gt_label(t)
        for k in range(n_cand):
            tn = t + int(tgt_gap[k])
            j = int(tgt_pairs[k, 1])
            gl_s = int(src_gt[int(tgt_pairs[k, 0])])
            gl_d = int(dets.gt_label(tn)[j])
            if gl_s == 0 or gl_d == 0:
                continue                      # 任一端无 GT 身份 ⇒ 负样本
            if gl_s == gl_d:
                labels[k] = 1
            elif int(gt_parent.get(gl_d, 0)) == gl_s:
                labels[k] = 1
                label_div[k] = 1              # 分裂边也是"真实前驱"（式29 二分类）

    # ---- 边的统一编号：帧内边在前，目标边在后 ----
    target_edges = np.stack([tgt_pairs[:, 0] + offset_of[t],
                             tgt_pairs[:, 1] + np.array(
                                 [offset_of[t + int(g)] for g in tgt_gap])], axis=1) \
        if n_cand else np.zeros((0, 2), dtype=np.int64)
    edge_index = np.concatenate([intra, target_edges], axis=0)
    edge_feat = np.concatenate([intra_e, tgt_feat], axis=0)
    is_target = np.concatenate([np.zeros(intra.shape[0], dtype=bool),
                               np.ones(n_cand, dtype=bool)])

    return {
        "node_feat": node_feat.astype(np.float32),
        "node_frame": node_fr.astype(np.int64),
        "node_det": node_dt.astype(np.int64),
        "edge_index": edge_index.T.astype(np.int64),          # (2, E)
        "edge_feat": edge_feat.astype(np.float32),
        "is_target": is_target,
        # 以下键名与 `celltracker.gnn.data.collate` 对齐，便于复用批处理
        "cand_edges": tgt_pairs.astype(np.int64),             # (E_t, 2) 局部检测下标
        "cand_feat": tgt_feat.astype(np.float32),
        "cand_label": labels,                                 # 式(29)
        "cand_label_div": label_div,                          # 诊断用（不参与损失）
        "cand_gap": tgt_gap.astype(np.int64),                 # Δt
        "label": labels,
        "meta": {"t": int(t), "n_src": int(dets.n(t)),
                 "n_dst": int(dets.n(t + 1)),
                 "n_frames_in_graph": len(frames),
                 "node_dim": int(node_feat.shape[1]),
                 "edge_dim": EDGE_DIM},
        # 顶层标量键：让 `celltracker.gnn.data.{PairDataset, collate}` 可直接复用
        "t": np.int64(t), "n_src": np.int64(dets.n(t)),
        "n_dst": np.int64(dets.n(t + 1)),
        "n_frames_in_graph": np.int64(len(frames)),
        "node_dim": np.int64(node_feat.shape[1]), "edge_dim": np.int64(EDGE_DIM),
    }


def build_dataset(dets: Detections, couplings: dict[int, PairCoupling],
                  jump: dict[int, PairCoupling], cfg: GraphConfig,
                  mcfg: MeasureConfig, out_dir: str | Path,
                  gt_parent: dict[int, int] | None = None,
                  encoder_feats: dict[int, np.ndarray] | None = None,
                  spacing: tuple[float, ...] | None = None,
                  r_max: float = 3.0) -> dict:
    """把整个序列的图写成 `pair_XXXX.npz`（可直接喂给训练脚本）。"""
    ts = dets.t_range
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_written = n_nodes = n_cand_pos = 0
    for pos, t in enumerate(ts[:-1]):
        g = build_graph(dets, t, ts, couplings.get(pos), jump.get(t), cfg, mcfg,
                        gt_parent=gt_parent, encoder_feats=encoder_feats,
                        spacing=spacing, r_max=r_max)
        if not g or g["cand_edges"].shape[0] == 0:
            continue
        payload = {k: v for k, v in g.items() if k != "meta"}
        np.savez_compressed(out / f"pair_{t:04d}.npz", **payload)
        n_written += 1
        n_nodes += int(g["node_feat"].shape[0])
        n_cand_pos += int(g["cand_label"].sum())
    stats = {"n_graphs": n_written, "n_nodes": n_nodes,
             "n_positive_candidates": n_cand_pos, "out_dir": str(out)}
    (out / "meta.json").write_text(__import__("json").dumps(stats, indent=2))
    return stats
