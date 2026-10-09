"""时间展开图与边监督（ideas.pdf 式25–29）。

h_i=[x_i,s_i,f_i,t/T]；时间候选满足 Γ_ij≥θΓ 且 C_ij≤θC；
e_ij=[C_ij,Γ_ij,‖x_i−x_j‖,s_i,s_j,Δt]。帧内双向 kNN 边加入空间/外观差异。
节点位置按视野尺寸归一化，物理体积取 log1p；边列的预处理在 EDGE_COLS 中命名。
窗口内所有相邻帧候选参与消息传递，只对指定源帧的直接/桥接边计算监督损失。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from celltracker.track.base import Detections
from ..config import GraphConfig, MeasureConfig
from ..coupling.pairwise import PairCoupling
from ..representation.measure import knn_structure, pairwise_distance

NODE_COLS = ["x0", "x1", "x2", "log1p_volume_um3", "appearance...", "t_norm"]
EDGE_COLS = ["C_ij", "Gamma_ij", "disp", "log1p_s_i", "log1p_s_j", "dt",
             "f_dist", "W_ik", "is_intra", "is_bridge"]
EDGE_DIM = len(EDGE_COLS)
EDGE_COST = EDGE_COLS.index("C_ij")
EDGE_MASS = EDGE_COLS.index("Gamma_ij")


def node_dim_for(f_source, encoder_dim=0):
    return 5 + (2 if f_source == "intensity" else int(encoder_dim))


@dataclass
class PairGraph:
    data: dict


def load_encoder_features(path, dets, spacing=None):
    """优先按实例 label 精确对齐；旧侧车按物理质心做一一匹配。

    相同实例数不能保证相同顺序；缺帧、重复身份、维度变化或非有限值都会报错。
    """
    out, width = {}, None
    with np.load(Path(path), allow_pickle=False) as z:
        for t in dets.t_range:
            key = f"frame_{t}"
            if key not in z:
                raise ValueError(f"encoder 特征缺帧 {t}")
            feats = np.asarray(z[key], dtype=np.float32)
            if feats.ndim != 2 or not np.isfinite(feats).all():
                raise ValueError(f"帧 {t} 特征形状/数值无效")
            if width is None:
                width = feats.shape[1]
            if feats.shape[1] != width:
                raise ValueError("encoder 特征维度跨帧不一致")
            if f"label_{t}" in z:
                labels = np.asarray(z[f"label_{t}"], dtype=int)
                if len(labels) != len(feats) or len(set(labels.tolist())) != len(labels):
                    raise ValueError("encoder 侧车实例身份重复或行数不符")
                lookup = {int(v): i for i,v in enumerate(labels)}
                try:
                    idx = np.array([lookup[int(v)] for v in dets.label(t)], dtype=int)
                except KeyError as exc:
                    raise ValueError(f"encoder 缺少实例 {exc}") from exc
            else:
                if f"centroid_{t}" not in z:
                    raise ValueError("encoder 侧车必须保存 label 或 centroid")
                cents = np.asarray(z[f"centroid_{t}"], dtype=float)
                if len(cents) != len(feats) or len(cents) != dets.n(t):
                    raise ValueError("encoder 与检测的实例数量不一致")
                distance = pairwise_distance(cents, dets.centroid(t), spacing)
                idx = np.argmin(distance, axis=0) if len(cents) else np.empty(0, dtype=int)
                if len(set(idx.tolist())) != len(idx) or (len(idx) and np.max(distance[idx, np.arange(len(idx))]) > 1e-3):
                    raise ValueError("encoder 质心无法与检测一一对齐")
            out[t] = feats[idx]
    return out


def _candidates(art, cfg, r_max, gap=1):
    plan, cost = art.plan, art.cost
    finite = np.isfinite(cost)
    mask = finite.copy()
    if cfg.use_ot_candidates:
        threshold = art.theta_gamma(cfg.theta_gamma_frac, cfg.theta_gamma)
        mask &= (cost <= art.theta_c(cfg.theta_c, r_max)) & (plan >= threshold[:,None]) & (plan > 0)
        if cfg.cand_topk > 0 and plan.size:
            k = min(cfg.cand_topk, plan.shape[1])
            order = np.argsort(-np.where(finite, plan, -1), axis=1)[:,:k]
            rows, cols = np.repeat(np.arange(len(plan)), k), order.ravel()
            valid = finite[rows,cols]
            mask[rows[valid], cols[valid]] = True
    i,j = np.where(mask)
    return np.column_stack([i,j]).astype(np.int64), plan[i,j], cost[i,j]


def _edge_features(disp, cost, gamma, s_src, s_dst, f_dist, dt, w_ik, kind):
    n = len(disp)
    return np.column_stack([cost, gamma, disp, np.log1p(s_src), np.log1p(s_dst), dt,
                            f_dist, w_ik, np.full(n, kind == "intra"),
                            np.full(n, kind == "bridge")]).astype(np.float32)


def _row_norm(a,b,spacing=None):
    delta = np.asarray(a)-np.asarray(b)
    if spacing is not None:
        delta = delta*np.asarray(spacing)
    return np.linalg.norm(delta, axis=1)


def _gt_ids(dets, t, i):
    values = dets.frames[t].get("gt_ids")
    return set(int(v) for v in (values[i] if values is not None else [dets.gt_label(t)[i]]) if v > 0)


def _edge_label(source_ids, target_ids, parent, gap):
    if source_ids & target_ids:
        return 1, 0
    for target in target_ids:
        ancestor = target
        for _ in range(gap):
            ancestor = int(parent.get(ancestor, 0))
            if ancestor == 0:
                break
            if ancestor in source_ids:
                return 1, 1
    return 0, 0


def build_graph(dets, t, ts, coupling_direct, coupling_bridge, cfg, mcfg,
                gt_parent=None, encoder_feats=None, spacing=None, r_max=3.0,
                context_couplings=None, context_jump=None):
    """构建一个带上下文的决策图。

    context_couplings 按 ts 的相邻帧索引取值；传入完整映射时历史帧才能参与消息传递。
    仅传一对耦合时生成局部图，不添加无法连接的历史节点。
    """
    pos = ts.index(t)
    win = cfg.ctx_window if cfg.context_enabled and context_couplings is not None else 0
    lo = max(0, pos-win)
    hi = min(len(ts)-1, pos+max(1, cfg.bridge_gap if cfg.bridge else 1))
    frames = ts[lo:hi+1]
    shape = np.asarray(dets.meta.get("shape", np.ones(3)), dtype=float)
    voxel_volume = float(np.prod(spacing)) if spacing is not None else 1.0
    appearance, volumes, offsets = {}, {}, {}
    nodes, node_frames, node_dets = [], [], []
    edges, features, targets = [], [], []
    offset = 0
    for f in frames:
        xy, n = dets.centroid(f), dets.n(f)
        volume = dets.volume(f)*voxel_volume
        if encoder_feats is not None:
            if f not in encoder_feats:
                raise ValueError(f"缺少帧 {f} 的外观特征")
            fv = encoder_feats[f]
        else:
            table = dets.frames[f]
            fv = np.column_stack([table.get("intensity_mean", np.zeros(n)),
                                  table.get("intensity_std", np.zeros(n))]).astype(float)/255
        appearance[f], volumes[f], offsets[f] = fv, volume, offset
        nodes.append(np.column_stack([xy/np.maximum(shape,1), np.log1p(volume), fv,
                                     np.full(n, f/max(ts[-1],1))]).astype(np.float32))
        node_frames.append(np.full(n,f,dtype=np.int64))
        node_dets.append(np.arange(n))
        if cfg.intra_enabled:
            _, W, adj = knn_structure(xy, dets.volume(f), replace(mcfg,knn_k=cfg.intra_knn), spacing)
            if not cfg.use_intra_similarity:
                W = np.zeros_like(W)
            i,j = np.where(adj)
            if len(i):
                edges.append(np.column_stack([i+offset,j+offset]))
                features.append(_edge_features(_row_norm(xy[i],xy[j],spacing),np.zeros(len(i)),
                    np.zeros(len(i)),volume[i],volume[j],_row_norm(fv[i],fv[j]),np.zeros(len(i)),W[i,j],"intra"))
                targets.append(np.zeros(len(i),dtype=bool))
        offset += n
    candidate_pairs, candidate_features, candidate_gaps = [], [], []
    for source_pos in range(lo,hi):
        source = ts[source_pos]
        direct = coupling_direct if source == t else (context_couplings or {}).get(source_pos)
        bridge = coupling_bridge if source == t else (context_jump or {}).get(source)
        for gap,art in [(1,direct),(cfg.bridge_gap,bridge)]:
            target = source+gap
            if art is None or target not in offsets or (gap > 1 and not cfg.bridge):
                continue
            pairs,gamma,cost = _candidates(art,cfg,r_max*gap,gap)
            if gap > 1 and cfg.bridge_scope == "missing_only" and len(pairs):
                # “节点缺失”由几何可达性判断，不把 OT 阈值拒绝误当成漏检。
                missing = ~np.isfinite(direct.cost).any(axis=1) if direct is not None else np.ones(dets.n(source),dtype=bool)
                keep = missing[pairs[:,0]]
                pairs,gamma,cost = pairs[keep],gamma[keep],cost[keep]
            if len(pairs) == 0:
                continue
            i,j = pairs.T
            feat = _edge_features(_row_norm(dets.centroid(source)[i],dets.centroid(target)[j],spacing),
                cost,gamma,volumes[source][i],volumes[target][j],
                _row_norm(appearance[source][i],appearance[target][j]),np.full(len(i),gap),np.zeros(len(i)),
                "bridge" if gap>1 else "time")
            if not cfg.use_ot_features:
                feat[:,[EDGE_COST,EDGE_MASS]] = 0
            is_target = source == t
            edges.append(pairs+np.array([offsets[source],offsets[target]]))
            features.append(feat)
            targets.append(np.full(len(i),is_target,dtype=bool))
            if is_target:
                candidate_pairs.append(pairs); candidate_features.append(feat)
                candidate_gaps.append(np.full(len(i),gap,dtype=np.int64))
    if not candidate_pairs:
        return {}
    pairs, feat, gap = np.concatenate(candidate_pairs), np.concatenate(candidate_features), np.concatenate(candidate_gaps)
    labels, div = np.zeros(len(pairs),dtype=np.int64), np.zeros(len(pairs),dtype=np.int64)
    if gt_parent is not None:
        for row,((i,j),dt) in enumerate(zip(pairs,gap)):
            labels[row],div[row] = _edge_label(_gt_ids(dets,t,int(i)),_gt_ids(dets,t+int(dt),int(j)),gt_parent,int(dt))
    node_feat = np.concatenate(nodes)
    return {"node_feat":node_feat,"node_frame":np.concatenate(node_frames),"node_det":np.concatenate(node_dets),
            "edge_index":np.concatenate(edges).T.astype(np.int64),"edge_feat":np.concatenate(features),
            "is_target":np.concatenate(targets),"cand_edges":pairs,"cand_feat":feat,"cand_gap":gap,
            "cand_label":labels,"cand_label_div":div,"label":labels,
            "t":np.int64(t),"n_src":np.int64(dets.n(t)),"n_dst":np.int64(dets.n(t+1)),
            "n_frames_in_graph":np.int64(len(frames)),"node_dim":np.int64(node_feat.shape[1]),"edge_dim":np.int64(EDGE_DIM),
            "support_start":np.int64(frames[0]),"support_end":np.int64(frames[-1]),
            "meta":{"t":t,"n_frames_in_graph":len(frames),"node_dim":node_feat.shape[1],"edge_dim":EDGE_DIM}}


def build_dataset(dets,couplings,jump,cfg,mcfg,out_dir,gt_parent=None,encoder_feats=None,spacing=None,r_max=3.0,
                  resume=False):
    if gt_parent is None:
        raise ValueError("生成训练图需要有效 GT 血缘；无分裂时可传空字典")
    out = Path(out_dir)
    out.mkdir(parents=True,exist_ok=True)
    if list(out.glob('pair_*.npz')) and not resume:
        raise FileExistsError(f"图目录非空，不能覆盖已有实验：{out}")
    count=nodes=positive=edges=0
    for pos,t in enumerate(dets.t_range[:-1]):
        g = build_graph(dets,t,dets.t_range,couplings.get(pos),jump.get(t),cfg,mcfg,gt_parent,
                        encoder_feats,spacing,r_max,context_couplings=couplings,context_jump=jump)
        if not g:
            continue
        path=out/f"pair_{t:04d}.npz"
        payload={k:v for k,v in g.items() if k!='meta'}
        if path.exists():
            with np.load(path) as previous:
                if set(previous.files)!=set(payload) or any(not np.array_equal(previous[k],v) for k,v in payload.items()):
                    raise ValueError(f'已有图与恢复配置不一致：{path}')
        else:
            pending=path.with_suffix('.pending.npz')
            np.savez_compressed(pending,**payload)
            pending.replace(path)
        count+=1; nodes+=len(g['node_feat']); positive+=int(g['label'].sum()); edges+=len(g['label'])
    stats={"n_graphs":count,"n_nodes":nodes,"n_positive_candidates":positive,"n_candidates":edges,"out_dir":str(out)}
    (out/'meta.json').write_text(json.dumps(stats,indent=2))
    return stats
