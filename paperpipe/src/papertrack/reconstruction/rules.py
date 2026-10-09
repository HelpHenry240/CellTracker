"""从边决策恢复轨迹和谱系（ideas.pdf 式23/24、§1.6、§2.0.1）。

OT 路径使用行最大质量及 Γ/C 双阈值；GNN 路径使用式(33) 的边概率。
两条路径共用生死判据、有限父子、体积守恒和空洞连接。冲突按短时间跨度、
高置信度排序，是将软关联离散化的工程规则；不额外引入学习目标。
"""
from __future__ import annotations

from dataclasses import replace
import numpy as np
from celltracker.data.ctc import Track
from celltracker.track.base import TrackResult
from .tracks import normalize_tracks


def volume_consistent(s_parent, s_children, tol):
    """§1.6：|Σs_child−s_parent|≤tol×s_parent。"""
    return bool(s_children and s_parent > 0 and abs(float(np.sum(s_children))-s_parent) <= tol*s_parent)


def mass_flags(art, cfg):
    """§1.6：行/列质量不足分别标记终止/起点；relative 为工程对照。"""
    if not cfg.birth_death_enabled:
        return np.zeros(len(art.mass_a),dtype=bool),np.zeros(len(art.mass_b),dtype=bool)
    a = art.mass_a if cfg.mass_threshold_mode == "relative" else np.ones_like(art.mass_a)
    b = art.mass_b if cfg.mass_threshold_mode == "relative" else np.ones_like(art.mass_b)
    return art.row_sum < cfg.eta_death*a, art.col_sum < cfg.eta_birth*b


def reconstruct_from_ot(dets,couplings,cfg,r_max,jump=None,bridge=True,bridge_gap=2):
    """式(23)/(24)：j*=argmax Γ；双阈值通过后才加入移动/分裂候选。"""
    decisions = {}
    for pos,t in enumerate(dets.t_range[:-1]):
        direct = couplings.get(pos)
        rows, scores, gaps = [], [], []
        for gap,art in [(1,direct),(bridge_gap,(jump or {}).get(t) if bridge else None)]:
            if art is None or art.plan.shape[1] == 0:
                continue
            threshold = art.theta_gamma(cfg.theta_gamma_frac,cfg.theta_gamma)
            limit = art.theta_c(cfg.theta_c,r_max*gap)
            for i,plan in enumerate(art.plan):
                if gap > 1 and direct is not None and np.isfinite(direct.cost[i]).any():
                    continue
                order = np.argsort(-plan)
                first = int(order[0])
                if plan[first] <= 0 or plan[first] < threshold[i] or art.cost[i,first] > limit:
                    continue
                selected = [first]
                row_sum = float(plan.sum())
                if gap == 1 and cfg.division_enabled and cfg.max_children > 1:
                    significant = [int(j) for j in order if plan[j] >= threshold[i]
                        and plan[j] > 0 and plan[j]/max(row_sum,1e-300) >= cfg.div_ratio
                        and art.cost[i,j] <= limit]
                    if len(significant) >= 2:
                        children = significant[:cfg.max_children]
                        if not cfg.volume_conservation or volume_consistent(dets.volume(t)[i],
                                dets.volume(t+1)[children].tolist(),cfg.vol_tol):
                            selected = children
                for j in selected:
                    rows.append((i,j)); scores.append(float(plan[j]/max(row_sum,1e-300))); gaps.append(gap)
        decisions[t] = {"pairs":np.asarray(rows,dtype=np.int64).reshape(-1,2),
                        "score":np.asarray(scores),"gap":np.asarray(gaps,dtype=np.int64)}
    result = reconstruct_from_edges(dets,decisions,replace(cfg,tau_edge=0.0),couplings,jump)
    result.meta.update(decision="ot_rule",n_link_accept=result.meta['n_edges_accepted']-result.meta['n_bridge_used'],
                       n_bridge_accept=result.meta['n_bridge_used'])
    return result


def reconstruct_from_edges(dets,decisions,cfg,couplings=None,jump=None):
    """式(33) 边概率→有限父子谱系，同时登记被冲突筛选的区域。"""
    ts = dets.t_range
    pair_of = {t:(couplings or {}).get(pos) for pos,t in enumerate(ts[:-1])}
    flags = {t:mass_flags(art,cfg) for t,art in pair_of.items() if art is not None}
    stats = {"n_division":0,"n_division_rejected_volume":0,"n_bridge_used":0,
             "n_bridge_dropped":0,"n_edges_accepted":0,"n_death":sum(int(x.sum()) for x,_ in flags.values()),
             "n_birth":sum(int(x.sum()) for _,x in flags.values()),"n_mass_veto":0}
    edges = []
    indegree, outdegree = {}, {}
    for t,dec in decisions.items():
        if not dec or dec.get('pairs') is None:
            continue
        pairs = np.asarray(dec['pairs'],dtype=np.int64).reshape(-1,2)
        gaps = np.asarray(dec.get('gap',np.ones(len(pairs))),dtype=int)
        for (i,j),score,gap in zip(pairs,np.asarray(dec['score']),gaps):
            if not np.isfinite(score) or score < cfg.tau_edge:
                continue
            t,i,j,gap = int(t),int(i),int(j),int(gap)
            target = t+gap
            if target not in dets.frames or not (0 <= i < dets.n(t) and 0 <= j < dets.n(target)):
                raise ValueError('决策边引用不存在的检测')
            art = pair_of.get(t) if gap == 1 else (jump or {}).get(t)
            if art is not None:
                death,birth = mass_flags(art,cfg)
                if (cfg.death_veto and death[i]) or (cfg.birth_veto and birth[j]):
                    stats['n_mass_veto'] += 1
                    continue
            edge = (t,i,target,j,float(score))
            edges.append(edge)
            indegree[target,j] = indegree.get((target,j),0)+1
            outdegree[t,i] = outdegree.get((t,i),0)+1
    # 先分配直接边，目标被占用时其他源仍可尝试其余候选，避免提前丢弃备选。
    incoming, outgoing = {}, {}
    kept = []
    for edge in sorted(edges,key=lambda e:(e[2]-e[0],-e[4],e[0],e[1],e[3])):
        t,i,tn,j,score = edge
        source,target = (t,i),(tn,j)
        if target in incoming:
            continue
        existing = outgoing.get(source,[])
        if existing:
            if tn-t > 1 or existing[0][2]-t > 1:
                stats['n_bridge_dropped'] += int(tn-t>1)
                continue
            if not cfg.division_enabled or len(existing) >= cfg.max_children:
                continue
            art = pair_of.get(t)
            if art is not None:
                ids = [e[3] for e in existing]+[j]
                row = max(float(art.row_sum[i]),1e-300)
                if any(art.plan[i,k]/row < cfg.div_ratio for k in ids):
                    continue
            sizes = [float(dets.volume(e[2])[e[3]]) for e in existing]+[float(dets.volume(tn)[j])]
            if cfg.volume_conservation and not volume_consistent(float(dets.volume(t)[i]),sizes,cfg.vol_tol):
                stats['n_division_rejected_volume'] += 1
                continue
        outgoing.setdefault(source,[]).append(edge)
        incoming[target] = edge
        kept.append(edge)
    stats['n_division'] = sum(len(items)>1 for items in outgoing.values())
    stats['n_edges_accepted'] = len(kept)
    stats['n_bridge_used'] = sum(e[2]-e[0]>1 for e in kept)
    if cfg.mark_uncertainty:
        uncertain = {node for node,degree in indegree.items() if degree>1}
        uncertain.update(node for node,degree in outdegree.items() if degree>cfg.max_children)
        stats['uncertain_nodes'] = [{"frame":t,"det_index":i,"incoming_candidates":indegree.get((t,i),0),
                                   "outgoing_candidates":outdegree.get((t,i),0)} for t,i in sorted(uncertain)]
        stats['n_uncertain_nodes'] = len(uncertain)
    active = set(incoming)|set(outgoing)
    assignment = {t:np.zeros(dets.n(t),dtype=np.int64) for t in ts}
    tracks, next_id = {}, 1
    suppressed = 0
    for t in ts:
        for i in range(dets.n(t)):
            if cfg.filter_isolated and (t,i) not in active:
                suppressed += 1
                continue
            if assignment[t][i] == 0:
                assignment[t][i] = next_id
                tracks[next_id] = Track(next_id,t,t,0)
                next_id += 1
            source_id = int(assignment[t][i])
            kids = outgoing.get((t,i),[])
            for _,_,tn,j,_ in kids:
                if len(kids)>1:
                    assignment[tn][j] = next_id
                    tracks[next_id] = Track(next_id,tn,tn,source_id)
                    next_id += 1
                else:
                    assignment[tn][j] = source_id
    stats['n_isolated_suppressed'] = suppressed
    if all('gt_label' in table for table in dets.frames.values()):
        stats['isolated_suppression_diagnostic'] = {
            'mapped_gt':sum(int(np.sum((assignment[t]==0)&(dets.gt_label(t)>0))) for t in ts),
            'unmatched':sum(int(np.sum((assignment[t]==0)&(dets.gt_label(t)==0))) for t in ts)}
    assignment,tracks,normalization = normalize_tracks(assignment,tracks,cfg.hole_policy)
    stats.update(normalization)
    stats['accepted_edges'] = [list(e) for e in kept]
    return TrackResult(assignment=assignment,tracks=tracks,meta={"decision":"gnn_edge",**stats})
