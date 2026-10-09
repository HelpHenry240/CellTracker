"""以完整上游计算产生的 Γ/C 标定阈值，并报告不可逆候选损失。"""
from __future__ import annotations
from dataclasses import replace
import numpy as np
from ..graph.build import _gt_ids,_edge_label,_candidates
from ..representation.measure import pairwise_distance


def distribution(values):
    values=np.asarray(values,dtype=float)
    values=values[np.isfinite(values)]
    if not len(values):
        return {'n':0}
    return {'n':len(values),'min':float(values.min()),'max':float(values.max()),
            **{f'p{q}':float(np.percentile(values,q)) for q in [0.1,1,5,50,95,99,99.9]}}


def candidate_funnel(run,parent):
    """统计映射后的所有真实前驱边；欠分割多身份不会被降为单一 GT 标签。

    分母是预测实例空间内可定义的真实边，不是全体 GT 边。因此必须同时报告
    前端实例覆盖情况，不能把高候选召回误写为端到端召回。
    """
    counts={'true_detection_pairs':0,'within_physical_gate':0,'after_candidates':0,
            'all_candidate_pairs':0,'division_pairs':0,'division_candidates':0}
    gamma,cost,displacement,live_rows,dead_rows,live_cols,unmatched_cols=[],[],[],[],[],[],[]
    ts=run.dets.t_range
    for pos,t in enumerate(ts[:-1]):
        target=ts[pos+1]
        art=run.couplings[pos]
        pairs,_,_=_candidates(art,run.config.graph,run.config.coupling.r_max)
        selected={tuple(p) for p in pairs.tolist()}
        counts['all_candidate_pairs']+=len(pairs)
        positives=np.zeros(art.plan.shape,dtype=bool)
        for i in range(run.dets.n(t)):
            source_ids=_gt_ids(run.dets,t,i)
            for j in range(run.dets.n(target)):
                truth,division=_edge_label(source_ids,_gt_ids(run.dets,target,j),parent,1)
                if not truth:
                    continue
                positives[i,j]=True
                counts['true_detection_pairs']+=1
                counts['within_physical_gate']+=int(np.isfinite(art.cost[i,j]))
                counts['after_candidates']+=int((i,j) in selected)
                counts['division_pairs']+=division
                counts['division_candidates']+=division*int((i,j) in selected)
                gamma.append(art.plan[i,j]); cost.append(art.cost[i,j]); displacement.append(art.d_cur[i,j])
            if source_ids:
                (live_rows if positives[i].any() else dead_rows).append(art.row_sum[i])
        for j in range(run.dets.n(target)):
            (live_cols if positives[:,j].any() else unmatched_cols).append(art.col_sum[j])
    counts['candidate_true_recall']=counts['after_candidates']/max(counts['true_detection_pairs'],1)
    counts['division_pair_recall']=counts['division_candidates']/max(counts['division_pairs'],1)
    return {'counts':counts,'distributions':{'true_gamma':distribution(gamma),'true_cost':distribution(cost),
        'true_displacement_um':distribution(displacement),'live_out_mass':distribution(live_rows),
        'no_successor_out_mass':distribution(dead_rows),'live_in_mass':distribution(live_cols),
        'no_predecessor_in_mass':distribution(unmatched_cols)}}


def calibrate_run(run,parent):
    """建议值以真实边低分位保守标定，并重新量化候选损失；不以本地代理性能选参。"""
    report=candidate_funnel(run,parent)
    distributions=report['distributions']
    gamma=distributions['true_gamma'].get('p0.1',0.)
    out=distributions['live_out_mass'].get('p1',0.)
    incoming=distributions['live_in_mass'].get('p1',0.)
    from copy import deepcopy
    cfg=deepcopy(run.config)
    cfg.graph.theta_gamma=gamma
    cfg.graph.theta_gamma_frac=None
    feature_limit=distributions['true_cost'].get('p99.9')
    if feature_limit is not None:
        cfg.graph.theta_c=feature_limit
        cfg.reconstruct.theta_c=feature_limit
    cfg.reconstruct.theta_gamma=gamma
    cfg.reconstruct.theta_gamma_frac=None
    cfg.reconstruct.eta_death=out
    cfg.reconstruct.eta_birth=incoming
    cfg.reconstruct.mass_threshold_mode='absolute'
    calibrated=replace(run,config=cfg)
    report['suggestions']={'theta_gamma':gamma,'theta_c':feature_limit,'eta_death':out,'eta_birth':incoming,
                          'r_max_required_p999_um':distributions['true_displacement_um'].get('p99.9')}
    report['after_calibration']=candidate_funnel(calibrated,parent)['counts']
    report['protocol']='真实边 p0.1 传输质量、p99.9 完整特征代价、生死 p1；部署值需双序列官方验证'
    report['physical_volume_um3']=distribution(np.concatenate([run.dets.volume(t)*np.prod(run.spacing) for t in run.dets.t_range]))
    report['cost_C']=distribution(np.concatenate([c.cost[np.isfinite(c.cost)] for c in run.couplings.values()]))
    report['source_sha256']=run.info['input_sha256']
    report['input_contract']=run.info['input_contract']
    if report['after_calibration']['candidate_true_recall']<.99:
        report['candidate_warning']='真实边覆盖不足 99%；须扫描 R_max/epsilon/保底并记录分裂损失，不能直接冻结此配置'
    return cfg,report
