"""按 ideas.pdf 顺序编排检测→OT→多尺度→动态图/GNN→重建→tracklet。

体数据由检测与导出阶段逐帧读取；运行阶段仅保留实例表和小型耦合矩阵。
每个阶段记录耗时、输入契约和诊断计数。耦合缓存带输入文件及配置指纹，
中断后可以复用已经完成的帧对，不能用其他配置的结果替代当前计算。
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
import numpy as np

from celltracker.track.base import Detections, TrackResult
from ..config import PipelineConfig, validate_config
from ..coupling.pairwise import PairCoupling, solve_coupling, solve_jump_coupling, load_coupling
from ..graph.build import build_dataset, load_encoder_features
from ..longrange.motion import attach_velocity, estimate_velocity, velocity_report
from ..longrange.tracklet import link_tracklets
from ..reconstruction.rules import reconstruct_from_edges, reconstruct_from_ot
from ..reconstruction.tracks import holes_of
from ..representation.measure import resolve_spacing
from ..temporal.multiscale import refine_couplings
from .contracts import file_hash, pipeline_contract


def load_detections(h5_path):
    """读取实例表、完整 GT 映射、外观统计与物理间距，不载入三维掩码。"""
    return Detections.from_h5(h5_path)


def load_gt_parent(gt_h5):
    """读取谱系。缺失轨迹表是监督输入错误；合法无分裂序列可返回空字典。"""
    import h5py
    with h5py.File(gt_h5, 'r') as f:
        if 'tracks' not in f:
            raise ValueError(f'GT 缺少 tracks 表：{gt_h5}')
        tracks = np.asarray(f['tracks'])
    return {int(l):int(p) for l,p in zip(tracks['label'], tracks['parent'])}


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


def run_pipeline(h5_path, cfg, gt_h5=None, ckpt=None, frames=None, artifacts_dir=None,
                 dump_graphs=None, device='cpu', raw_image=None, verbose=True,
                 cache_dir=None,resume_graphs=False):
    """执行完整实例追踪。无权重时走式(23)/(24)，可用于建图和接口冒烟。

    GNN 推理必须使用相同检测来源和上游配置训练的权重。主配置中的 encoder
    特征不能静默回退为强度统计；强度与无外观特征分别是显式配置对照。
    """
    validate_config(cfg)
    started = last_stage = time.monotonic()
    stage_seconds = {}
    def say(message, stage):
        nonlocal last_stage
        now = time.monotonic()
        stage_seconds[stage] = round(now-last_stage, 3)
        last_stage = now
        if verbose:
            print(f'[{stage}] {message} ({stage_seconds[stage]:.1f}s)', flush=True)
    if ckpt and not cfg.gnn.enabled:
        raise ValueError('GNN 已关闭，不能同时传入权重')
    h5_path = Path(h5_path)
    dets = load_detections(h5_path)
    if frames is not None:
        keep = set(frames)
        dets = Detections({t:d for t,d in dets.frames.items() if t in keep}, meta=dets.meta)
    ts = dets.t_range
    if len(ts) < 2 or any(b != a+1 for a,b in zip(ts[:-1],ts[1:])):
        raise ValueError('至少提供两帧连续输入；漏检帧应保留为空实例表')
    spacing = resolve_spacing(cfg.measure.spacing_zyx,h5_path,raw_image)
    if spacing is None or len(spacing) != 3 or not np.isfinite(spacing).all() or min(spacing) <= 0:
        raise ValueError('无法确定合法物理间距，请提供 measure.spacing_zyx 或 H5 元数据')
    actual_source = str(dets.meta.get('detection_source',cfg.detection_source))
    if actual_source != cfg.detection_source:
        raise ValueError(f'检测来源不一致：H5={actual_source}，config={cfg.detection_source}')
    dets.meta['spacing_zyx'] = spacing
    encoder_feats, encoder_meta = _appearance(cfg,dets,spacing)
    contract = pipeline_contract(cfg,dets.meta,encoder_meta)
    info = {'schema':cfg.schema_version, 'n_frames':len(ts), 'frames':ts,
            'decision':'gnn' if ckpt else 'ot_rule', 'gnn_enabled':cfg.gnn.enabled,
            'detection_source':actual_source, 'spacing_zyx':list(spacing),
            'h5':str(h5_path), 'gt_h5':str(gt_h5) if gt_h5 else None,
            'input_sha256':file_hash(h5_path), 'input_contract':contract,
            'mass_mode':cfg.measure.mass_mode, 'f_source':cfg.node.f_source,
            'encoder_metadata':encoder_meta}
    if dump_graphs is not None:
        if gt_h5 is None:
            raise ValueError('训练图需要显式 --gt-h5，不能把预测标签当作谱系真值')
        if cfg.detection_source == 'nnunet_pred' and any('gt_ids' not in d for d in dets.frames.values()):
            raise ValueError('预测检测监督需要 gt_ids 完整映射；请先运行 relabel_detections.py')
    cache = _prepare_cache(cache_dir,info,cfg) if cache_dir else None
    say(f'{len(ts)} 帧；检测={actual_source}；外观={cfg.node.f_source}', 'load')
    ccfg = replace(cfg.coupling,alpha_pred=cfg.motion.alpha_pred if cfg.motion.enabled else 0.0)
    pred_xy = {}
    if cfg.motion.enabled and cfg.motion.alpha_pred > 0:
        coarse = _solve_all(dets,replace(ccfg,alpha_pred=0.0),cfg.measure,spacing,cache=cache,phase='coarse')
        coarse_result = reconstruct_from_ot(dets,coarse,cfg.reconstruct,ccfg.r_max,bridge=False)
        # v2 仅保留历史复现；v3 中被排除的检测不能借背景编号互相匹配。
        legacy_background = cfg.schema_version == 'ideas-v2'
        vel,valid = estimate_velocity(dets,coarse_result,ignore_background=not legacy_background)
        attach_velocity(dets,vel,valid)
        pred_xy = {t:dets.centroid(t)+vel[t] for t in ts}
        info['motion'] = {'enabled':True,'alpha_pred':cfg.motion.alpha_pred,
                          'legacy_background_matching':legacy_background,
                          'pass1_tracks':coarse_result.n_tracks(),'velocity':velocity_report(dets,spacing)}
    else:
        info['motion'] = {'enabled':False}
    say('运动先验初始化完成', 'motion')
    couplings = _solve_all(dets,ccfg,cfg.measure,spacing,pred_xy,cache,'adjacent')
    info['n_couplings'] = len(couplings)
    info['coupling_eps'] = [float(c.eps_eff) for c in couplings.values()]
    say(f'{len(couplings)} 对相邻帧耦合', 'couplings')
    xy,vol = {t:dets.centroid(t) for t in ts},{t:dets.volume(t) for t in ts}
    ms = refine_couplings(couplings,ts,xy,vol,cfg.multiscale,ccfg,cfg.measure,spacing,pred_xy)
    couplings = ms.couplings
    info['multiscale'] = ms.info
    jump = {}
    gap = cfg.graph.bridge_gap
    if cfg.graph.bridge:
        for pos,t in enumerate(ts):
            if pos+gap >= len(ts):
                continue
            jump[t] = ms.direct.get((pos,gap))
            if jump[t] is None:
                path = cache/f'jump_{t:04d}.npz' if cache else None
                if path and path.exists():
                    jump[t] = load_coupling(path)
                else:
                    jump[t] = solve_jump_coupling(xy[t],xy[ts[pos+gap]],gap,ccfg,cfg.measure,
                                                  vol[t],vol[ts[pos+gap]],spacing,eps=couplings[pos].eps_eff)
                    if path:
                        jump[t].save(path)
    info['n_jump_couplings'] = len(jump)
    say(f'多尺度={cfg.multiscale.enabled}；桥接耦合={len(jump)}', 'multiscale')
    if ckpt:
        from ..gnn.infer import make_edge_decider
        decider = make_edge_decider(ckpt,cfg,device,cfg.measure,spacing,input_contract=contract)
        decisions = decider(dets,couplings,jump,encoder_feats,ccfg.r_max)
        result = reconstruct_from_edges(dets,decisions,cfg.reconstruct,couplings,jump)
        info['checkpoint_sha256'] = file_hash(ckpt)
    else:
        result = reconstruct_from_ot(dets,couplings,cfg.reconstruct,ccfg.r_max,jump,
                                     cfg.graph.bridge,bridge_gap=gap)
    info['tracks_after_decision'] = result.n_tracks()
    say(f'{result.n_tracks()} 条轨迹', 'decision')
    tl = link_tracklets(dets,result,cfg.tracklet,ccfg,cfg.measure,spacing)
    result = tl.result
    info['tracklet'] = tl.info
    info['tracks_final'] = result.n_tracks()
    say(f'二层 OT={cfg.tracklet.enabled}；最终 {result.n_tracks()} 条轨迹', 'tracklet')
    holes = holes_of(result.assignment)
    info['holes'] = {'n_tracks_with_holes':len(holes),'n_hole_frames':sum(map(len,holes.values())),
                     'hole_policy':cfg.reconstruct.hole_policy}
    info['reconstruct'] = dict(result.meta)
    if dump_graphs is not None:
        parent = load_gt_parent(gt_h5)
        graph_dir=Path(dump_graphs)
        graph_dir.mkdir(parents=True,exist_ok=True)
        graph_identity={'input_contract':contract,'detection_sha256':info['input_sha256'],
                        'gt_sha256':file_hash(gt_h5),'frames':ts}
        graph_identity=json.loads(json.dumps(graph_identity))
        identity_path=graph_dir/'graph_identity.json'
        if identity_path.exists():
            if json.loads(identity_path.read_text())!=graph_identity:
                raise ValueError('恢复图的数据来源或配置不匹配')
        elif any(graph_dir.glob('pair_*.npz')):
            raise ValueError('已有图缺少来源契约，不能恢复')
        else:
            identity_path.write_text(json.dumps(graph_identity,ensure_ascii=False,indent=2))
        stats = build_dataset(dets,couplings,jump,cfg.graph,cfg.measure,dump_graphs,
                              parent,encoder_feats,spacing,ccfg.r_max,resume=resume_graphs)
        info['graph_dataset'] = stats
        manifest = {'schema':cfg.schema_version,'input_contract':contract,
                    'detection_sha256':info['input_sha256'],'gt_sha256':file_hash(gt_h5),
                    'frames':ts,'stats':stats}
        (Path(dump_graphs)/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
        say(f'{stats["n_graphs"]} 张监督图', 'dump_graphs')
    info.update(stage_seconds=stage_seconds,total_seconds=round(time.monotonic()-started,3))
    art_dir = Path(artifacts_dir) if artifacts_dir else None
    if art_dir:
        art_dir.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(art_dir/'couplings.npz',**{f'plan_{i}':c.plan for i,c in couplings.items()})
        np.savez_compressed(art_dir/'assignment.npz',**{f'frame_{t}':v for t,v in result.assignment.items()})
        (art_dir/'run_info.json').write_text(json.dumps(info,ensure_ascii=False,indent=2,default=str))
    return PipelineRun(cfg,dets,couplings,jump,result,spacing,holes,info,art_dir)


def _appearance(cfg,dets,spacing):
    if cfg.node.f_source == 'none':
        return {t:np.empty((dets.n(t),0),dtype=np.float32) for t in dets.t_range},{}
    if cfg.node.f_source == 'intensity':
        return None,{}
    if not cfg.node.encoder_feat_path:
        raise ValueError('encoder_npz 需要 node.encoder_feat_path，不能回退为强度统计')
    with np.load(cfg.node.encoder_feat_path,allow_pickle=False) as stored:
        if 'metadata_json' not in stored:
            raise ValueError('encoder 侧车缺少模型与预处理元数据')
        meta = json.loads(str(stored['metadata_json'].item()))
    return load_encoder_features(cfg.node.encoder_feat_path,dets,spacing),meta


def _prepare_cache(root,info,cfg):
    root = Path(root)
    root.mkdir(parents=True,exist_ok=True)
    identity = {'schema':cfg.schema_version,'input_sha256':info['input_sha256'],
                'frames':info['frames'],'spacing':info['spacing_zyx'],
                'measure':asdict(cfg.measure),'coupling':asdict(cfg.coupling),
                'motion':asdict(cfg.motion),'reconstruct':asdict(cfg.reconstruct),
                'multiscale':asdict(cfg.multiscale),'bridge_gap':cfg.graph.bridge_gap}
    if cfg.coupling.enabled and (cfg.coupling.tau_a is None or cfg.coupling.tau_b is None):
        # 新硬边际校验不能被历史截断迭代缓存绕过；软边际算例的数值路径保持相同。
        identity['hard_constraint_validation'] = 'sinkhorn_final_marginals_v1'
    # JSON 往返统一 tuple/list 以及数值字典键。
    identity = json.loads(json.dumps(identity))
    manifest = root/'cache_manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text()) != identity:
            raise ValueError('耦合缓存输入/配置不匹配，请使用独立目录')
    elif list(root.glob('*.npz')):
        raise ValueError('缓存缺少来源契约，不能复用')
    else:
        manifest.write_text(json.dumps(identity,ensure_ascii=False,indent=2))
    return root


def _solve_all(dets,ccfg,mcfg,spacing,pred_xy=None,cache=None,phase='adjacent'):
    output = {}
    for pos,(t,tn) in enumerate(zip(dets.t_range[:-1],dets.t_range[1:])):
        path = cache/f'{phase}_{t:04d}.npz' if cache else None
        if path and path.exists():
            output[pos] = load_coupling(path)
        else:
            output[pos] = solve_coupling(dets.centroid(t),dets.centroid(tn),dets.volume(t),
                dets.volume(tn),None if pred_xy is None else pred_xy.get(t),ccfg,mcfg,spacing)
            if path:
                output[pos].save(path)
    return output
