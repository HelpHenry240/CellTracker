"""模块消融注册表：按基准的实际状态反向切换，所有变体使用独立训练目录。

开关用于隔离机制，不代表预先确定的优化方向。复合的 ot 消融会一起关闭
依赖运输质量的消费者，避免名义关闭 OT 而重建/损失仍然使用 OT 的情况。
"""
from __future__ import annotations
from dataclasses import dataclass
from ..config import override


@dataclass(frozen=True)
class Module:
    title: str
    field: str
    on: object = True
    off: object = False


MODULES = {
    'fgw': Module('式(9) FGW 结构项','coupling.eta',0.3,0.0),
    'motion': Module('式(20)–(22) 运动先验','motion.enabled'),
    'multiscale': Module('式(17)–(19) 多尺度时间正则','multiscale.enabled'),
    'tracklet': Module('§1.5 二层 tracklet OT','tracklet.enabled'),
    'ot_candidates': Module('式(26) 质量与代价候选筛选','graph.use_ot_candidates'),
    'topk': Module('论文之外的候选 top-k 保底','graph.cand_topk',3,0),
    'ot_features': Module('式(27) OT 代价与质量特征','graph.use_ot_features'),
    'intra': Module('式(28) 帧内邻域边','graph.intra_enabled'),
    'similarity': Module('式(6) 帧内高斯相似度','graph.use_intra_similarity'),
    'context': Module('§2.0.1 多帧消息传递','graph.context_enabled'),
    'bridge': Module('§2.0.1 漏检跨帧桥接','graph.bridge'),
    'gnn': Module('式(30)–(35) 学习型边判定','gnn.enabled'),
    'ot_loss': Module('式(35) OT 一致性损失','gnn.lambda_ot',0.2,0.0),
    'division': Module('§1.6 分裂识别','reconstruct.division_enabled'),
    'volume': Module('§1.6 分裂体积守恒','reconstruct.volume_conservation'),
    'birth_death': Module('§1.6 出生与终止质量判据','reconstruct.birth_death_enabled'),
    'isolated': Module('§2.0.1 孤立节点过滤','reconstruct.filter_isolated'),
    'uncertainty': Module('§2.0.1 欠分割不确定区域记录','reconstruct.mark_uncertainty'),
    'line_search': Module('完整目标回溯线搜索','multiscale.line_search'),
    'size_cost': Module('式(8) 尺寸差代价','coupling.beta',1.0,0.0),
}


def _get(cfg, field):
    section,key = field.split('.')
    return getattr(getattr(cfg,section),key)


def variant(cfg,name):
    """返回独立配置及可审计的变更说明，基准配置保持不变。"""
    if name == 'ot':
        enabled = cfg.coupling.enabled
        if not enabled:
            raise ValueError('OT 已关闭；开启整个 OT 链需要显式基准配置')
        changes = ['coupling.enabled=false','graph.use_ot_candidates=false','graph.use_ot_features=false',
                   'motion.enabled=false','multiscale.enabled=false','tracklet.enabled=false',
                   'reconstruct.birth_death_enabled=false','gnn.lambda_ot=0']
        output = override(cfg,changes)
    elif name == 'unbalanced':
        enabled = cfg.coupling.tau_a is not None or cfg.coupling.tau_b is not None
        value = 'null' if enabled else '1.0'
        changes = [f'coupling.tau_a={value}',f'coupling.tau_b={value}']
        output = override(cfg,changes)
    elif name == 'appearance':
        enabled = cfg.node.f_source == 'encoder_npz'
        if enabled:
            changes = ['node.f_source=intensity','node.encoder_feat_path=null']
        elif cfg.node.encoder_feat_path:
            changes = ['node.f_source=encoder_npz']
        else:
            raise ValueError('开启 encoder 外观需要显式 encoder_feat_path')
        output = override(cfg,changes)
    elif name == 'mass':
        enabled = cfg.measure.mass_mode == 'volume'
        changes = [f'measure.mass_mode={"uniform" if enabled else "volume"}']
        output = override(cfg,changes)
    elif name == 'structure':
        enabled = cfg.measure.structure_mode == 'full'
        changes = [f'measure.structure_mode={"knn" if enabled else "full"}']
        output = override(cfg,changes)
    else:
        if name not in MODULES:
            raise KeyError(f'未知模块 {name}；可用：{", ".join(module_names())}')
        module = MODULES[name]
        current = _get(cfg,module.field)
        enabled = bool(current)
        value = module.off if enabled else module.on
        import yaml
        literal = yaml.safe_dump(value,default_flow_style=True).split('\n')[0]
        changes = [f'{module.field}={literal}']
        output = override(cfg,changes)
    return output, {'module':name,'direction':'off' if enabled else 'on',
                    'changes':changes,'independent_training':output.gnn.enabled}


def module_names():
    return ['ot','unbalanced','appearance','mass','structure',*MODULES]


def describe(cfg):
    report = []
    for name in module_names():
        try:
            _,entry = variant(cfg,name)
            report.append(entry)
        except ValueError as error:
            report.append({'module':name,'requires_configuration':str(error)})
    return report
