"""ideas.pdf 的模块配置、序列化与严格字段检查。

物理位置/速度使用 µm 和 µm/帧，体积使用 µm³。None 类型的参数由输入元数据
或显式标定文件提供。每个独立机制都有开关；工程变体在字段说明中注明。
"""
from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass
class MeasureConfig:
    """式(1)–(7)：a=s/Σs，f=[x,s]，空间 kNN 与高斯相似度。"""
    mass_mode: str = "volume"
    knn_k: int = 6
    sigma_x: float | None = None
    sigma_f: float | None = None
    spacing_zyx: tuple[float, float, float] | None = None
    structure_mode: str = "full"       # full=式(7)；knn=§1.7 的局部截断近似


@dataclass
class CouplingConfig:
    """式(8)–(14)：特征/FGW 代价、熵正则和可选 KL 边际松弛。"""
    enabled: bool = True
    alpha: float = 1.0
    alpha_pred: float = 0.0
    beta: float = 1.0
    sigma_s: float | None = None
    r_max: float = 3.0
    eta: float = 0.3
    eps_rel: float | None = 0.1        # 工程标定：ε=比例×正代价中位数
    eps: float | None = None
    tau_a: float | None = 1.0         # None=硬边际约束
    tau_b: float | None = 1.0
    sinkhorn_iters: int = 1000
    fgw_outer: int = 10


@dataclass
class MotionConfig:
    """式(20)–(22)：v=x_t−x_prev，预测 x_next=x_t+v。"""
    enabled: bool = True
    alpha_pred: float = 1.0


@dataclass
class MultiscaleConfig:
    """式(17)–(19)：Σ单对完整目标+Σ_k λ_k‖Γ_direct−ΠΓ‖²。"""
    enabled: bool = True
    ks: tuple[int, ...] = (2, 3, 5)
    lambda_temp: dict[int, float] = field(default_factory=lambda: {2: 0.1, 3: 0.05, 5: 0.02})
    n_rounds: int = 2
    norm: str = "raw"                # cond 是显式工程对照，不是式(18)/(19) 的字面形式
    line_search: bool = True
    jump_r_max_scale: float = 1.0


@dataclass
class TrackletConfig:
    """§1.5：滑动短窗口内的高置信链→汇总特征→二层 OT。"""
    enabled: bool = True
    window: int = 5
    theta_link: float = 0.5           # 二层计划占源质量的接受比例（工程离散化）
    local_confidence: float = 0.8     # 第一层边置信度；需严格于常规接受阈值
    tau: float = 0.5
    max_gap: int = 3
    velocity_weight: float = 1.0
    size_weight: float = 1.0
    merge_division_children: bool = False
    hole_policy: str = "fill"


@dataclass
class GraphConfig:
    """式(25)–(29)：物理特征、双阈值候选、帧内边和真实前驱监督。"""
    theta_gamma_frac: float | None = None  # 按源质量缩放的工程对照
    theta_gamma: float | None = 0.0       # 绝对质量阈值，部署配置由标定产生
    theta_c: float | None = None
    cand_topk: int = 0                    # 论文外候选保底，0 完全关闭
    ctx_window: int = 2
    bridge: bool = True
    bridge_gap: int = 2
    bridge_scope: str = "missing_only"
    intra_knn: int = 4
    intra_enabled: bool = True
    context_enabled: bool = True
    use_ot_candidates: bool = True
    use_ot_features: bool = True
    use_intra_similarity: bool = True


@dataclass
class ReconstructConfig:
    """式(23)/(24)、§1.6 生死/分裂与 §2.0.1 有限父子约束。"""
    theta_gamma_frac: float | None = None
    theta_gamma: float | None = 0.0
    theta_c: float | None = None
    eta_death: float = 0.0               # 绝对流出质量下限，需标定
    eta_birth: float = 0.0               # 绝对流入质量下限，需标定
    mass_threshold_mode: str = "absolute"  # relative 为工程对照
    birth_death_enabled: bool = True
    death_veto: bool = True
    birth_veto: bool = True
    division_enabled: bool = True
    volume_conservation: bool = True
    div_ratio: float = 0.05
    vol_tol: float = 0.5
    max_children: int = 2
    tau_edge: float = 0.5
    hole_policy: str = "fill"            # CTC 适配：fill 补画 / split 拆段
    filter_isolated: bool = False        # §2.0.1；主配置开启，保留全检测对照
    mark_uncertainty: bool = True


@dataclass
class NodeFeatureConfig:
    """式(25)：实例的分割网络外观特征；intensity 为工程近似。"""
    f_source: str = "intensity"
    encoder_feat_path: str | None = None


@dataclass
class GNNConfig:
    """式(30)–(35)：边/节点交替更新、二分类和可选 OT 损失。"""
    enabled: bool = True
    hidden: int = 64
    layers: int = 3
    dropout: float = 0.1
    lambda_ot: float = 0.2
    class_weighted_ce: bool = False
    residual: bool = False


@dataclass
class PipelineConfig:
    measure: MeasureConfig = field(default_factory=MeasureConfig)
    coupling: CouplingConfig = field(default_factory=CouplingConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    multiscale: MultiscaleConfig = field(default_factory=MultiscaleConfig)
    tracklet: TrackletConfig = field(default_factory=TrackletConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    reconstruct: ReconstructConfig = field(default_factory=ReconstructConfig)
    node: NodeFeatureConfig = field(default_factory=NodeFeatureConfig)
    gnn: GNNConfig = field(default_factory=GNNConfig)
    seed: int = 20261008
    detection_source: str = "nnunet_pred"
    schema_version: str = "ideas-v2"


def _to_plain(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    return obj


def validate_config(cfg: PipelineConfig) -> None:
    if cfg.measure.mass_mode not in {"volume", "uniform"}:
        raise ValueError("measure.mass_mode 必须为 volume/uniform")
    if cfg.measure.structure_mode not in {"full", "knn"}:
        raise ValueError("measure.structure_mode 必须为 full/knn")
    if cfg.multiscale.norm not in {"raw", "cond"}:
        raise ValueError("multiscale.norm 必须为 raw/cond")
    if cfg.node.f_source not in {"intensity", "encoder_npz", "none"}:
        raise ValueError("node.f_source 必须为 encoder_npz/intensity/none")
    if cfg.reconstruct.mass_threshold_mode not in {"absolute", "relative"}:
        raise ValueError("mass_threshold_mode 必须为 absolute/relative")
    if not 0 <= cfg.coupling.eta <= 1 or cfg.coupling.r_max <= 0:
        raise ValueError("eta 须在 [0,1]，r_max 须为正")
    if cfg.graph.cand_topk < 0 or cfg.graph.ctx_window < 0:
        raise ValueError("候选数量和上下文长度不能为负")
    if cfg.graph.bridge_gap < 2 or any(int(k) < 2 for k in cfg.multiscale.ks):
        raise ValueError("桥接与多尺度时间跨度至少为 2")
    if not cfg.coupling.enabled and any((cfg.graph.use_ot_candidates,cfg.graph.use_ot_features,
            cfg.motion.enabled,cfg.multiscale.enabled,cfg.reconstruct.birth_death_enabled,
            cfg.gnn.lambda_ot > 0,cfg.tracklet.enabled)):
        raise ValueError('关闭 OT 时须关闭质量筛选/特征、生死、运动初始化、时间正则、OT 损失和二层 OT')
    if cfg.tracklet.window < 2 or cfg.tracklet.max_gap < 1:
        raise ValueError('tracklet 窗口至少为 2，最大跨度至少为 1')
    if cfg.reconstruct.max_children not in (1, 2):
        raise ValueError("当前 CTC 适配支持最多两个子轨迹")
    spacing = cfg.measure.spacing_zyx
    if spacing is not None and (len(spacing) != 3 or any(v <= 0 for v in spacing)):
        raise ValueError("spacing_zyx 必须为三个正物理间距")


def save_config(cfg: PipelineConfig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(_to_plain(asdict(cfg)), allow_unicode=True, sort_keys=False))
    return path


def load_config(path: str | Path | None = None) -> PipelineConfig:
    cfg = PipelineConfig()
    data = yaml.safe_load(Path(path).read_text()) or {} if path else {}
    known = {f.name for f in fields(cfg)}
    for name, values in data.items():
        if name not in known:
            raise KeyError(f"未知配置段 {name}")
        section = getattr(cfg, name)
        if hasattr(section, '__dataclass_fields__'):
            if not isinstance(values, dict):
                raise TypeError(f"配置段 {name} 必须是字典")
            for key, value in values.items():
                _set(section, key, value, name)
        else:
            setattr(cfg, name, values)
    validate_config(cfg)
    return cfg


def _set(section, key, value, name):
    if not hasattr(section, key):
        raise KeyError(f"未知配置项 {name}.{key}")
    if name == "multiscale" and key == "lambda_temp":
        value = {int(k): float(v) for k, v in value.items()}
    if isinstance(value, list) and isinstance(getattr(section, key), tuple):
        value = tuple(value)
    setattr(section, key, value)


def override(cfg: PipelineConfig, pairs: list[str]) -> PipelineConfig:
    cfg = copy.deepcopy(cfg)
    for pair in pairs:
        key, sep, val = pair.partition("=")
        if not sep:
            raise ValueError(f"覆盖格式应为 段.字段=值：{pair}")
        value = yaml.safe_load(val)
        section, dot, name = key.partition(".")
        if not hasattr(cfg, section):
            raise KeyError(key)
        if dot:
            _set(getattr(cfg, section), name, value, section)
        else:
            _set(cfg, section, value, "pipeline")
    validate_config(cfg)
    return cfg


def to_jsonable(cfg: PipelineConfig) -> str:
    return json.dumps(_to_plain(asdict(cfg)), ensure_ascii=False, indent=2)


def with_section(cfg: PipelineConfig, name: str, **kw) -> PipelineConfig:
    return replace(cfg, **{name: replace(getattr(cfg, name), **kw)})
