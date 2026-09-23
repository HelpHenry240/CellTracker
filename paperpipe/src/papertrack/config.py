"""严格论文口径的统一配置（每个字段都标注 ideas.pdf 出处）。

设计原则
--------
1. **一个论文符号一个字段**，字段名尽量与公式里的符号对应（α / β / σ_s / R_max /
   η / ε / τ / λ_temp / θ_Γ / θ_C / η_death / η_birth / K）。
2. **凡是论文没给数值的量，一律显式标注**：
   - `PAPER_PARAM` = 论文给出（含取值范围）；
   - `CALIBRATED` = 论文未给数值，本实现按物理量纲标定（R3），
     标定方法与经验分布写在字段注释里，可用 `scripts/calibrate_params.py` 复算；
   - `ENG_SUPP`   = 论文之外的工程补充（R2 要求单独标注）。
3. 距离/代价相关量一律 **µm**（R3）；`spacing_zyx` 决定体素↔物理换算。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "MeasureConfig", "CouplingConfig", "MotionConfig", "MultiscaleConfig",
    "TrackletConfig", "GraphConfig", "ReconstructConfig", "NodeFeatureConfig",
    "GNNConfig", "PipelineConfig", "load_config", "save_config", "override",
]


@dataclass
class MeasureConfig:
    """§1.2 帧内表示：经验测度（式1-3）与帧内邻域图（式4-7）。"""

    # 式(1)：a_i^t = s_i^t / Σ_k s_k^t —— 原文"质量与尺寸成正比"。
    # 注意：CTC 的 `_GT/TRA` 标记点是等体积的（父子 marker 体积比恒 2.00，
    # 见 docs/decisions/0001），在**GT 标记设定**下"质量∝尺寸"会被解耦成 uniform；
    # 本 pipeline 面向 nnU-Net **预测实例**（体积真实可变），故默认按原文取 volume。
    mass_mode: str = "volume"          # PAPER_PARAM 式(1)："volume"（论文口径）| "uniform"
    knn_k: int = 6                     # CALIBRATED 式(5)：帧内 kNN 度数（论文未给 k）
    sigma_x: float | None = None       # CALIBRATED 式(6)：None = 正距离中位数（自适应）
    sigma_f: float | None = None       # CALIBRATED 式(6)：None = 正特征距离中位数
    # R3：任何距离必须按物理间距换算。None = 自动解析（h5 attrs → tif 头 → (1,1,1)）。
    spacing_zyx: tuple[float, float, float] | None = None


@dataclass
class CouplingConfig:
    """§1.3 相邻帧最优传输（式 8-14）。"""

    alpha: float = 1.0                 # PAPER_PARAM 式(8) α>0：位移代价
    # 式(22) 的运动先验权重 α′。语义上属于 §1.5，但按原文"在下文如无特殊说明，
    # C_feat 均可理解为包含这一运动先验的扩展版本"，它直接进入 C_feat，
    # 故这里保存**生效值**（由 `MotionConfig.alpha_pred` 在 pipeline 里覆盖）。
    alpha_pred: float = 0.0            # PAPER_PARAM 式(22) α′（0 = 单遍，不带先验）
    beta: float = 1.0                  # PAPER_PARAM 式(8) β>0：尺寸变化代价（原文要求 >0）
    sigma_s: float | None = None       # CALIBRATED 式(8)：None = 尺寸中位数（量纲标定）
    r_max: float = 3.0                 # CALIBRATED 式(8)：候选位移上限，单位 **µm**
    eta: float = 0.3                   # PAPER_PARAM 式(9) η∈[0,1]：结构项权重
    # ε：论文只写 ε>0（式12）。整数 ε 无意义时必须按代价尺度标定：
    # eps_rel=0.1 → ε = 0.1 × median(C)。这是 **ENG_SUPP**（原文没有自适应公式），
    # 依据是量纲：C 是平方距离，取 ε=1 会让 Sinkhorn 退化成硬分配。
    eps_rel: float | None = 0.1        # ENG_SUPP
    eps: float | None = None           # PAPER_PARAM 式(12)：显式给定时优先于 eps_rel
    # τ：论文说"τ 小 → 允许大规模出生/死亡/分裂"（式14）。
    # 取有限值即启用非平衡 OT；**η_death/η_birth（式24）只有在非平衡设定下才有意义**
    # （平衡 OT 的行/列和恒等于 a，阈值永不触发），故默认启用式(14)。
    tau_a: float | None = 1.0          # CALIBRATED 式(14) τ_t
    tau_b: float | None = 1.0          # CALIBRATED 式(14) τ_{t+1}
    sinkhorn_iters: int = 1000         # ENG_SUPP 数值迭代上限
    # 式(9) FGW 的条件梯度外层轮数上限：原文 §1.7 只说"在最坏情况下 O(n²m²)，
    # 实际实现通常结合稀疏邻接与低秩/随机近似降低复杂度"，没给迭代次数 → ENG_SUPP。
    fgw_outer: int = 10


@dataclass
class MotionConfig:
    """§1.5 运动先验（式 20-22）：两遍式，速度来自"上一轮追踪"的硬关联。"""

    alpha_pred: float = 1.0            # PAPER_PARAM 式(22) α′>0
    enabled: bool = True               # 关掉即退化为单遍（α′=0）


@dataclass
class MultiscaleConfig:
    """§1.4 + §1.5 多尺度时间耦合（式 17-19）。"""

    enabled: bool = True
    ks: tuple[int, ...] = (2, 3, 5)    # PAPER_PARAM 式(19)："K 如 {2,3,5}"
    # 式(19) 是**逐尺度**权重 λ^(k)_temp。论文未给数值 → 按"跳帧项是软正则、
    # 不得主导"（§1.7）取随 k 递减的默认值。
    lambda_temp: dict[int, float] = field(
        default_factory=lambda: {2: 0.1, 3: 0.05, 5: 0.02})   # CALIBRATED
    n_rounds: int = 2                  # CALIBRATED 交替优化轮数（§1.7 建议"少量迭代"）
    # 式(18)(19) 字面是 `Γ^t Γ^{t+1}` 的乘积；原文紧接着给出条件概率定义
    # P((j,t+1)|(i,t)) = Γ_ij / (Σ_j' Γ_ij' + δ)，并说明 Γ 诱导一个 Markov 链。
    # "cond" = 在该条件概率空间比较（默认，"raw" 会拿两个不同质量尺度的量相减）；
    # "raw"  = 字面乘积，供对照。
    norm: str = "cond"                 # PAPER_PARAM 原文提供两种口径
    line_search: bool = True           # ENG_SUPP 回溯线搜索（§1.7 只说"少量迭代微调"）
    jump_r_max_scale: float = 1.0      # 跳帧 OT 的 R_max 放宽系数（1.0 = 按 gap 线性放宽）


@dataclass
class TrackletConfig:
    """§1.5 末段 + §1.6：短时窗 tracklet → 超级节点 → 第二层 OT。"""

    enabled: bool = True
    window: int = 5                    # PAPER_PARAM 原文"较短滑动时间窗口（如 3–5 帧）"
    theta_link: float = 0.5            # CALIBRATED 原文"较严格的阈值"（占源质量的比）
    tau: float = 0.5                   # CALIBRATED 第二层用式(14) 的非平衡 OT
    max_gap: int = 3                   # CALIBRATED 允许串联的最大时间间隔
    velocity_weight: float = 1.0       # CALIBRATED 式(20-22) 在 tracklet 级的权重
    merge_division_children: bool = False  # 默认不把分裂子轨迹并回父（分裂边要保留）
    hole_policy: str = "fill"          # ENG_SUPP 见 reconstruction/exporter 说明


@dataclass
class GraphConfig:
    """§2.0.1 时间展开图（式 25-29）。"""

    # 式(26) 的 θ_Γ 是**原始传输质量** Γ_ij 的下限。Γ 的量级随质量归一化变化
    # （Σ_i a_i = 1 ⇒ Γ 的元素量级 ~1/(n·m)），故默认按"源细胞质量的占比"标定：
    #   θ_Γ(i) = theta_gamma_frac × a_i^t
    # 若显式给 `theta_gamma`（绝对阈值），则按原文的单一标量使用。
    theta_gamma_frac: float | None = 0.05   # CALIBRATED 式(26)
    theta_gamma: float | None = None        # PAPER_PARAM 式(26) 绝对值（优先）
    theta_c: float | None = None            # PAPER_PARAM 式(26) 代价上限；None → R_max²
    cand_topk: int = 0                 # ENG_SUPP 每行质量前 k 保底；**0 = 原文口径**
    ctx_window: int = 1                # §2.0.1"多步时间上下文"：并入前后各 n 帧做节点/帧内边
    bridge: bool = True                # §2.0.1 跨帧桥接边 (i,t−1)→(k,t+1)
    bridge_gap: int = 2                # PAPER_PARAM §2.0.1 原文即"跨一帧"（Δt=2）
    # 桥接边的适用条件：原文前置是"某一帧 t 的真实细胞**未被分割出来**，对应节点
    # (i,t) 在图中缺失"。因此默认只在"源细胞在 t+1 连一个 R_max 内的候选都没有"
    # 时生成桥接边；"all" = 对所有源都生成（对照口径，会显著增加空洞）。
    bridge_scope: str = "missing_only"
    intra_knn: int = 4                 # CALIBRATED 式(28) 帧内 kNN 度数
    use_ot_candidates: bool = True     # 消融：False = 纯 R_max 几何门控（非原文口径）


@dataclass
class ReconstructConfig:
    """§1.6 由传输计划重建轨迹（式 23-24 + 死亡/出生/分裂判据）。"""

    theta_gamma_frac: float | None = 0.05  # CALIBRATED 式(24)：同式(26) 的标定方式
    theta_gamma: float | None = None       # PAPER_PARAM 式(24) 绝对值（优先）
    theta_c: float | None = None           # PAPER_PARAM 式(24) 代价上限；None → R_max²
    eta_death: float = 0.5                 # CALIBRATED 式(24)：行和占 a_i 的比例下限
    eta_birth: float = 0.5                 # CALIBRATED 式(24)：列和占 a_j 的比例下限
    div_ratio: float = 0.5                 # CALIBRATED 原文"显著质量（超过一定比例阈值）"
    vol_tol: float = 0.5                   # CALIBRATED 原文"体积守恒（给定容差内）"
    # 论文原文把 η_death/η_birth 描述成"标记终止点/起点"的判据，字面上可以**否决**
    # 式(24) 已接受的关联。但一个标定不准的 η 会不可逆地删掉正确边（R5），
    # 故默认关闭（η 只做诊断计数）；打开即回到原文的字面读法。
    death_veto: bool = False
    max_children: int = 2                  # PAPER_PARAM §2.0.1"有限个子节点"
    # GNN 路径的接受阈值（原文只给了 θ_Γ/θ_C，没有给 ŷ 的阈值）→ CALIBRATED
    tau_edge: float = 0.5
    hole_policy: str = "fill"              # ENG_SUPP CTC 格式约束，见 exporter


@dataclass
class NodeFeatureConfig:
    """式(25) 节点初始特征的来源（h^(0)=[x, s, f, t̃]）。"""

    # f_i 在原文里是"由分割网络（如 nnU-Net encoder）提取的外观特征向量"。
    # 本实现提供两条路：
    #   "intensity"   = 用实例内像素强度统计（mean/std）近似 f（**工程近似**，
    #                   原仓库用的就是这条；论文若要写"用了 encoder 特征"必须换成下一条）；
    #   "encoder_npz" = 读取侧车 npz（每个检测一行特征），由云端 nnU-Net encoder
    #                   前向 hook 导出。接口已留好，导出脚本见 scripts/。
    f_source: str = "intensity"
    encoder_feat_path: str | None = None


@dataclass
class GNNConfig:
    """§2.0.1 GNN（式 30-35）。"""

    hidden: int = 64
    layers: int = 3
    dropout: float = 0.1
    lambda_ot: float = 0.2             # CALIBRATED 式(35) L_OT-reg 的权重
    # 式(34) 是**边级二元交叉熵**（y_e∈{0,1}，式29），本实现不做类别加权。
    class_weighted_ce: bool = False    # ENG_SUPP 若置 True 则用类频倒数加权（非原文口径）


@dataclass
class PipelineConfig:
    """整条链路配置（可 yaml 序列化，随实验八件套入库）。"""

    measure: MeasureConfig = field(default_factory=MeasureConfig)
    coupling: CouplingConfig = field(default_factory=CouplingConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    multiscale: MultiscaleConfig = field(default_factory=MultiscaleConfig)
    tracklet: TrackletConfig = field(default_factory=TrackletConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    reconstruct: ReconstructConfig = field(default_factory=ReconstructConfig)
    node: NodeFeatureConfig = field(default_factory=NodeFeatureConfig)
    gnn: GNNConfig = field(default_factory=GNNConfig)
    seed: int = 20260923
    # 记录"这一轮检测来自哪一档"（AGENTS.md §8.3 要求分别报告，不得混用）
    detection_source: str = "nnunet_pred"


# ---------------------------------------------------------------------------
# 序列化与点号覆盖（与仓库既有脚本风格一致）
# ---------------------------------------------------------------------------

def _to_plain(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {(_k if isinstance(_k, (str, int)) else str(_k)): _to_plain(v)
                for _k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    return obj


def save_config(cfg: PipelineConfig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(_to_plain(asdict(cfg)), allow_unicode=True,
                                   sort_keys=False))
    return path


def load_config(path: str | Path | None = None) -> PipelineConfig:
    """从 yaml 读配置；缺省字段用 dataclass 默认值补齐（向后兼容）。"""
    cfg = PipelineConfig()
    if path is None:
        return cfg
    data = yaml.safe_load(Path(path).read_text()) or {}
    sections = {f.name: f.type for f in fields(PipelineConfig)}
    for name, values in data.items():
        if name not in sections or not isinstance(values, dict):
            setattr(cfg, name, values)
            continue
        section = getattr(cfg, name)
        for k, v in values.items():
            if not hasattr(section, k):
                raise KeyError(f"未知配置项 {name}.{k}")
            cur = getattr(section, k)
            if isinstance(v, list) and isinstance(cur, tuple):
                v = tuple(v)
            if name == "multiscale" and k == "lambda_temp" and isinstance(v, dict):
                v = {int(kk): float(vv) for kk, vv in v.items()}
            setattr(section, k, v)
    return cfg


def override(cfg: PipelineConfig, pairs: list[str]) -> PipelineConfig:
    """按 `段.字段=值` 覆盖（如 `coupling.eta=0`）。"""
    def coerce(v: str):
        low = v.lower()
        if low in ("true", "false"):
            return low == "true"
        if low in ("none", "null"):
            return None
        for cast in (int, float):
            try:
                return cast(v)
            except ValueError:
                pass
        if "," in v:
            try:
                return tuple(float(x) for x in v.split(","))
            except ValueError:
                pass
        return v

    for kv in pairs:
        key, sep, val = kv.partition("=")
        if not sep:
            raise ValueError(f"覆盖项格式应为 段.字段=值，收到 {kv!r}")
        sec, _, fld = key.partition(".")
        target = getattr(cfg, sec)
        v = coerce(val)
        if sec == "multiscale" and fld == "lambda_temp" and isinstance(v, tuple):
            v = {int(v[i]): float(v[i + 1]) for i in range(0, len(v) - 1, 2)}
        setattr(target, fld, v)
    return cfg


def to_jsonable(cfg: PipelineConfig) -> str:
    return json.dumps(_to_plain(asdict(cfg)), ensure_ascii=False, indent=2)


def with_section(cfg: PipelineConfig, name: str, **kw) -> PipelineConfig:
    return replace(cfg, **{name: replace(getattr(cfg, name), **kw)})
