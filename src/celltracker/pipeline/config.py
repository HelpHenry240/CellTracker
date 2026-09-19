"""Pipeline 统一配置与消融开关（Phase A1 的接口层）。

设计目标（见 AGENTS.md）：
  1. 每个论文模块对应**一段独立配置**，可单独开关；
  2. 提供**统一的消融入口** `apply_ablation(cfg, {"fgw", "unbalanced", ...})`，
     使消融实验只是"换个开关组合"，不需要改代码；
  3. 配置可 yaml 序列化，随实验记录入库。

对应关系（论文 → 配置段）：
  §1.2 式(1)(4-7)    → MeasureConfig
  §1.3 式(8)(9)(11)(12)(14) → OTConfig
  §1.5 式(20-22)     → 运动先验（OTConfig.alpha_pred，两遍式由 runner 驱动）
  §1.4 式(17-19)     → MultiscaleConfig
  §1.6 tracklet 二层 OT → TrackletConfig
  §2.0.1 式(25-28)   → GraphConfig
  §1.6 式(23-24)     → ReconstructConfig
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import yaml

__all__ = [
    "MeasureConfig", "OTConfig", "MultiscaleConfig", "TrackletConfig",
    "GraphStageConfig", "ReconstructConfig", "PipelineConfig",
    "ABLATIONS", "apply_ablation", "load_config", "save_config",
]


@dataclass
class MeasureConfig:
    """§1.2 帧内表示：经验测度 + 邻域图。"""

    mass_mode: str = "uniform"     # 式(1)：uniform（依据 decisions/0001）或 volume
    knn_k: int = 6                 # 式(5)：帧内 kNN 度数
    sigma_x: float | None = None   # 式(6)：空间尺度（None=按中位距离自适应）
    sigma_f: float | None = None   # 式(6)：特征尺度


@dataclass
class OTConfig:
    """§1.3 相邻帧最优传输（式 8/9/11/12/14）。"""

    # --- 式(8) 特征代价 ---
    alpha: float = 1.0             # α：位移代价
    beta: float = 0.0              # β：尺寸变化代价
    sigma_s: float = 1.0           # σ_s：尺寸归一化
    r_max: float = 30.0            # R_max：候选位移上限（超出置 +inf）
    # --- 式(22) 运动先验（两遍式第二步启用）---
    alpha_pred: float = 0.0        # α′：匀速外推残差代价
    # --- 式(9) FGW 结构项 ---
    eta: float = 0.0               # η：结构项权重（0 = 纯特征 OT）
    # --- 式(11)(12) 熵正则 ---
    eps: float = 1.0               # ε 绝对值（eps_rel 优先）
    eps_rel: float | None = 0.1    # ε = eps_rel × median(C)，避免量纲失配
    # --- 式(14) 非平衡 OT（KL 松弛）---
    tau_a: float | None = None     # None = 平衡 OT（硬边缘约束）
    tau_b: float | None = None


@dataclass
class MultiscaleConfig:
    """§1.4 多尺度时间一致性（式 17-19）。"""

    enabled: bool = False          # 关闭时退化为逐对独立求解
    ks: tuple[int, ...] = (2, 3, 5)  # K：时间尺度集合
    # λ_temp：时间正则权重。实现里已把正则梯度按量级归一化并缩放到代价尺度，
    # 所以 λ_temp 的含义是"每轮允许的扰动幅度 = λ_temp × median(C)"。
    # 交替优化**没有线搜索**，λ_temp 过大会震荡甚至发散（实测 ≥0.5 时正则跳升 3 个量级），
    # 建议取值区间 [0.01, 0.2]；加入回溯线搜索是 Phase B 的候选改进。
    lambda_temp: float = 0.1
    n_rounds: int = 2              # 交替优化轮数


@dataclass
class TrackletConfig:
    """§1.6 短轨迹片段与二层 OT。"""

    enabled: bool = False
    max_gap: int = 3               # 允许合并的最大时间间隔
    r_max: float = 30.0            # 门限（实际门限 = r_max × gap）
    velocity_weight: float = 1.0
    merge_division_children: bool = False
    theta_link: float = 0.2        # 第二层 OT 的行归一化质量下限（接受关联）
    tau: float = 0.5               # 第二层用**非平衡** OT：轨迹可在此终止/起始
    # 关键格式约束：CTC 要求轨迹在起止帧之间**每帧都出现**。
    # 因此只有 gap==1（前后紧邻）的合并天然合法；gap>1 会留下帧空洞，
    # 提交时会被格式校验拆回去（实测：19 个连接里 16 个被拆断）。
    # 若要支持跨空洞的遮挡恢复，必须**在空洞帧里补出检测/掩码**（见 Phase C）。
    allow_gap_filling: bool = False


@dataclass
class GraphStageConfig:
    """§2.0.1 时间展开图（式 25-28）。"""

    cand_from_ot: bool = True      # 式(26)：候选边由 OT 筛选（False = 纯 R_max 几何门控）
    theta_gamma: float = 0.02      # 式(26)：Γ_ij ≥ θ_Γ（行归一化质量）
    theta_c: float | None = None   # 式(26)：C_ij ≤ θ_C（None → r_max²）
    cand_topk: int = 3             # 工程补充：每行额外保留质量前 k（见 day1_summary）
    window: int = 0                # 时间上下文窗口（0 = 仅相邻两帧）
    intra_knn: int = 4             # 式(28)：帧内 kNN 边


@dataclass
class ReconstructConfig:
    """§1.6 轨迹重建（式 23-24）。"""

    tau_move: float = 0.5          # 接受"移动边"的分数阈值
    tau_div: float = 0.5           # 接受"分裂边"的分数阈值
    max_children: int = 2


@dataclass
class PipelineConfig:
    """整条链路的配置（可 yaml 序列化）。"""

    measure: MeasureConfig = field(default_factory=MeasureConfig)
    ot: OTConfig = field(default_factory=OTConfig)
    multiscale: MultiscaleConfig = field(default_factory=MultiscaleConfig)
    tracklet: TrackletConfig = field(default_factory=TrackletConfig)
    graph: GraphStageConfig = field(default_factory=GraphStageConfig)
    reconstruct: ReconstructConfig = field(default_factory=ReconstructConfig)
    seed: int = 20260919
    # 检测来源：gt_tra（分割无关上界设定）| nnunet（Phase C）
    detection_source: str = "gt_tra"
    # 记录本次实验被关掉的模块（由 apply_ablation 写入）
    ablated: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# 消融开关：名字 → 配置修改
# ---------------------------------------------------------------------------

ABLATIONS: dict[str, str] = {
    "fgw": "式(9) 结构项：η → 0（退化为纯特征 OT）",
    "unbalanced": "式(14) 非平衡 OT：τ → None（退化为平衡 OT）",
    "motion": "式(20-22) 运动先验：α′ → 0",
    "multiscale": "式(17-19) 多尺度时间正则：关闭",
    "tracklet": "§1.6 tracklet 二层 OT：关闭",
    "ot_cand": "式(26) OT 候选筛选：关闭（退化为 R_max 几何门控）",
    "cand_topk": "工程补充 top-k 保底：关闭",
    "gnn": "§2.0.1 GNN：关闭（用 OT/式23-24 规则重建）",
}


def apply_ablation(cfg: PipelineConfig, ablate: set[str] | list[str]) -> PipelineConfig:
    """按消融名集合关闭对应模块，返回**新配置**（不修改入参）。"""
    names = set(ablate)
    unknown = names - set(ABLATIONS)
    if unknown:
        raise ValueError(f"未知消融项 {sorted(unknown)}；可选：{sorted(ABLATIONS)}")

    ot = replace(cfg.ot)
    ms = replace(cfg.multiscale)
    tl = replace(cfg.tracklet)
    gr = replace(cfg.graph)
    if "fgw" in names:
        ot.eta = 0.0
    if "unbalanced" in names:
        ot.tau_a = ot.tau_b = None
    if "motion" in names:
        ot.alpha_pred = 0.0
    if "multiscale" in names:
        ms.enabled = False
    if "tracklet" in names:
        tl.enabled = False
    if "ot_cand" in names:
        gr.cand_from_ot = False
    if "cand_topk" in names:
        gr.cand_topk = 0
    return replace(cfg, ot=ot, multiscale=ms, tracklet=tl, graph=gr,
                   ablated=tuple(sorted(names)))


def save_config(cfg: PipelineConfig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(asdict(cfg), allow_unicode=True, sort_keys=False))
    return path


def load_config(path: str | Path) -> PipelineConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return PipelineConfig(
        measure=MeasureConfig(**raw.get("measure", {})),
        ot=OTConfig(**raw.get("ot", {})),
        multiscale=MultiscaleConfig(**{**raw.get("multiscale", {}),
                                       **({"ks": tuple(raw["multiscale"]["ks"])}
                                          if "ks" in raw.get("multiscale", {}) else {})}),
        tracklet=TrackletConfig(**raw.get("tracklet", {})),
        graph=GraphStageConfig(**raw.get("graph", {})),
        reconstruct=ReconstructConfig(**raw.get("reconstruct", {})),
        seed=raw.get("seed", 20260919),
        detection_source=raw.get("detection_source", "gt_tra"),
    )
