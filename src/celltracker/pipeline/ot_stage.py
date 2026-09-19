"""§1.3 相邻帧最优传输（式 8/9/11/12/14）+ §1.5 运动先验（式 20-22）。

这是 A2 的核心：把 FGW 结构项（η）、非平衡 KL 松弛（τ）、熵正则（ε）
与运动先验（α′）统一到一个可复用、可落盘的阶段，供候选边生成（式26）与
边特征（式27）消费。

对应原文：
  式(8)  C_feat_ij = α‖x_i−x_j‖² + β((s_i−s_j)/σ_s)²          （+R_max 门限）
  式(9)  L(Γ) = (1−η)Σ C_ij Γ_ij + η Σ (D_ik−D'_jl)² Γ_ij Γ_kl
  式(11) Ent(Γ) = Σ Γ_ij (log Γ_ij − 1)
  式(12) 平衡 OT：Γ1 = a, Γᵀ1 = b
  式(14) 非平衡 OT：τ_a·KL(Γ1‖a) + τ_b·KL(Γᵀ1‖b)
  式(22) C̃_ij = α‖x_i−x_j‖² + α′‖x̂_i−x_j‖² + β((s_i−s_j)/σ_s)²

ε 的量纲说明：C 是平方距离（CE 数据的中位数量级 ~4×10²）。若直接取 ε=1，
传输计划会退化成硬分配（实测真实分裂边质量直接为 0），
因此默认用 `eps_rel × median(C)` 自适应。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..cost.features import CostConfig, build_cost, gaussian_knn_graph, masses
from ..ot.fgw import fused_gw
from ..ot.sinkhorn import sinkhorn_log
from .config import MeasureConfig, OTConfig

__all__ = ["CouplingArtifacts", "compute_pairwise_plan", "save_coupling", "load_coupling"]


@dataclass
class CouplingArtifacts:
    """一对相邻帧的 OT 中间产物（可落盘复用）。"""

    plan: np.ndarray                                  # Γ (n_src, n_dst)
    cost: np.ndarray                                  # C_feat (n_src, n_dst)
    mass_a: np.ndarray
    mass_b: np.ndarray
    eps_eff: float
    info: dict = field(default_factory=dict)

    @property
    def rnorm(self) -> np.ndarray:
        """行归一化传输质量（式(26) 的候选边判据、式(27) 的边特征都用它）。"""
        return self.plan / (self.plan.sum(axis=1, keepdims=True) + 1e-12)

    def save(self, path: str | Path) -> Path:
        return save_coupling(self, path)


def compute_pairwise_plan(
    src_xy: np.ndarray,
    dst_xy: np.ndarray,
    cfg: OTConfig,
    src_vol: np.ndarray | None = None,
    dst_vol: np.ndarray | None = None,
    pred_xy: np.ndarray | None = None,
    measure: MeasureConfig | None = None,
    D: np.ndarray | None = None,
    Dp: np.ndarray | None = None,
    return_info: bool = True,
) -> CouplingArtifacts:
    """求解一对相邻帧的（非平衡 / FGW）熵正则 OT。

    参数
    ----
    pred_xy : (n_src, d) 匀速度外推位置（式22 的 x̂）；给定时启用运动先验 α′
    D, Dp   : 帧内距离矩阵（式7）；η>0 且未提供时按 kNN 图现场构造
    """
    measure = measure or MeasureConfig()
    n_src, n_dst = src_xy.shape[0], dst_xy.shape[0]

    # --- 式(8)(22) 特征代价 + R_max 门限 ---
    cost_cfg = CostConfig(alpha=cfg.alpha, alpha_pred=cfg.alpha_pred, beta=cfg.beta,
                          sigma_s=cfg.sigma_s, r_max=cfg.r_max,
                          mass_mode=measure.mass_mode)
    C, info = build_cost(src_xy, dst_xy, src_vol, dst_vol, pred_xy, cost_cfg)

    # --- 式(1) 质量向量 ---
    a = masses(src_vol, n_src, measure.mass_mode)
    b = masses(dst_vol, n_dst, measure.mass_mode)

    # --- 式(11) 熵正则强度：按代价尺度自适应，避免退化为硬分配 ---
    finite = C[np.isfinite(C)]
    eps_eff = (cfg.eps_rel * float(np.median(finite))
               if cfg.eps_rel is not None and finite.size else float(cfg.eps))

    # --- 式(9)(12)(14) 求解 ---
    if cfg.eta > 0:
        if D is None or Dp is None:
            D, _ = gaussian_knn_graph(src_xy, k=measure.knn_k,
                                      sigma_x=measure.sigma_x)
            Dp, _ = gaussian_knn_graph(dst_xy, k=measure.knn_k,
                                       sigma_x=measure.sigma_x)
        plan, extra = fused_gw(C, a, b, D, Dp, eta=cfg.eta, eps=eps_eff,
                               tau_a=cfg.tau_a, tau_b=cfg.tau_b)
    else:
        plan = sinkhorn_log(C, a, b, eps=eps_eff,
                            tau_a=cfg.tau_a, tau_b=cfg.tau_b, return_log=False)
        extra = {}

    art = CouplingArtifacts(
        plan=plan, cost=C, mass_a=a, mass_b=b, eps_eff=eps_eff,
        info={"eta": cfg.eta, "tau_a": cfg.tau_a, "tau_b": cfg.tau_b,
              "alpha_pred": cfg.alpha_pred, "r_max": cfg.r_max,
              "n_candidates": int(np.isfinite(C).sum()), **extra})
    if return_info:
        art.info["cmax"] = float(np.nanmax(C[np.isfinite(C)])) if finite.size else 0.0
    return art


def save_coupling(art: CouplingArtifacts, path: str | Path) -> Path:
    """落盘：npz 存数组 + json 存元信息（便于复现与调试）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, plan=art.plan, cost=art.cost,
                        mass_a=art.mass_a, mass_b=art.mass_b,
                        eps_eff=np.array([art.eps_eff]))
    path.with_suffix(".json").write_text(
        json.dumps(art.info, ensure_ascii=False, indent=2, default=str))
    return path


def load_coupling(path: str | Path) -> CouplingArtifacts:
    path = Path(path)
    z = np.load(path)
    info_path = path.with_suffix(".json")
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    return CouplingArtifacts(plan=z["plan"], cost=z["cost"], mass_a=z["mass_a"],
                             mass_b=z["mass_b"], eps_eff=float(z["eps_eff"][0]),
                             info=info)
