"""§1.3 相邻帧最优传输耦合（ideas.pdf 式 8–14）。

原文公式（R1 抄录）
------------------
式(9)   L(Γ^t) = (1−η) Σ_{i,j} C_feat_ij Γ_ij
               + η Σ_{i,k} Σ_{j,ℓ} (D^t_ik − D^{t+1}_jℓ)² Γ_ij Γ_kℓ
式(10)  Σ_j Γ^t_ij = a_i^t ,  Σ_i Γ^t_ij = a_j^{t+1}
式(11)  Ent(Γ^t) = Σ_{i,j} Γ_ij (log Γ_ij − 1)
式(12)  min_{Γ≥0} L(Γ^t) + ε Ent(Γ^t)  s.t. Γ^t 1 = a^t, (Γ^t)^⊤ 1 = a^{t+1}
式(13)  KL(u‖v) = Σ_i u_i log(u_i/v_i) − u_i + v_i
式(14)  min_{Γ≥0} L + ε Ent(Γ) + τ_t KL(Γ1‖a^t) + τ_{t+1} KL(Γ^⊤1‖a^{t+1})

复用与重写
----------
* 式(9)：**直接复用** `celltracker.ot.fgw`（`structural_term` 是四重和的等价矩阵形式，
  非近似；条件梯度 + Sinkhorn 线性 oracle）。
* 式(10)–(14)：**直接复用** `celltracker.ot.sinkhorn.sinkhorn_log`
  （log-domain 非平衡 Sinkhorn，缩放因子 λ=τ/(τ+ε) 正是式(14) 的不动点形式）。
* 式(8) 的代价：本包 `measure.build_cost` 重写（原仓库的 σ_s 分母多了 `mean(s_dst)`，
  且默认 β=0，而原文明确 α,β>0）。这一处差异见 FORMULA_MAP.md D2。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from celltracker.ot.fgw import fused_gw, structural_term  # 式(9)：直接复用
from celltracker.ot.sinkhorn import sinkhorn_log            # 式(10)-(14)：直接复用

from ..config import CouplingConfig, MeasureConfig
from ..representation.measure import build_cost, knn_structure, masses

__all__ = ["PairCoupling", "solve_coupling", "solve_jump_coupling",
           "save_coupling", "load_coupling", "eps_from"]


def eps_from(C: np.ndarray, cfg: CouplingConfig) -> tuple[float, str]:
    """式(11)(12) 的 ε（熵正则强度）。

    原文只要求 ε>0。若显式给 `cfg.eps` 则按原文标量使用；否则用
    **ENG_SUPP** `ε = eps_rel × median(C)`：C 是平方距离，取固定 ε 会随数据尺度
    失配（R3）。返回 (ε, 口径说明)。
    """
    if cfg.eps is not None:
        return float(cfg.eps), "explicit"
    finite = C[np.isfinite(C)]
    if cfg.eps_rel is not None and finite.size:
        return float(cfg.eps_rel * np.median(finite)), "median_scaled"
    return 1.0, "default_1.0"


@dataclass
class PairCoupling:
    """一对相邻帧的 OT 耦合及其全部中间量（可落盘复用）。"""

    plan: np.ndarray                 # Γ^t（式12/14 的解）
    cost: np.ndarray                 # C_feat（式8/22）
    mass_a: np.ndarray               # a^t（式1）
    mass_b: np.ndarray               # a^{t+1}（式3）
    d_cur: np.ndarray                # ‖x_i−x_j‖₂（µm）
    eps_eff: float                   # 实际使用的 ε
    info: dict = field(default_factory=dict)

    # ---- 供式(24)(26) 使用的原始量（**原文口径**：原始 Γ，不做行归一化） ----
    @property
    def row_sum(self) -> np.ndarray:
        """Γ1（每个源细胞实际流出的质量，式14）。"""
        return self.plan.sum(axis=1)

    @property
    def col_sum(self) -> np.ndarray:
        """Γ^⊤1（每个目标细胞收到的质量）。"""
        return self.plan.sum(axis=0)

    def theta_gamma(self, frac: float | None, absolute: float | None) -> np.ndarray:
        """式(24)/(26) 的 θ_Γ：默认按"占源质量的比"标定（见 config 注释）。"""
        if absolute is not None:
            return np.full(self.plan.shape[0], float(absolute))
        f = 0.05 if frac is None else float(frac)
        return f * np.maximum(self.mass_a, 1e-12)

    def theta_c(self, absolute: float | None, r_max: float) -> float:
        if absolute is not None:
            return float(absolute)
        return float(r_max) ** 2

    def save(self, path: str | Path) -> Path:
        return save_coupling(self, path)


def solve_coupling(src_xy: np.ndarray, dst_xy: np.ndarray,
                   src_vol: np.ndarray | None = None,
                   dst_vol: np.ndarray | None = None,
                   pred_xy: np.ndarray | None = None,
                   coupling_cfg: CouplingConfig | None = None,
                   measure_cfg: MeasureConfig | None = None,
                   spacing: tuple[float, ...] | None = None,
                   eps: float | None = None,
                   r_max_scale: float = 1.0) -> PairCoupling:
    """求解式(9)(12)(14)：一对帧之间的（FGW / 非平衡）熵正则 OT。

    `pred_xy` 给定时启用式(22) 的运动先验项 α′；
    `r_max_scale` 供跳帧耦合（式18/19 的 Γ^{t,t+k}_direct）按 gap 放宽 R_max。
    """
    from dataclasses import replace as _replace

    ccfg = coupling_cfg or CouplingConfig()
    mcfg = measure_cfg or MeasureConfig()
    if r_max_scale != 1.0:
        ccfg = _replace(ccfg, r_max=ccfg.r_max * float(r_max_scale))

    n_src, n_dst = int(np.asarray(src_xy).shape[0]), int(np.asarray(dst_xy).shape[0])
    if n_src == 0 or n_dst == 0:
        # 空帧：没有可传输的质量（对应"整帧漏检/序列边界"），返回空计划。
        empty = np.zeros((n_src, n_dst))
        return PairCoupling(plan=empty, cost=np.full((n_src, n_dst), np.inf),
                            mass_a=masses(src_vol, n_src, mcfg.mass_mode),
                            mass_b=masses(dst_vol, n_dst, mcfg.mass_mode),
                            d_cur=np.zeros((n_src, n_dst)), eps_eff=float(eps or 1.0),
                            info={"empty_frame": True, "n_src": n_src, "n_dst": n_dst})

    C, d_cur, cinfo = build_cost(src_xy, dst_xy, src_vol, dst_vol, pred_xy,
                                 ccfg, spacing)
    a = masses(src_vol, C.shape[0], mcfg.mass_mode)          # 式(1)
    b = masses(dst_vol, C.shape[1], mcfg.mass_mode)          # 式(3)
    eps_eff, eps_src = (float(eps), "given") if eps is not None else eps_from(C, ccfg)

    extra: dict = {}
    if ccfg.eta > 0:
        # 式(9)：FGW（特征项 + 结构项）—— 复用原仓库实现
        D, _, _ = knn_structure(src_xy, src_vol, mcfg, spacing)
        Dp, _, _ = knn_structure(dst_xy, dst_vol, mcfg, spacing)
        plan, extra = fused_gw(C, a, b, D, Dp, eta=ccfg.eta, eps=eps_eff,
                               tau_a=ccfg.tau_a, tau_b=ccfg.tau_b,
                               n_outer=int(ccfg.fgw_outer))
        extra["structural_term"] = structural_term(plan, D, Dp)
    else:
        # 式(12)（τ=None ⇒ 平衡）或式(14)（τ 有限 ⇒ 非平衡）
        plan = sinkhorn_log(C, a, b, eps=eps_eff, tau_a=ccfg.tau_a,
                            tau_b=ccfg.tau_b, n_iter=ccfg.sinkhorn_iters)

    info = {"eta": float(ccfg.eta), "tau_a": ccfg.tau_a, "tau_b": ccfg.tau_b,
            "alpha": float(ccfg.alpha), "alpha_pred": float(ccfg.alpha_pred),
            "beta": float(ccfg.beta), "r_max": float(ccfg.r_max),
            "eps": float(eps_eff), "eps_source": eps_src,
            "mass_mode": mcfg.mass_mode,
            "spacing_zyx": list(spacing) if spacing is not None else None,
            "n_finite_pairs": int(np.isfinite(C).sum()),
            "balance": "balanced" if (ccfg.tau_a is None and ccfg.tau_b is None)
                       else "unbalanced",
            **cinfo, **extra}
    return PairCoupling(plan=np.asarray(plan, dtype=float), cost=C, mass_a=a,
                        mass_b=b, d_cur=np.asarray(d_cur, dtype=float),
                        eps_eff=float(eps_eff), info=info)


def solve_jump_coupling(src_xy: np.ndarray, dst_xy: np.ndarray, gap: int,
                        coupling_cfg: CouplingConfig,
                        measure_cfg: MeasureConfig,
                        src_vol=None, dst_vol=None,
                        spacing: tuple[float, ...] | None = None,
                        eps: float | None = None) -> PairCoupling:
    """式(18)/(19) 的 `Γ^{t,t+k}_direct`：跨 k 帧的**直接** OT 耦合。

    原文（§1.7）："对空间坐标进行粗网格下采样 … 或对细胞随机采样一个子集" 只是
    控制复杂度的**工程选项**；本实现按 R_max×gap 放宽候选半径，在原始分辨率上求解。
    注意跳帧耦合不使用运动先验（α′=0），也不带结构项（η=0）——原文把 FGW 结构项
    定义在相邻帧上，跳帧项只作为一致性正则的目标。
    """
    from dataclasses import replace as _replace

    jcfg = _replace(coupling_cfg, alpha_pred=0.0, eta=0.0)
    return solve_coupling(src_xy, dst_xy, src_vol, dst_vol, None, jcfg,
                          measure_cfg, spacing, eps=eps,
                          r_max_scale=float(gap))


def save_coupling(c: PairCoupling, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, plan=c.plan, cost=c.cost, mass_a=c.mass_a,
                        mass_b=c.mass_b, d_cur=c.d_cur,
                        eps_eff=np.array([c.eps_eff]))
    path.with_suffix(".json").write_text(
        json.dumps(c.info, ensure_ascii=False, indent=2, default=str))
    return path


def load_coupling(path: str | Path) -> PairCoupling:
    path = Path(path)
    z = np.load(path, allow_pickle=False)
    jp = path.with_suffix(".json")
    info = json.loads(jp.read_text()) if jp.exists() else {}
    return PairCoupling(plan=z["plan"], cost=z["cost"], mass_a=z["mass_a"],
                        mass_b=z["mass_b"], d_cur=z["d_cur"],
                        eps_eff=float(z["eps_eff"][0]), info=info)
