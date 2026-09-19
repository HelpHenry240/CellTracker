"""§1.4 多尺度时间一致性与全局目标（式 17-19）—— pipeline 阶段实现。

原文目标（式17）::

    min_{Γ}  Σ_t [ L(Γ_t) + ε·Ent(Γ_t) + τ_t·KL(Γ_t 1‖a_t) + τ_{t+1}·KL(Γ_tᵀ1‖a_{t+1}) ]
           + λ_temp Σ_{k∈K} Σ_t R^(k)(Γ_t … Γ_{t+k−1})

其中多尺度一致性（式19）::

    R^(k) = ‖ D_{t,t+k} − Γ_t Γ_{t+1} … Γ_{t+k−1} ‖_F²

`D_{t,t+k}` 是帧 t 与 t+k 之间的**直接** OT 耦合（式17 的下标说明允许在粗分辨率上求）。

实现策略：交替优化——先用相邻帧独立解初始化，再按时间正则的梯度对每个 Γ_t 做少量重解::

    Γ_t ← Sinkhorn( C_t + λ_temp · Σ_k ∂R^(k)/∂Γ_t , a_t, a_{t+1} )

本模块只消费/产出 `CouplingArtifacts`，与 OT 阶段的接口一致，
因此可以插在 OT 阶段与图构建阶段之间，且 `enabled=False` 时是完全的恒等映射（消融对照干净）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..ot.multiscale import direct_jump_coupling
from ..ot.sinkhorn import sinkhorn_log
from ..track.base import Detections
from .config import MeasureConfig, MultiscaleConfig, OTConfig
from .ot_stage import CouplingArtifacts

__all__ = ["MultiscaleResult", "refine_couplings"]


@dataclass
class MultiscaleResult:
    """多尺度精炼结果。`couplings` 的键是帧在 `dets.t_range` 中的**索引** i，
    对应相邻帧对 (ts[i], ts[i+1])。"""

    couplings: dict[int, CouplingArtifacts]
    history: list[float] = field(default_factory=list)      # 每轮的时间正则值
    direct: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)
    info: dict = field(default_factory=dict)


def refine_couplings(
    dets: Detections,
    couplings: dict[int, CouplingArtifacts],
    cfg: MultiscaleConfig,
    ot_cfg: OTConfig,
    measure: MeasureConfig | None = None,
) -> MultiscaleResult:
    """按式(17)-(19) 精炼相邻帧耦合。

    `couplings[i]` 对应帧对 (ts[i], ts[i+1])；返回同键结构的结果。
    `cfg.enabled=False` 时原样返回（恒等映射）。
    """
    measure = measure or MeasureConfig()
    if not cfg.enabled or not couplings:
        return MultiscaleResult(couplings=dict(couplings), history=[],
                                info={"enabled": False})

    ts = dets.t_range
    n_frames = len(ts)
    ks = [int(k) for k in cfg.ks if int(k) >= 1]
    lam_ks = {k: float(cfg.lambda_temp) for k in ks}

    # 当前耦合（按索引）
    P: dict[int, np.ndarray] = {i: np.array(c.plan, dtype=float) for i, c in couplings.items()}
    base_cost: dict[int, np.ndarray] = {i: np.array(c.cost, dtype=float)
                                        for i, c in couplings.items()}
    mass: dict[int, tuple[np.ndarray, np.ndarray]] = {
        i: (np.array(c.mass_a, dtype=float), np.array(c.mass_b, dtype=float))
        for i, c in couplings.items()}
    eps_eff: dict[int, float] = {i: float(c.eps_eff) for i, c in couplings.items()}

    # 式(19) 的直接跳帧耦合 D_{t,t+k}（在粗分辨率/子采样上求解以控复杂度）
    direct: dict[tuple[int, int], np.ndarray] = {}
    for i in range(n_frames):
        for k in ks:
            if i + k <= n_frames - 1:
                direct[(i, k)] = direct_jump_coupling(
                    dets.centroid(ts[i]), dets.centroid(ts[i + k]), k, ot_cfg,
                    eps=eps_eff.get(i, ot_cfg.eps),
                    tau_a=ot_cfg.tau_a, tau_b=ot_cfg.tau_b)

    def _prod(idxs) -> np.ndarray:
        idxs = list(idxs)
        out = _cond(P[idxs[0]])
        for i in idxs[1:]:
            out = out @ _cond(P[i])
        return out

    def _cond(M: np.ndarray) -> np.ndarray:
        """行归一化为条件概率（式17 下方的定义）。

        **为什么必须归一化**：Γ 的行和是质量 a_t（≈1/n），两个 Γ 相乘会让质量
        被平方（实测：D 的行和 1/43，而 Γ_tΓ_{t+1} 的行和 ≈ (1/43)²）。
        直接相减等于拿两个不同质量尺度的东西比较，式(18) 也就失去意义。
        原文在 §1.4 给出条件概率定义
        `P((j,t+1)|(i,t)) = Γ_ij / (Σ_j' Γ_ij' + δ)`，复合即在该空间进行。
        """
        return M / (M.sum(axis=1, keepdims=True) + 1e-12)

    def _reg_value(P_cur: dict[int, np.ndarray]) -> float:
        """当前耦合下的时间正则总值（式19 的加权和，不含 λ 之外的缩放）。"""
        total = 0.0
        for (i, k), D in direct.items():
            idxs = list(range(i, i + k))
            if any(j not in P_cur for j in idxs):
                continue
            out = _cond(P_cur[idxs[0]])
            for j in idxs[1:]:
                out = out @ _cond(P_cur[j])
            total += lam_ks[k] * float(np.sum((_cond(D) - out) ** 2))
        return total

    def _gauss_seidel_once(P_cur: dict[int, np.ndarray], step: float) -> dict[int, np.ndarray]:
        """一轮 Gauss-Seidel 更新（每对帧依次用扰动后的代价重解 Sinkhorn）。"""
        out = {i: P_cur[i].copy() for i in P_cur}
        for i in range(n_frames - 1):
            if i not in out:
                continue
            G = np.where(np.isfinite(base_cost[i]), base_cost[i], 0.0)
            grad = np.zeros_like(G)
            # (a) Γ_i 作为乘积**首因子**：区间 (i, k)
            for k in lam_ks:
                if (i, k) not in direct:
                    continue
                tail = (np.eye(out[i].shape[1]) if k == 1
                        else _prod_range(out, range(i + 1, i + k)))
                diff = (_cond(out[i]) @ tail) - _cond(direct[(i, k)])
                grad = grad + 2.0 * (diff @ tail.T)
            # (b) Γ_i 作为乘积的**中间/末尾因子**：区间 (j, k)，j < i ≤ j+k−1
            for k in lam_ks:
                for j in range(max(0, i - k + 1), i):
                    if (j, k) not in direct:
                        continue
                    head = (np.eye(out[j].shape[0]) if j == i
                            else _prod_range(out, range(j, i)))
                    tail_idxs = list(range(i + 1, j + k))
                    tail = (np.eye(out[i].shape[1]) if not tail_idxs
                            else _prod_range(out, tail_idxs))
                    diff = (head @ _cond(out[i]) @ tail) - _cond(direct[(j, k)])
                    grad = grad + 2.0 * (head.T @ diff @ tail.T)
            # --- 量纲归一化（关键，见模块 docstring）---
            cost_scale = float(np.median(base_cost[i][np.isfinite(base_cost[i])])) \
                if np.isfinite(base_cost[i]).any() else 1.0
            # RMS 而非中位数：耦合在 log-domain Sinkhorn 下稀疏，median 可能恰为 0
            g_scale = float(np.sqrt(np.mean(grad ** 2)))
            if g_scale > 0:
                G = G + step * grad * (cost_scale / g_scale)
            G = np.where(np.isfinite(base_cost[i]), G, np.inf)
            a_i, b_i = mass[i]
            out[i] = sinkhorn_log(G, a_i, b_i, eps=eps_eff[i],
                                  tau_a=ot_cfg.tau_a, tau_b=ot_cfg.tau_b)
        return out

    def _prod_range(P_cur: dict[int, np.ndarray], idxs) -> np.ndarray:
        idxs = list(idxs)
        out = _cond(P_cur[idxs[0]])
        for j in idxs[1:]:
            out = out @ _cond(P_cur[j])
        return out

    # --- 交替优化 + 回溯线搜索 ---
    # 无线搜索时固定步长会把"本来已经一致"的解破坏掉（实测：在匀速运动的
    # 合成长序列上正则从 0.0037 跳升到 2.32）。这里按"时间正则不增"回溯半砍步长，
    # 使精炼**单调不增**，也让 λ_temp 的含义更稳定。
    history: list[float] = []
    for _round in range(max(int(cfg.n_rounds), 1)):
        reg_before = _reg_value(P)
        step = float(cfg.lambda_temp)
        best_P, best_reg = P, reg_before
        for _attempt in range(5):
            cand = _gauss_seidel_once(P, step)
            reg_cand = _reg_value(cand)
            if reg_cand <= reg_before:
                best_P, best_reg = cand, reg_cand
                break
            step *= 0.5
        P = best_P
        history.append(best_reg)

    # 包装回 CouplingArtifacts（保留原 cost/质量/ε，仅替换计划）
    out: dict[int, CouplingArtifacts] = {}
    for i, c in couplings.items():
        out[i] = CouplingArtifacts(
            plan=P[i], cost=c.cost, mass_a=c.mass_a, mass_b=c.mass_b,
            eps_eff=c.eps_eff,
            info={**(c.info or {}), "multiscale": {"ks": list(ks),
                                                   "lambda_temp": cfg.lambda_temp,
                                                   "n_rounds": cfg.n_rounds}},
            d_cur=c.d_cur)
    return MultiscaleResult(couplings=out, history=history, direct=direct,
                            info={"enabled": True, "ks": list(ks),
                                  "lambda_temp": cfg.lambda_temp,
                                  "n_pairs": len(out),
                                  "lambda_temp": cfg.lambda_temp})
