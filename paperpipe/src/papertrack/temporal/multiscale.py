"""ideas.pdf 式(17)–(19)：在固定原始代价上联合精炼多尺度耦合。

F({Γ}) = Σ_t [L(Γ_t)+εEnt(Γ_t)+τ_a KL(Γ_t1‖a)+τ_b KL(Γ_tᵀ1‖b)]
         + Σ_k λ_k Σ_t ‖Γ_direct(t,k) − Γ_t…Γ_(t+k−1)‖²_F。

对一对帧进行广义条件梯度更新：线性化结构项和时间正则，保留熵及 KL，
用 Sinkhorn 得到方向，再在完整固定目标上回溯。原始 C 始终供候选筛选和 GNN 使用。
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from celltracker.ot.fgw import (entropy, kl_divergence, regularized_objective,
                                structural_grad)
from celltracker.ot.sinkhorn import sinkhorn_log
from ..config import CouplingConfig, MeasureConfig, MultiscaleConfig
from ..coupling.pairwise import PairCoupling, solve_jump_coupling
from ..representation.measure import knn_structure


@dataclass
class MultiscaleResult:
    couplings: dict[int, PairCoupling]
    direct: dict[tuple[int, int], PairCoupling] = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)
    info: dict = field(default_factory=dict)


def pair_objective(P, C, D, Dp, a, b, ccfg, eps):
    return regularized_objective(P, C, D, Dp, a, b, ccfg.eta, eps, ccfg.tau_a, ccfg.tau_b)


def time_expanded_edges(couplings, ts, theta_frac=0.0):
    """式(15)/(16)：返回 Γ>0 的前向边及质量；theta_frac 是可选工程筛选。"""
    edges = []
    for pos, art in sorted(couplings.items()):
        for i, j in zip(*np.where(art.plan > 0)):
            value = float(art.plan[i, j])
            if value >= float(theta_frac or 0) * art.mass_a[i]:
                edges.append((ts[pos], int(i), int(j), value))
    return edges


def normalized_plan(plan, mode):
    if mode == "raw":
        return plan
    if mode == "cond":
        return plan / (plan.sum(axis=1, keepdims=True) + 1e-12)
    raise ValueError(f"未知乘积空间 {mode}")


def product_regularizer(plans, direct, mode="raw"):
    """式(18)/(19) 的单个窗口项以及每个输入 Γ 的解析梯度。

    cond 对照令 Q=Γ/(r+δ)，链式导数为 (g−Σ_j g_j Q_j)/(r+δ)。
    raw 直接在质量空间求导，不进行额外归一化。
    """
    matrices = [normalized_plan(p, mode) for p in plans]
    left = [np.eye(matrices[0].shape[0])]
    for matrix in matrices:
        left.append(left[-1] @ matrix)
    right = [None] * (len(matrices) + 1)
    right[-1] = np.eye(matrices[-1].shape[1])
    for i in reversed(range(len(matrices))):
        right[i] = matrices[i] @ right[i+1]
    residual = left[-1] - normalized_plan(direct, mode)
    gradients = []
    for i, plan in enumerate(plans):
        grad = 2 * left[i].T @ residual @ right[i+1].T
        if mode == "cond":
            row_sum = plan.sum(axis=1, keepdims=True) + 1e-12
            grad = (grad - np.sum(grad * matrices[i], axis=1, keepdims=True)) / row_sum
        gradients.append(grad)
    return float(np.sum(residual**2)), gradients


def refine_couplings(couplings, ts, frame_xy, frame_vol, cfg, ccfg, mcfg,
                     spacing=None, pred_xy=None):
    if not cfg.enabled or not couplings:
        return MultiscaleResult(couplings=dict(couplings), info={"enabled": False})
    base = couplings
    plans = {i: art.plan.copy() for i, art in base.items()}
    distances = {}
    if ccfg.eta:
        distances = {i: knn_structure(frame_xy[t], frame_vol.get(t), mcfg, spacing,
                         d_full=mcfg.structure_mode == "full")[0] for i, t in enumerate(ts)}
    direct = {}
    jump_cfg = replace(ccfg, r_max=ccfg.r_max * cfg.jump_r_max_scale)
    weights = {int(k): float(v) for k, v in cfg.lambda_temp.items()}
    for i in sorted(base):
        for k in cfg.ks:
            k = int(k)
            if i+k < len(ts):
                direct[i,k] = solve_jump_coupling(frame_xy[ts[i]], frame_xy[ts[i+k]], k,
                    jump_cfg, mcfg, frame_vol[ts[i]], frame_vol[ts[i+k]], spacing, eps=base[i].eps_eff)
    windows = {i: [key for key in direct if key[0] <= i < sum(key)] for i in plans}

    def base_objective(i, P):
        a = base[i]
        return pair_objective(P, a.cost, distances.get(i), distances.get(i+1),
                              a.mass_a, a.mass_b, ccfg, a.eps_eff)

    def regularizer(key, current):
        start, k = key
        return product_regularizer([current[j] for j in range(start, start+k)], direct[key].plan, cfg.norm)

    def local_objective(i, current):
        return base_objective(i, current[i]) + sum(weights.get(k, 0) * regularizer((a,k), current)[0]
                                                   for a,k in windows[i])

    base_values = {i: base_objective(i, plans[i]) for i in plans}
    def record(round_index, accepted):
        reg = sum(weights.get(k, 0) * regularizer((a,k), plans)[0] for a,k in direct)
        return {"round": round_index, "objective": float(sum(base_values.values())+reg),
                "reg": float(reg), "accepted_updates": accepted}
    history = [record(0, 0)]
    for round_index in range(1, max(0, cfg.n_rounds)+1):
        accepted = 0
        for i in sorted(plans):
            art, P = base[i], plans[i]
            if P.size == 0:
                continue
            finite = np.isfinite(art.cost)
            G = (1 - ccfg.eta) * np.where(finite, art.cost, 0.0)
            if ccfg.eta:
                G += ccfg.eta * structural_grad(P, distances[i], distances[i+1])
            for start,k in windows[i]:
                _, gradients = regularizer((start,k), plans)
                G += weights.get(k, 0) * gradients[i-start]
            G = np.where(finite, G, np.inf)
            S = sinkhorn_log(G, art.mass_a, art.mass_b, eps=art.eps_eff,
                             tau_a=ccfg.tau_a, tau_b=ccfg.tau_b, n_iter=max(ccfg.sinkhorn_iters,10000) if ccfg.tau_a is None or ccfg.tau_b is None else ccfg.sinkhorn_iters, strict_marginals=True)
            if not np.isfinite(S).all():
                continue
            before = local_objective(i, plans)
            step = 1.0
            for _ in range(24):
                trial = dict(plans)
                trial[i] = (1-step)*P + step*S
                value = local_objective(i, trial)
                # 关闭回溯是显式求解器消融；主配置始终检查固定完整目标。
                if (not cfg.line_search and np.isfinite(value)) or value <= before:
                    plans[i] = trial[i]
                    base_values[i] = base_objective(i, plans[i])
                    accepted += 1
                    break
                step *= 0.5
        history.append(record(round_index, accepted))
    refined = {i: replace(art, plan=plans[i], info={**art.info, "multiscale_refined": True,
                    "norm": cfg.norm, "ks": list(cfg.ks)}) for i, art in base.items()}
    return MultiscaleResult(refined, direct, history,
        {"enabled": True, "norm": cfg.norm, "ks": list(cfg.ks), "lambda_temp": weights,
         "n_rounds": cfg.n_rounds, "line_search": cfg.line_search, "history": history,
         "original_cost_preserved": True, "objective_kind": "eq17_fixed_complete"})


def resolve_with_cost(art, G, ccfg, Ds=None):
    """求解线性化子问题；代理 G 不覆盖外部可见的原始特征代价 C。"""
    plan = sinkhorn_log(G, art.mass_a, art.mass_b, eps=art.eps_eff,
                       tau_a=ccfg.tau_a, tau_b=ccfg.tau_b, n_iter=max(ccfg.sinkhorn_iters,10000) if ccfg.tau_a is None or ccfg.tau_b is None else ccfg.sinkhorn_iters, strict_marginals=True)
    return replace(art, plan=plan) if np.isfinite(plan).all() else None
