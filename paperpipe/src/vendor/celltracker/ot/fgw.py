"""Fused Gromov–Wasserstein（式 (9)）的求解：条件梯度（Frank–Wolfe）+ Sinkhorn 线性 oracle。

目标::

    L(P) = (1-eta) <C, P> + eta * sum_{i,k,j,l} (D_ik - D'_jl)^2 P_ij P_kl

结构项的矩阵形式（r = P1, c = P^T 1）::

    S(P) = r^T (D∘D) r + c^T (D'∘D') c - 2 <P, D P D'>
    dS/dP = 2 (D∘D) r 1^T + 2 1 ((D'∘D') c)^T - 4 D P D'
"""

from __future__ import annotations

import numpy as np

from .sinkhorn import sinkhorn_log

__all__ = ["structural_term", "structural_grad", "fgw_objective", "fused_gw",
           "entropy", "kl_divergence", "regularized_objective"]


def entropy(P: np.ndarray) -> float:
    """ideas.pdf 式(11)：Σ P(log P − 1)，约定 0 log 0 = 0。"""
    p = np.asarray(P, dtype=float)
    positive = p > 0
    return float(np.sum(p[positive] * (np.log(p[positive]) - 1.0)))


def kl_divergence(u: np.ndarray, v: np.ndarray) -> float:
    """式(13) 的广义 KL，保留非单位总质量的 −u+v 项。"""
    u, v = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
    positive = u > 0
    if np.any((v <= 0) & positive):
        return float("inf")
    return float(np.sum(v - u) + np.sum(u[positive] * np.log(u[positive] / v[positive])))


def regularized_objective(P, C, D, Dp, a, b, eta, eps, tau_a=None, tau_b=None):
    """式(12)/(14)：L(P)+εEnt(P)+τ_a KL(P1‖a)+τ_b KL(Pᵀ1‖b)。

    tau=None 表示由求解器维护硬边际，因而这里不添加对应 KL 惩罚。
    禁止边上的非零质量属于不可行解，不能通过把无穷代价替换为零而忽略。
    """
    if not np.isfinite(P).all() or np.any(P < 0) or np.any(P[~np.isfinite(C)] > 0):
        return float("inf")
    value = (1.0 - eta) * float(np.sum(P * np.where(np.isfinite(C), C, 0.0)))
    if eta:
        value += eta * structural_term(P, D, Dp)
    value += eps * entropy(P)
    if tau_a is not None:
        value += tau_a * kl_divergence(P.sum(axis=1), a)
    if tau_b is not None:
        value += tau_b * kl_divergence(P.sum(axis=0), b)
    return float(value)


def structural_term(P: np.ndarray, D: np.ndarray, Dp: np.ndarray) -> float:
    r = P.sum(axis=1)
    c = P.sum(axis=0)
    D2 = D * D
    Dp2 = Dp * Dp
    return float(r @ (D2 @ r) + c @ (Dp2 @ c) - 2.0 * np.sum(P * (D @ P @ Dp)))


def structural_grad(P: np.ndarray, D: np.ndarray, Dp: np.ndarray) -> np.ndarray:
    r = P.sum(axis=1)
    c = P.sum(axis=0)
    D2 = D * D
    Dp2 = Dp * Dp
    return (2.0 * (D2 @ r)[:, None] + 2.0 * (Dp2 @ c)[None, :]
            - 4.0 * (D @ P @ Dp))


def fgw_objective(P: np.ndarray, C: np.ndarray, D: np.ndarray, Dp: np.ndarray,
                  eta: float) -> float:
    C_eff = np.where(np.isfinite(C), C, 0.0)
    return float((1.0 - eta) * np.sum(P * C_eff) + eta * structural_term(P, D, Dp))


def fused_gw(C: np.ndarray, a: np.ndarray, b: np.ndarray,
             D: np.ndarray, Dp: np.ndarray, eta: float = 0.3,
             eps: float = 0.05, tau_a: float | None = None,
             tau_b: float | None = None, n_outer: int = 40,
             tol: float = 1e-7, n_inner: int = 1000,
             strict_marginals: bool = False) -> tuple[np.ndarray, dict]:
    """广义条件梯度求解式(12)/(14)，在完整目标上回溯。

    只线性化光滑的特征/结构项，熵和边际 KL 留在 Sinkhorn 子问题中。
    线搜索在旧计划与子问题解之间插值，所以平衡情形保持边际约束。
    """
    if not 0 <= eta <= 1:
        raise ValueError("eta 必须在 [0,1] 内")
    finite = np.isfinite(C)
    # 硬边际在近置换计划上收敛较慢，给平衡子问题足够迭代预算。
    inner_iterations = max(n_inner, 10000) if tau_a is None and tau_b is None else n_inner
    initial_cost = np.where(finite, (1.0 - eta) * np.where(finite, C, 0.0), np.inf)
    P = sinkhorn_log(initial_cost, a, b, eps=eps, tau_a=tau_a, tau_b=tau_b, n_iter=inner_iterations, strict_marginals=strict_marginals)
    def objective(plan):
        return regularized_objective(plan, C, D, Dp, a, b, eta, eps, tau_a, tau_b)
    obj = objective(P)
    history = [obj]
    for _ in range(n_outer):
        G = (1.0 - eta) * np.where(np.isfinite(C), C, 0.0) \
            + eta * structural_grad(P, D, Dp)
        G = np.where(np.isfinite(C), G, np.inf)
        S = sinkhorn_log(G, a, b, eps=eps, tau_a=tau_a, tau_b=tau_b, n_iter=inner_iterations, strict_marginals=strict_marginals)

        gamma, improved = 1.0, False
        for _ls in range(20):
            Pn = (1.0 - gamma) * P + gamma * S
            objn = objective(Pn)
            if objn <= obj:
                improved = True
                break
            gamma /= 2.0
        if not improved:
            break
        P, obj_prev, obj = Pn, obj, objn
        history.append(obj)
        if abs(obj_prev - obj) <= tol * max(1.0, abs(obj_prev)):
            break

    return P, {"objective": obj, "history": history, "iters": len(history) - 1,
               "objective_kind": "eq12_eq14_complete",
               "feature_structure_objective": fgw_objective(P, C, D, Dp, eta)}
