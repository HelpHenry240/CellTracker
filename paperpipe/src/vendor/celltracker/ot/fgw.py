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

__all__ = ["structural_term", "structural_grad", "fgw_objective", "fused_gw"]


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
             tol: float = 1e-7) -> tuple[np.ndarray, dict]:
    """求解 FGW，返回 (P, info)。eta=0 时退化为带熵正则的（非平衡）OT。"""
    P = sinkhorn_log(C, a, b, eps=eps, tau_a=tau_a, tau_b=tau_b)
    obj = fgw_objective(P, C, D, Dp, eta)
    history = [obj]
    for _ in range(n_outer):
        G = (1.0 - eta) * np.where(np.isfinite(C), C, 0.0) \
            + eta * structural_grad(P, D, Dp)
        G = np.where(np.isfinite(C), G, np.inf)
        S = sinkhorn_log(G, a, b, eps=eps, tau_a=tau_a, tau_b=tau_b)

        gamma, improved = 1.0, False
        for _ls in range(20):
            Pn = (1.0 - gamma) * P + gamma * S
            objn = fgw_objective(Pn, C, D, Dp, eta)
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

    return P, {"objective": obj, "history": history, "iters": len(history) - 1}
