"""熵正则（非平衡）最优传输的 log-domain Sinkhorn 求解器。

求解::

    min_{P>=0}  <P, C> + eps * KL(P | K)
                + tau_a * KL(P 1 | a) + tau_b * KL(1^T P | b)

其中 `K = exp(-C/eps)`。当 `tau_a = tau_b = None`（无穷）时退化为**平衡 OT**：
硬约束 `P1 = a`、`P^T 1 = b`；否则为**非平衡 OT**（KL 松弛，允许出生/死亡/分裂）。

log-domain 迭代（数值稳定，允许 eps 很小）::

    f_i = lambda_a * eps * ( log a_i - logsumexp_j((g_j - C_ij)/eps) )
    g_j = lambda_b * eps * ( log b_j - logsumexp_i((f_i - C_ij)/eps) )

其中 `lambda = tau/(tau+eps) ∈ (0,1)`；平衡情形取 `lambda = 1`。
"""

from __future__ import annotations

import numpy as np

__all__ = ["sinkhorn_log"]


def _lse(M: np.ndarray, axis: int) -> np.ndarray:
    """log-sum-exp（抑制溢出）。"""
    m = np.max(M, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    out = m + np.log(np.sum(np.exp(M - m), axis=axis, keepdims=True) + 1e-300)
    return np.squeeze(out, axis=axis)


def sinkhorn_log(
    C: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    eps: float = 0.05,
    tau_a: float | None = None,
    tau_b: float | None = None,
    n_iter: int = 1000,
    tol: float = 1e-10,
    return_log: bool = False,
):
    """求解（非平衡）熵正则 OT，返回传输计划 `P`。

    参数
    ----
    C : (n, m) 代价矩阵（可用 `np.inf` 表示禁止匹配）
    a, b : (n,), (m,) 边际质量（正数，未归一化也可）
    eps : 熵正则强度
    tau_a, tau_b : KL 惩罚系数；`None` 表示无穷 → 硬边际约束（平衡 OT）
    """
    C = np.asarray(C, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n, m = C.shape
    if a.shape != (n,) or b.shape != (m,):
        raise ValueError(f"边际形状不匹配: C{C.shape}, a{a.shape}, b{b.shape}")
    if eps <= 0:
        raise ValueError("eps 必须为正")

    log_a = np.log(np.maximum(a, 1e-300))
    log_b = np.log(np.maximum(b, 1e-300))
    lam_a = 1.0 if tau_a is None else tau_a / (tau_a + eps)
    lam_b = 1.0 if tau_b is None else tau_b / (tau_b + eps)

    f = np.zeros(n)
    g = np.zeros(m)
    # 屏蔽禁止匹配：把 -inf 代价抬到有限大值，避免 nan
    C_eff = np.where(np.isinf(C), 1e12, C)

    err = np.inf
    for _ in range(n_iter):
        f_prev = f
        f = lam_a * eps * (log_a - _lse((g[None, :] - C_eff) / eps, axis=1))
        g = lam_b * eps * (log_b - _lse((f[:, None] - C_eff) / eps, axis=0))
        err = float(np.max(np.abs(f - f_prev)))
        # 非平衡情形 f/g 不会收敛到不动点差值，用相对变化判断
        if err < tol * max(1.0, float(np.max(np.abs(f)))):
            break

    logP = (f[:, None] + g[None, :] - C_eff) / eps
    P = np.exp(logP)
    P = np.where(np.isfinite(C), P, 0.0)

    if return_log:
        return P, {"f": f, "g": g, "iters": _, "err": err,
                   "lam_a": lam_a, "lam_b": lam_b}
    return P
