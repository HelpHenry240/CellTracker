"""多尺度时间一致性与全局目标（ideas.pdf 式 17–19）。

基础目标（逐对独立求解）::

    sum_t [ L(P_t) + eps Ent(P_t) + tau_a KL(P_t 1 | a_t) + tau_b KL(P_t^T 1 | a_{t+1}) ]

加上多尺度时间正则::

    R^(k)(P_t..P_{t+k-1}) = || D_{t,t+k} - P_t P_{t+1} ... P_{t+k-1} ||_F^2

其中 `D_{t,t+k}` 是帧 t 与 t+k 之间**直接**求解（可在空间上粗化/下采样）的 OT 耦合。
本模块提供：
  - `direct_jump_coupling`：跳帧直接 OT
  - `temporal_regularizer`：多步复合与直接耦合的不一致度
  - `multiscale_refine`：交替优化（先逐对求解，再按时间正则梯度微调）
"""

from __future__ import annotations

import numpy as np

from ..cost.features import CostConfig, build_cost, masses
from .sinkhorn import sinkhorn_log

__all__ = ["direct_jump_coupling", "temporal_regularizer", "multiscale_refine"]


def direct_jump_coupling(src_xy: np.ndarray, dst_xy: np.ndarray, gap: int,
                         cfg: CostConfig, eps: float, tau_a: float | None = None,
                         tau_b: float | None = None, subsample: int = 0,
                         rng: np.random.Generator | None = None) -> np.ndarray:
    """帧 t 与 t+gap 之间的直接 OT 耦合（可选随机下采样以控制复杂度）。"""
    a = masses(None, src_xy.shape[0], "uniform")
    b = masses(None, dst_xy.shape[0], "uniform")
    cfg_jump = CostConfig(**{**cfg.__dict__, "r_max": cfg.r_max * gap})
    C, _ = build_cost(src_xy, dst_xy, None, None, None, cfg_jump)
    return sinkhorn_log(C, a, b, eps=eps, tau_a=tau_a, tau_b=tau_b)


def temporal_regularizer(P_list: list[np.ndarray], t: int, k: int,
                         D_direct: np.ndarray) -> float:
    """式 (19)：`R^(k) = ||D_{t,t+k} - P_t···P_{t+k-1}||_F^2`。"""
    if k < 1:
        raise ValueError("k 必须 >= 1")
    prod = P_list[t]
    for i in range(1, k):
        prod = prod @ P_list[t + i]
    return float(np.sum((D_direct - prod) ** 2))


def multiscale_refine(dets, pair_cfg, gap_scales=(2, 3), lambdas: dict[int, float] | None = None,
                      n_rounds: int = 2, eps: float = 1.0,
                      tau_a: float | None = None, tau_b: float | None = None,
                      verbose: bool = False) -> dict:
    """交替优化：逐对求解 → 用多尺度时间正则的梯度微调。

    返回 {"P": {t: P_t}, "history": [...], "reg": {...}}。
    """
    ts = dets.t_range
    n_frames = len(ts)
    lambdas = lambdas or {k: 0.5 for k in gap_scales}
    # P[i] = 帧 ts[i] 与 ts[i+1] 之间的耦合，i = 0..n_frames-2
    P: dict[int, np.ndarray] = {}

    # --- 初始化：逐对（平衡）OT ---
    cost_cache: dict[int, np.ndarray] = {}
    for i in range(n_frames - 1):
        C, _ = build_cost(dets.centroid(ts[i]), dets.centroid(ts[i + 1]),
                          None, None, None, pair_cfg)
        cost_cache[i] = C
        a = masses(None, dets.n(ts[i]), pair_cfg.mass_mode)
        b = masses(None, dets.n(ts[i + 1]), pair_cfg.mass_mode)
        P[i] = sinkhorn_log(C, a, b, eps=eps, tau_a=tau_a, tau_b=tau_b)

    # 跳帧直接耦合（每个尺度预计算一次）
    direct: dict[tuple[int, int], np.ndarray] = {}
    for i in range(n_frames):
        for k in gap_scales:
            if i + k <= n_frames - 1:
                direct[(i, k)] = direct_jump_coupling(
                    dets.centroid(ts[i]), dets.centroid(ts[i + k]), k, pair_cfg, eps,
                    tau_a, tau_b)

    def _prod(idxs) -> np.ndarray:
        out = P[idxs[0]]
        for i in idxs[1:]:
            out = out @ P[i]
        return out

    history = []
    for _round in range(n_rounds):
        reg_total = 0.0
        for i in range(n_frames - 1):
            C = cost_cache[i]
            G = np.where(np.isfinite(C), C, 0.0)
            # --- P[i] 作为乘积首因子的项：区间 (i, k) ---
            for k, lam in lambdas.items():
                if (i, k) not in direct:
                    continue
                tail = np.eye(P[i].shape[1]) if k == 1 else _prod(range(i + 1, i + k))
                D = direct[(i, k)]
                diff = (P[i] @ tail) - D
                G = G + lam * 2.0 * (diff @ tail.T)
                reg_total += lam * float(np.sum(diff ** 2))
            # --- P[i] 作为乘积中间/末尾因子：区间 (j, k)，j < i <= j+k-1 ---
            for k, lam in lambdas.items():
                for j in range(max(0, i - k + 1), i):
                    if (j, k) not in direct:
                        continue
                    head = np.eye(P[j].shape[0]) if j == i else _prod(range(j, i))
                    tail_idxs = list(range(i + 1, j + k))
                    tail = np.eye(P[i].shape[1]) if not tail_idxs else _prod(tail_idxs)
                    D = direct[(j, k)]
                    diff = (head @ P[i] @ tail) - D
                    G = G + lam * 2.0 * (head.T @ diff @ tail.T)
                    reg_total += lam * float(np.sum(diff ** 2))
            G = np.where(np.isfinite(C), G, np.inf)
            a = masses(None, dets.n(ts[i]), pair_cfg.mass_mode)
            b = masses(None, dets.n(ts[i + 1]), pair_cfg.mass_mode)
            P[i] = sinkhorn_log(G, a, b, eps=eps, tau_a=tau_a, tau_b=tau_b)
        history.append(reg_total)
        if verbose:
            print(f"  round {_round}: temporal regularizer = {reg_total:.6f}")

    return {"P": P, "history": history, "direct": direct}
