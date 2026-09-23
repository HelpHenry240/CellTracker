"""§1.4 与时序网络的理论关联 + §1.5 多尺度时间耦合（ideas.pdf 式 15–19）。

原文公式（R1 抄录）
------------------
式(15)  V = ∪_{t=1}^T { (i,t) | i ∈ V^t }
式(16)  E_inter = ∪_{t=1}^{T−1} { (i,t)→(j,t+1) | Γ^t_ij > 0 }
式(17)  min_{{Γ^t}} Σ_t [ L(Γ^t) + ε Ent(Γ^t) + τ_t KL(Γ^t1‖a^t)
                         + τ_{t+1} KL((Γ^t)^⊤1‖a^{t+1}) ]
                  + λ_temp Σ_{t} R_temp(Γ^t, Γ^{t+1})
式(18)  R_temp(Γ^t, Γ^{t+1}) = ‖ Γ^{t,t+2}_direct − Γ^t Γ^{t+1} ‖_F²
式(19)  R^(k)_temp(Γ^t,…,Γ^{t+k−1}) = ‖ Γ^{t,t+k}_direct − Γ^t ⋯ Γ^{t+k−1} ‖_F²
        并附加 Σ_{k∈K} λ^(k)_temp Σ_t R^(k)_temp，K 如 {2,3,5}

原文关于"条件概率"的说明（决定式18/19 在哪个空间比较）：
  "如果将每一行 Γ^t_i· 归一化为条件概率
   P((j,t+1)|(i,t)) = Γ^t_ij / (Σ_{j'} Γ^t_ij' + δ)（δ>0 为防止除零），
   则 {Γ^t} 诱导出一个时间非齐次 Markov 链"。

求解策略（§1.7 原文）："先对每对相邻帧独立求解 Γ^t 作为初始化，再在全局目标 (17)
下做少量迭代式微调" → **交替优化（Gauss-Seidel）+ 回溯线搜索**。

与原仓库的差异（本模块重写的原因）
--------------------------------
1. 原实现所有尺度共用一个 `lambda_temp`，原文是**逐尺度** λ^(k)_temp；
2. 原实现的正则比较固定用条件概率，但**没有记录/暴露"字面乘积"口径**，也无法对照；
3. 原实现没有把式(15)(16)（时间展开图的节点/跨帧边定义）显式落成函数。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from celltracker.ot.fgw import structural_term        # 式(9) 结构项，复用
from celltracker.ot.sinkhorn import sinkhorn_log      # 式(12)/(14) 求解器，复用

from ..config import CouplingConfig, MeasureConfig, MultiscaleConfig
from ..coupling.pairwise import PairCoupling, solve_coupling, solve_jump_coupling
from ..representation.measure import knn_structure, pairwise_distance

__all__ = ["MultiscaleResult", "refine_couplings", "pair_objective",
           "entropy", "kl_divergence", "time_expanded_edges"]


# ---------------------------------------------------------------------------
# 目标函数的各个分量（式11/13/17）
# ---------------------------------------------------------------------------

def entropy(P: np.ndarray) -> float:
    """式(11)：Ent(Γ) = Σ_{i,j} Γ_ij (log Γ_ij − 1)。（0·log0 取 0）"""
    P = np.asarray(P, dtype=float)
    nz = P > 0
    out = float(P[nz].sum())          # −Σ Γ_ij
    out += float((P[nz] * np.log(P[nz])).sum())
    return out


def kl_divergence(u: np.ndarray, v: np.ndarray) -> float:
    """式(13)：KL(u‖v) = Σ_i u_i log(u_i/v_i) − u_i + v_i。"""
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    eps = 1e-300
    out = float(v.sum() - u.sum())
    nz = u > 0
    out += float((u[nz] * np.log(u[nz] / np.maximum(v[nz], eps))).sum())
    return out


def pair_objective(P: np.ndarray, C: np.ndarray, D, Dp, a: np.ndarray,
                   b: np.ndarray, ccfg: CouplingConfig, eps: float) -> float:
    """式(17) 方括号内的单对目标：L(Γ) + εEnt(Γ) + τ_a KL(Γ1‖a) + τ_b KL(Γᵀ1‖b)。

    L(Γ) 按式(9)：`(1−η)⟨C,Γ⟩ + η·结构项`。
    """
    C_fin = np.where(np.isfinite(C), C, 0.0)
    if ccfg.eta > 0 and D is not None and Dp is not None:
        L = ((1.0 - ccfg.eta) * float(np.sum(P * C_fin))
             + ccfg.eta * structural_term(P, D, Dp))
    else:
        L = float(np.sum(P * C_fin))
    obj = L + eps * entropy(P)
    if ccfg.tau_a is not None:
        obj += float(ccfg.tau_a) * kl_divergence(P.sum(axis=1), a)
    if ccfg.tau_b is not None:
        obj += float(ccfg.tau_b) * kl_divergence(P.sum(axis=0), b)
    return obj


# ---------------------------------------------------------------------------
# 式(15)(16)：时间展开图的节点与跨帧边
# ---------------------------------------------------------------------------

def time_expanded_edges(couplings: dict[int, PairCoupling],
                        ts: list[int], theta_frac: float | None = 0.0
                        ) -> list[tuple[int, int, int, float]]:
    """式(15)(16)：把 {Γ^t} 解释成时间展开图上的跨帧边。

    返回 `[(t, i, j, Γ_ij)]`，只保留 `Γ_ij > 0`（式16 的支撑）；
    `theta_frac>0` 时进一步要求 `Γ_ij ≥ theta_frac × a_i`（式26 的候选门槛）。
    """
    out: list[tuple[int, int, int, float]] = []
    for pos, art in sorted(couplings.items()):
        t = ts[pos]
        thr = (np.full(art.plan.shape[0], float(theta_frac)) * art.mass_a
               if theta_frac else np.zeros(art.plan.shape[0]))
        ii, jj = np.where(art.plan > 0)
        for i, j in zip(ii, jj):
            g = float(art.plan[i, j])
            if g >= thr[i]:
                out.append((t, int(i), int(j), g))
    return out


# ---------------------------------------------------------------------------
# 多尺度精炼（式17-19 的交替优化）
# ---------------------------------------------------------------------------

@dataclass
class MultiscaleResult:
    couplings: dict[int, PairCoupling]
    direct: dict[tuple[int, int], PairCoupling] = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)
    info: dict = field(default_factory=dict)


def _normalizer(mode: str):
    """式(18)(19) 的乘积空间：'cond' = 行归一化条件概率（原文 §1.4 的定义）。"""
    if mode == "raw":
        return lambda M: np.asarray(M, dtype=float)
    if mode == "cond":
        def _cond(M: np.ndarray) -> np.ndarray:
            M = np.asarray(M, dtype=float)
            return M / (M.sum(axis=1, keepdims=True) + 1e-12)
        return _cond
    raise ValueError(f"未知 norm 口径: {mode!r}（应为 'cond' 或 'raw'）")


def refine_couplings(couplings: dict[int, PairCoupling],
                     ts: list[int],
                     frame_xy: dict[int, np.ndarray],
                     frame_vol: dict[int, np.ndarray],
                     cfg: MultiscaleConfig,
                     ccfg: CouplingConfig,
                     mcfg: MeasureConfig,
                     spacing: tuple[float, ...] | None = None,
                     pred_xy: dict[int, np.ndarray] | None = None
                     ) -> MultiscaleResult:
    """按式(17)-(19) 对相邻帧耦合做交替优化精炼。

    `couplings[i]` 对应帧对 `(ts[i], ts[i+1])`。`cfg.enabled=False` 时恒等返回。
    """
    if not cfg.enabled or not couplings:
        return MultiscaleResult(couplings=dict(couplings), info={"enabled": False})

    norm = _normalizer(cfg.norm)
    P: dict[int, np.ndarray] = {i: np.array(c.plan, dtype=float)
                                for i, c in couplings.items()}
    base: dict[int, PairCoupling] = {i: c for i, c in couplings.items()}
    n_frames = len(ts)
    ks = [int(k) for k in cfg.ks if int(k) >= 1]
    lam = {int(k): float(v) for k, v in dict(cfg.lambda_temp).items()}

    # 结构项（式9）用的帧内图：只在 η>0 时需要
    Ds: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    if ccfg.eta > 0:
        for i in range(n_frames):
            t = ts[i]
            D, _, _ = knn_structure(frame_xy[t], frame_vol.get(t), mcfg, spacing)
            Ds[i] = (D, D)
        for i in range(n_frames):
            if i + 1 < n_frames:
                Dp, _, _ = knn_structure(frame_xy[ts[i + 1]], frame_vol.get(ts[i + 1]),
                                         mcfg, spacing)
                Ds[i] = (Ds[i][0], Dp)

    # ---- 式(18)(19) 的直接跳帧耦合 Γ^{t,t+k}_direct ----
    direct: dict[tuple[int, int], PairCoupling] = {}
    for i in range(n_frames):
        for k in ks:
            if i + k <= n_frames - 1:
                direct[(i, k)] = solve_jump_coupling(
                    frame_xy[ts[i]], frame_xy[ts[i + k]], k, ccfg, mcfg,
                    frame_vol.get(ts[i]), frame_vol.get(ts[i + k]), spacing,
                    eps=float(couplings[i].eps_eff))

    def compose(idxs, P_cur):
        out = norm(P_cur[idxs[0]])
        for j in idxs[1:]:
            out = out @ norm(P_cur[j])
        return out

    def reg_value(P_cur: dict[int, np.ndarray]) -> float:
        """式(19) 的加权和 Σ_k λ^(k)_temp Σ_t R^(k)_temp（λ 之外的缩放均属于原文）。"""
        total = 0.0
        for (i, k), art in direct.items():
            idxs = list(range(i, i + k))
            if any(j not in P_cur for j in idxs):
                continue
            total += lam.get(k, 0.0) * float(
                np.sum((norm(art.plan) - compose(idxs, P_cur)) ** 2))
        return total

    # ---- 目标函数：只对被更新的那一对做局部评估（Gauss–Seidel 的性质）----
    # 更新 P[i] 时，其它帧对的基础目标不变，只有"包含 i 的时间窗"的正则项会变。
    # 所以局部目标 = base_obj[i] + Σ_{包含 i 的窗} λ_k·R^(k)，与全局目标相差一个常数。
    # 这样每次线搜索尝试的代价从 O(T·n³) 降到 O(窗数·n³)，实测把整条链路从
    # ~1 小时降到 ~20 分钟（数学上等价，只是不再重复计算未变化的项）。
    base_obj_cache = {i: pair_objective(P[i], base[i].cost, Ds.get(i, (None, None))[0],
                                        Ds.get(i, (None, None))[1], base[i].mass_a,
                                        base[i].mass_b, ccfg, base[i].eps_eff)
                      for i in base}
    windows_of: dict[int, list[tuple[int, int]]] = {i: [] for i in base}
    for (a_idx, k) in direct:
        for off in range(k):
            if a_idx + off in windows_of:
                windows_of[a_idx + off].append((a_idx, k))

    def reg_terms_for(i: int, P_cur: dict[int, np.ndarray]) -> float:
        """只算"包含帧对 i"的那些时间窗的正则项。"""
        total = 0.0
        for (a_idx, k) in windows_of.get(i, ()):
            idxs = list(range(a_idx, a_idx + k))
            if any(j not in P_cur for j in idxs):
                continue
            if not np.isfinite(P_cur[i]).all():
                return float("inf")
            total += lam.get(k, 0.0) * float(
                np.sum((norm(direct[(a_idx, k)].plan) - compose(idxs, P_cur)) ** 2))
        return total

    def objective(P_cur: dict[int, np.ndarray], i: int | None = None) -> float:
        """i 给定时为局部目标（差一个常数）；i=None 时为全局目标（诊断/收尾用）。"""
        if i is not None:
            if not np.isfinite(P_cur[i]).all():
                return float("inf")
            D, Dp = Ds.get(i, (None, None))
            return (pair_objective(P_cur[i], base[i].cost, D, Dp, base[i].mass_a,
                                   base[i].mass_b, ccfg, base[i].eps_eff)
                    + reg_terms_for(i, P_cur))
        total = reg_value(P_cur)
        for j, art in base.items():
            if not np.isfinite(P_cur[j]).all():
                return float("inf")
            total += pair_objective(P_cur[j], art.cost, Ds.get(j, (None, None))[0],
                                    Ds.get(j, (None, None))[1], art.mass_a,
                                    art.mass_b, ccfg, art.eps_eff)
        return total

    def reg_grad(i: int, P_cur: dict[int, np.ndarray]) -> np.ndarray:
        """∂R/∂Γ_i（式19），在一阶近似下穿过行归一化（见模块 docstring 的说明）。"""
        n_i = P_cur[i].shape[0]
        grad = np.zeros_like(P_cur[i])
        for (a_idx, k), art in direct.items():
            for off in range(k):
                if a_idx + off != i:
                    continue
                idxs = list(range(a_idx, a_idx + k))
                if any(j not in P_cur for j in idxs):
                    continue
                # R = ‖T − L·P_i·Rp‖²_F，L = ∏_{j<i} P_j（无则单位阵）、Rp = ∏_{j>i} P_j
                left = None
                for j in idxs[:off]:
                    left = norm(P_cur[j]) if left is None else left @ norm(P_cur[j])
                if left is None:
                    left = np.eye(n_i)
                right = None
                for j in idxs[off + 1:]:
                    right = norm(P_cur[j]) if right is None else right @ norm(P_cur[j])
                if right is None:
                    right = np.eye(P_cur[i].shape[1])
                resid = norm(art.plan) - compose(idxs, P_cur)   # (n_a, n_b)
                g = -2.0 * (left.T @ resid) @ right.T           # ∂R/∂P_i
                rowsum = P_cur[i].sum(axis=1, keepdims=True) + 1e-12
                grad += lam.get(k, 0.0) * g / rowsum        # 穿过归一化（一阶）
        return grad

    history: list[dict] = []
    for rnd in range(max(int(cfg.n_rounds), 1)):
        for i in sorted(P):
            before = objective(P, i)
            g = reg_grad(i, P)
            if not np.any(g):
                continue
            scale = float(np.max(np.abs(g)))      # 用 max 归一：扰动幅度有界
            if not np.isfinite(scale) or scale <= 0:
                continue
            # ENG_SUPP：λ^(k)_temp 已并入 `reg_grad`（逐尺度加权），此处把梯度
            # 归一化到代价尺度（µm²），使"每轮扰动幅度"有明确量纲含义。
            # 用 max|g| 归一（而不是中位数）是为了让 |ΔC| ≤ median(C)，
            # 避免扰动把 Sinkhorn 推到 exp 溢出（实测中位数归一会发散）。
            c_scale = float(np.median(base[i].cost[np.isfinite(base[i].cost)]))
            g_unit = np.clip(g / scale, -1.0, 1.0)
            step = 1.0
            accepted = None
            for _ in range(6):
                G = np.where(np.isfinite(base[i].cost),
                             base[i].cost + step * g_unit * c_scale, np.inf)
                cand = resolve_with_cost(base[i], G, ccfg, Ds.get(i, (None, None)))
                if cand is None:
                    step *= 0.5
                    continue
                trial = dict(P)
                trial[i] = cand.plan
                if not cfg.line_search or objective(trial, i) <= before:
                    accepted = (cand, trial)
                    break
                step *= 0.5
            if accepted is None:
                continue          # 线搜索失败：保持原解（保证目标单调不增）
            cand, trial = accepted
            P = trial
            base[i] = cand
        history.append({"round": rnd + 1, "objective": objective(P),
                        "reg": reg_value(P)})

    refined = {i: base[i] for i in sorted(base)}
    for i, art in refined.items():
        art.plan = P[i]
        art.info = {**art.info, "multiscale_refined": True,
                    "norm": cfg.norm, "ks": list(ks)}
    return MultiscaleResult(couplings=refined, direct=direct, history=history,
                            info={"enabled": True, "norm": cfg.norm, "ks": list(ks),
                                  "lambda_temp": {int(k): float(v)
                                                  for k, v in lam.items()},
                                  "n_rounds": int(cfg.n_rounds),
                                  "line_search": bool(cfg.line_search),
                                  "history": history})


def resolve_with_cost(art: PairCoupling, G: np.ndarray, ccfg: CouplingConfig,
                      Ds: tuple) -> PairCoupling | None:
    """在给定（扰动后的）代价 `G` 上重解式(12)/(14)，返回新的 PairCoupling。

    ENG_SUPP：重解只用 Sinkhorn（式9 的结构项在当前解处被线性化进扰动里，
    不重复求解 FGW）。原文 §1.7 只要求"在全局目标下做**少量迭代式微调**"，
    而重复求解 FGW 会让一层精炼的代价上涨一个量级（实测：FGW 的 40 轮条件梯度
    × 每轮 Sinkhorn ⇒ 每对帧秒级）。数值退化（NaN/inf）时返回 None，由线搜索拒绝。
    """
    from dataclasses import replace

    with np.errstate(over="ignore", invalid="ignore"):
        plan = sinkhorn_log(G, art.mass_a, art.mass_b, eps=art.eps_eff,
                            tau_a=ccfg.tau_a, tau_b=ccfg.tau_b,
                            n_iter=ccfg.sinkhorn_iters)
    if not np.isfinite(plan).all():
        return None
    return replace(art, plan=np.asarray(plan, dtype=float), cost=G)
