"""§1.6 由传输计划重建细胞轨迹（ideas.pdf 式 23–24）+ §2.0.1 的边决策。

原文（R1 抄录）
--------------
式(23)  j*(i,t) = arg max_{1≤j≤n_{t+1}} Γ^t_ij
式(24)  接受关联 c_i^t → c_{j*}^{t+1}，当且仅当
          Γ^t_{i,j*} ≥ θ_Γ  且  C_feat_{i,j*} ≤ θ_C
补充判据（原文紧接式24 的三条）：
  • 死亡：行和 m^out_i = Σ_j Γ^t_ij 低于 η_death → 帧 t 为轨迹终止点
  • 出生：列和 m^in_j = Σ_i Γ^t_ij 低于 η_birth → 帧 t+1 为轨迹起点
  • 分裂：存在 j1,j2 使 Γ^t_{ij1}, Γ^t_{ij2} 均超过一定比例阈值，
    且 s^{t+1}_{j1} + s^{t+1}_{j2} ≈ s^t_i（给定容差内）

跨帧桥接（§2.0.1，原仓库缺失）
-----------------------------
原文："**若在 OT构图或后续轨迹拼接中允许跨帧候选**，例如从 (i,t−1) 直接连到
(k,t+1)…对分割空洞进行桥接，保持轨迹在时间上的连续性。"
→ 两条决策路径都实现：当某源细胞在 t+1 **没有可接受后继**时，用同一套式(23)(24)
  在跳帧耦合 Γ^{t,t+2}_direct 上寻找后继，轨迹 id 跨过空洞延续。

结构约束（原文 §2.0.1："每个节点最多一个父节点、有限个子节点"）
------------------------------------------------------------
每个节点 ≤1 父、每个源 ≤ `max_children` 个子；分裂后父轨迹终止。

与原仓库的差异
------------
1. 式(24)/(26) 的 θ_Γ 用**原始 Γ**（原文），原仓库用行归一化质量；
2. η_death / η_birth 原文有、原仓库没有 → 本模块实现；
3. 分裂判据补上**体积守恒**项（原文"且 s_j1+s_j2 ≈ s_i"），原仓库只用了质量占比；
4. 桥接边两条路径都有实现（原仓库完全没有）。
"""

from __future__ import annotations

import numpy as np

from celltracker.data.ctc import Track
from celltracker.track.base import Detections, TrackResult

from .config import ReconstructConfig
from .coupling import PairCoupling
from .tracks import normalize_tracks

__all__ = ["reconstruct_from_ot", "reconstruct_from_edges", "volume_consistent"]


def volume_consistent(s_parent: float, s_children: list[float], tol: float) -> bool:
    """原文分裂判据的第二半句：`s_j1 + s_j2 ≈ s_i`（给定容差内）。"""
    if not s_children or s_parent <= 0:
        return False
    return abs(float(np.sum(s_children)) - float(s_parent)) <= tol * float(s_parent)


def _threshold(frac: float | None, absolute: float | None, mass: np.ndarray) -> np.ndarray:
    """θ_Γ（式24/26）：显式绝对值优先；否则按"占源质量的比"标定（CALIBRATED）。"""
    if absolute is not None:
        return np.full(mass.shape[0], float(absolute))
    f = 0.05 if frac is None else float(frac)
    return f * np.maximum(mass, 1e-12)


def _theta_c(cfg: ReconstructConfig, r_max: float) -> float:
    return float(cfg.theta_c) if cfg.theta_c is not None else float(r_max) ** 2


# ---------------------------------------------------------------------------
# 路径 A：§1.6 的 OT 规则重建（式23/24 + 死亡/出生/分裂）
# ---------------------------------------------------------------------------

def reconstruct_from_ot(dets: Detections, couplings: dict[int, PairCoupling],
                        cfg: ReconstructConfig, r_max: float,
                        jump: dict[int, PairCoupling] | None = None,
                        bridge: bool = True) -> TrackResult:
    """`couplings[pos]` 对应帧对 `(ts[pos], ts[pos+1])`（与 `time_expanded_edges` 同键）。

    `jump[t]`：帧 t 与 t+2 之间的直接耦合（式18/19）；用于 §2.0.1 的跨空洞桥接。
    """
    ts = dets.t_range
    res = TrackResult(meta={"decision": "ot_rule"})
    assignment: dict[int, np.ndarray] = {}
    pending: dict[int, dict[int, int]] = {}      # 目标帧 → {检测下标: 轨迹 id}
    next_id = 1
    stats = {"n_death": 0, "n_birth": 0, "n_division": 0,
             "n_division_rejected_volume": 0, "n_link_accept": 0,
             "n_link_rejected_theta": 0, "n_link_rejected_cost": 0,
             "n_bridge_accept": 0, "n_bridge_rejected": 0}

    first = ts[0]
    ids0 = np.arange(next_id, next_id + dets.n(first))
    next_id += dets.n(first)
    assignment[first] = ids0
    for tid in ids0:
        res.tracks[int(tid)] = Track(int(tid), first, first, 0)

    for pos, (t, t_next) in enumerate(zip(ts[:-1], ts[1:])):
        art = couplings.get(pos)
        n_dst = dets.n(t_next)
        src_ids = assignment[t]
        s_src = dets.volume(t)
        s_dst = dets.volume(t_next)
        out_ids = np.zeros(n_dst, dtype=np.int64)
        claimed: set[int] = set()
        divided: set[int] = set()
        linked: set[int] = set()          # 已获得直接后继的源下标
        # §2.0.1 桥接的前置条件：该源在 t+1 **连一个 R_max 内的候选都没有**（节点缺失）
        no_direct_rows = (np.ones(len(src_ids), dtype=bool) if art is None
                          else ~np.isfinite(art.cost).any(axis=1))

        # ---- (0) 先落地上一步登记的桥接（此时它已占用目标节点）----
        for j, tid in pending.pop(t_next, {}).items():
            if j < n_dst:
                out_ids[j] = tid
                claimed.add(int(j))

        if art is not None and n_dst > 0:
            plan, cost = art.plan, art.cost
            thr = _threshold(cfg.theta_gamma_frac, cfg.theta_gamma, art.mass_a)
            theta_c = _theta_c(cfg, r_max)

            # ---- (1) 分裂：一行对多个目标分配显著质量，且体积守恒（原文）----
            for i in range(len(src_ids)):
                g = plan[i]
                if g.size < 2:
                    continue
                # 原文："存在 j1,j2 使 Γ^t_ij1, Γ^t_ij2 均超过**一定比例阈值**"。
                # 这里的"比例"是**行内占比** Γ_ij / Σ_j' Γ_ij'（与决策 0001 记的
                # "行内质量分配判据"一致），不是相对 a_i 的绝对量。
                # 实测依据：在预测实例 + 非平衡 OT 下，用相对 a_i 的阈值会因整行质量
                # 流失（分裂父的行和仅 ≈0.08 a_i）而**系统性漏掉全部分裂**（检出 0 个），
                # 改成行内占比后分裂才能被检出，体积守恒项再对候选做二次筛。
                row = float(g.sum())
                if row <= 0:
                    continue
                rn = g / row
                sig = [int(j) for j in np.argsort(-rn) if rn[j] >= cfg.div_ratio]
                if len(sig) < 2:
                    continue
                kids = sig[:cfg.max_children]
                if not volume_consistent(float(s_src[i]),
                                         [float(s_dst[j]) for j in kids], cfg.vol_tol):
                    stats["n_division_rejected_volume"] += 1
                    continue                  # 体积不守恒 → 不判为分裂
                if any(j in claimed for j in kids):
                    continue
                parent = int(src_ids[i])
                for j in kids:
                    child = next_id
                    next_id += 1
                    out_ids[j] = child
                    claimed.add(int(j))
                    res.tracks[child] = Track(child, t_next, t_next, parent)
                divided.add(i)
                stats["n_division"] += 1

            # ---- (2) 死亡判据（式24）：行和低于 η_death → 轨迹在 t 终止 ----
            for i in range(len(src_ids)):
                if i in divided:
                    continue
                if art.row_sum[i] < cfg.eta_death * float(art.mass_a[i]):
                    stats["n_death"] += 1

            # ---- (3) 单目标关联（式23 + 式24）----
            for i in range(len(src_ids)):
                if i in divided or plan[i].size == 0:
                    continue
                j = int(np.argmax(plan[i]))                     # 式(23)
                if j in claimed:
                    continue
                if plan[i, j] < thr[i]:                         # 式(24) 第一半
                    stats["n_link_rejected_theta"] += 1
                    continue
                if not np.isfinite(cost[i, j]) or cost[i, j] > theta_c:  # 式(24) 第二半
                    stats["n_link_rejected_cost"] += 1
                    continue
                out_ids[j] = int(src_ids[i])
                claimed.add(j)
                linked.add(i)
                stats["n_link_accept"] += 1

        _assign_bridges(t=t, t_last=ts[-1], src_ids=src_ids, divided=divided,
                        linked=linked, jump=jump, pending=pending, cfg=cfg,
                        r_max=r_max, bridge=bridge, stats=stats,
                        no_direct=no_direct_rows)

        if art is not None and n_dst > 0:
            # ---- (4) 出生判据（式24）：列和低于 η_birth → t+1 为起点 ----
            for j in range(n_dst):
                if out_ids[j] != 0:
                    continue
                if art.col_sum[j] < cfg.eta_birth * float(art.mass_b[j]):
                    stats["n_birth"] += 1
                out_ids[j] = next_id
                res.tracks[next_id] = Track(next_id, t_next, t_next, 0)
                next_id += 1
        else:
            # 目标帧没有检测：所有目标节点缺失（这正是需要桥接的场景）
            for j in range(n_dst):
                if out_ids[j] == 0:
                    out_ids[j] = next_id
                    res.tracks[next_id] = Track(next_id, t_next, t_next, 0)
                    next_id += 1

        assignment[t_next] = out_ids

    assignment, tracks, info = normalize_tracks(assignment, res.tracks,
                                                hole_policy=cfg.hole_policy)
    res.assignment, res.tracks = assignment, tracks
    res.meta.update({**stats, **info})
    return res


def _assign_bridges(t: int, t_last: int, src_ids: np.ndarray, divided: set,
                    linked: set, jump, pending: dict, cfg: ReconstructConfig,
                    r_max: float, bridge: bool, stats: dict,
                    no_direct: np.ndarray | None = None) -> None:
    """§2.0.1：直接后继缺失时，用跳帧耦合（式18/19）跨过空洞延续轨迹。

    复用式(23)(24) 的判据，只是把 Γ^t 换成 Γ^{t,t+2}_direct；接受后把 t+2 帧的
    目标节点登记为**同一轨迹 id**（该 id 在 t+1 帧形成空洞，由 exporter 处理）。
    """
    if not bridge or not jump:
        return
    art = jump.get(t)
    if art is None or art.plan.shape[0] != len(src_ids) or t + 2 > t_last:
        return
    thr = _threshold(cfg.theta_gamma_frac, cfg.theta_gamma, art.mass_a)
    theta_c = _theta_c(cfg, r_max)
    occupied = pending.setdefault(t + 2, {})
    for i in range(len(src_ids)):
        if i in divided or i in linked or art.plan[i].size == 0:
            continue
        if no_direct is not None and not bool(no_direct[i]):
            # 原文前置条件：只有"帧 t+1 该节点缺失"时才桥接（见 graph.bridge_scope）
            stats["n_bridge_rejected"] += 1
            continue
        if float(art.plan[i].max(initial=0.0)) < thr[i]:
            stats["n_bridge_rejected"] += 1
            continue
        j = int(np.argmax(art.plan[i]))                 # 式(23) 用在跳帧耦合上
        if not np.isfinite(art.cost[i, j]) or art.cost[i, j] > theta_c or j in occupied:
            stats["n_bridge_rejected"] += 1
            continue
        occupied[j] = int(src_ids[i])                    # 同一轨迹 id 延续
        stats["n_bridge_accept"] += 1


# ---------------------------------------------------------------------------
# 路径 B：§2.0.1 的 GNN 边决策（式33）
# ---------------------------------------------------------------------------

def reconstruct_from_edges(dets: Detections, decisions: dict[int, dict],
                           cfg: ReconstructConfig) -> TrackResult:
    """用 GNN 的边分类结果（式33）重建轨迹。

    `decisions[t]` 需包含：
      * `pairs`  (E,2)：`pairs[k,0]` 是源帧 t 的检测下标，`pairs[k,1]` 是目标帧的
      * `score`  (E,)  ŷ_e（式33）
      * `gap`    (E,)  Δt（1 = 相邻帧；2 = §2.0.1 的跨一帧桥接边）

    冲突消解（全局，先于赋值）：
      1. 每个目标节点 ≤1 父：优先 Δt 小（直接边优于桥接边），平局取分数高；
      2. 每个源 ≤ `max_children` 个子；有 ≥2 个直接子节点时按原文的
         **体积守恒**判据决定是否判为分裂，不守恒则只保留分数最高者；
      3. 桥接边只用于"直接后继缺失"的源（原文 §2.0.1 的桥接语义）。
    """
    tau = float(cfg.tau_edge)
    max_children = int(cfg.max_children)
    ts = dets.t_range

    edges: list[tuple[int, int, int, int, float]] = []   # (t, i, t_next, j, score)
    for t, dec in decisions.items():
        if not dec or dec.get("pairs") is None:
            continue
        pairs = np.asarray(dec["pairs"], dtype=np.int64).reshape(-1, 2)
        score = np.asarray(dec["score"], dtype=float).ravel()
        gap = (np.asarray(dec["gap"], dtype=np.int64).ravel()
               if dec.get("gap") is not None else np.ones(len(pairs), dtype=np.int64))
        for (i, j), sc, g in zip(pairs, score, gap):
            if sc >= tau:
                edges.append((int(t), int(i), int(t) + int(g), int(j), float(sc)))

    # ---- (1) 每目标一个父：Δt 小优先，其次分数高 ----
    incoming: dict[tuple[int, int], tuple[int, int, float]] = {}
    for (t, i, tn, j, sc) in edges:
        key = (tn, j)
        cur = incoming.get(key)
        if cur is None or (tn - t) < (tn - cur[1]) or \
                ((tn - t) == (tn - cur[1]) and sc > cur[2]):
            incoming[key] = (i, t, sc)

    # ---- (2)(3) 每源的子节点、分裂判据、桥接语义 ----
    children: dict[tuple[int, int], list[tuple[int, int, float]]] = {}
    for (tn, j), (i, t, sc) in incoming.items():
        children.setdefault((t, i), []).append((tn, j, sc))
    kept: set[tuple[int, int, int, int]] = set()
    stats = {"n_division": 0, "n_division_rejected_volume": 0,
             "n_bridge_used": 0, "n_bridge_dropped": 0, "n_edges_accepted": 0}
    for (t, i), kids in children.items():
        kids = sorted(kids, key=lambda x: (x[0] - t, -x[2]))
        direct = [k for k in kids if k[0] - t == 1]
        bridges = [k for k in kids if k[0] - t > 1]
        s_src = float(dets.volume(t)[i]) if dets.n(t) else 0.0
        if len(direct) >= 2:
            top = direct[:max_children]
            if volume_consistent(s_src, [float(dets.volume(tn)[j])
                                         for tn, j, _ in top], cfg.vol_tol):
                stats["n_division"] += 1
            else:
                stats["n_division_rejected_volume"] += 1
                top = direct[:1]
            for tn, j, _ in top:
                kept.add((t, i, tn, j))
        elif direct:
            tn, j, _ = direct[0]
            kept.add((t, i, tn, j))
        elif bridges:
            tn, j, _ = bridges[0]
            kept.add((t, i, tn, j))
            stats["n_bridge_used"] += 1
        if direct and bridges:
            stats["n_bridge_dropped"] += len(bridges)
    stats["n_edges_accepted"] = len(kept)

    # ---- 赋值：逐帧推进（继承 id / 分裂出子 id / 新出生）----
    result = TrackResult(meta={"decision": "gnn_edge"})
    assignment: dict[int, np.ndarray] = {t: np.zeros(dets.n(t), dtype=np.int64)
                                         for t in ts}
    merged: dict[int, dict[int, list[tuple[int, int]]]] = {}
    for (t, i, tn, j) in kept:
        merged.setdefault(t, {}).setdefault(i, []).append((tn, j))
    next_id = 1
    for pos, t in enumerate(ts):
        arr = assignment[t]
        for k in range(dets.n(t)):
            if arr[k] == 0:
                arr[k] = next_id                        # 未被继承 → 新轨迹（出生）
                result.tracks[next_id] = Track(next_id, t, t, 0)
                next_id += 1
        for i, targets in sorted(merged.get(t, {}).items()):
            parent = int(arr[i])
            if len(targets) >= 2:                       # 分裂：子轨迹取新 id
                for tn, j in targets[:max_children]:
                    if assignment[tn][j] != 0:
                        continue
                    assignment[tn][j] = next_id
                    result.tracks[next_id] = Track(next_id, tn, tn, parent)
                    next_id += 1
            else:
                tn, j = targets[0]
                if assignment[tn][j] == 0:              # 目标未被占用才继承（≤1 父）
                    assignment[tn][j] = parent

    assignment, tracks, info = normalize_tracks(assignment, result.tracks,
                                                hole_policy=cfg.hole_policy)
    result.assignment, result.tracks = assignment, tracks
    result.meta.update({**stats, **info})
    return result
