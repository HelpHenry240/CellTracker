"""§1.5 末段 + §1.6：短时窗 tracklet 与第二层 OT（coarse-graining）。

原文（R1 抄录，§1.5 末段"短轨迹片段与第二层 OT"）
------------------------------------------------
1. "在一个较短的滑动时间窗口（如 3–5 帧）内，采用上述 OT 框架（含 FGW 结构项与
   运动先验）求解 Γ^t，并通过**较严格的阈值**筛选，只保留置信度较高的跨帧关联，
   形成局部 tracklet"；
2. "将每个 tracklet 视作'超级节点'，为其定义汇总的特征（例如 tracklet 中所有
   细胞位置的均值轨迹、平均速度、整体尺寸变化等），**在更长时间范围内**构建
   tracklet 级别的测度与代价，再通过第二层 OT 或最小代价流将这些 tracklet
   串联起来"。

实现对应
--------
* 第一层：决策阶段（§1.6 式23/24 或 §2.0.1 式33）已只接受高置信边 → 其输出的
  轨迹就是"只保留高置信关联"的局部片段。本模块再把长轨迹**按窗口长度切分**，
  使长程关联必须由第二层决定（原文："第一层在高时间分辨率下解决局部匹配问题，
  第二层在低时间分辨率下解决长程关联问题"）。窗口长度 `window` ∈ {3,4,5}。
* 超级节点特征：复用原仓库 `celltracker.track.tracklets.tracklet_stats`
  （出现帧、首末位置、平均速度、时长），与原文列举的特征一一对应。
* 测度（CALIBRATED，原文未给公式）：质量 ∝ 时长（片段越长越可信）。
* 代价：把式(20)(21)(22) 用在超级节点上：
    d1 = ‖首(B) − 末(A)‖ ,  d2 = ‖首(B) − (末(A) + v_A·gap)‖
    cost = α·d1² + α′·d2² ，门限 ‖·‖ ≤ R_max·gap（与式8 的位移门限同尺度）
* 第二层求解：式(14) 的**非平衡** OT（找不到后继即允许质量流失 = 轨迹终止）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from celltracker.data.ctc import Track
from celltracker.ot.sinkhorn import sinkhorn_log
from celltracker.track.base import Detections, TrackResult
from celltracker.track.tracklets import tracklet_stats   # 复用：超级节点汇总特征

from .config import CouplingConfig, MeasureConfig, TrackletConfig
from .tracks import normalize_tracks

__all__ = ["TrackletResult", "link_tracklets", "cut_pieces"]


@dataclass
class _Piece:
    pid: int
    orig: int
    frames: list[int]
    cents: list[np.ndarray]
    parent: int = 0
    terminal_by_division: bool = False

    @property
    def begin(self) -> int:
        return self.frames[0]

    @property
    def end(self) -> int:
        return self.frames[-1]

    @property
    def start_xy(self) -> np.ndarray:
        return self.cents[0]

    @property
    def end_xy(self) -> np.ndarray:
        return self.cents[-1]

    @property
    def velocity(self) -> np.ndarray:
        if len(self.cents) >= 2:
            return self.cents[-1] - self.cents[-2]
        return np.zeros_like(self.cents[0])


@dataclass
class TrackletResult:
    result: TrackResult
    table: dict = field(default_factory=dict)
    coupling: np.ndarray | None = None
    info: dict = field(default_factory=dict)


def cut_pieces(result: TrackResult, dets: Detections, window: int
               ) -> tuple[list[_Piece], dict[int, dict[int, int]]]:
    """把每条第一层轨迹切成 ≤ `window` 帧的局部片段。

    返回 `(pieces, piece_of)`，其中 `piece_of[t][det_index] = pid`。
    """
    has_children = {int(tr.parent) for tr in result.tracks.values() if tr.parent}
    pieces: list[_Piece] = []
    piece_of: dict[int, dict[int, int]] = {}
    w = max(int(window), 1)

    nodes_of: dict[int, list[tuple[int, int]]] = {}
    for t in sorted(result.assignment):
        for k, tid in enumerate(result.assignment[t]):
            nodes_of.setdefault(int(tid), []).append((int(t), int(k)))

    for tid in sorted(nodes_of):
        nodes = nodes_of[tid]
        tr = result.tracks.get(tid)
        chunks = [nodes[i:i + w] for i in range(0, len(nodes), w)]
        for ci, chunk in enumerate(chunks):
            pid = len(pieces)
            pieces.append(_Piece(
                pid=pid, orig=tid,
                frames=[t for t, _ in chunk],
                cents=[dets.centroid(t)[k] for t, k in chunk],
                parent=(int(tr.parent) if (tr is not None and ci == 0) else 0),
                terminal_by_division=bool(ci == len(chunks) - 1
                                          and tid in has_children)))
            for t, k in chunk:
                piece_of.setdefault(t, {})[k] = pid
    return pieces, piece_of


def link_tracklets(dets: Detections, result: TrackResult, cfg: TrackletConfig,
                   ccfg: CouplingConfig, mcfg: MeasureConfig,
                   spacing: tuple[float, ...] | None = None) -> TrackletResult:
    """§1.5 末段：第二层（tracklet 级）非平衡 OT 串联。

    `cfg.enabled=False` 时是完全恒等映射（干净消融）。
    """
    if not cfg.enabled:
        return TrackletResult(result=result, info={"enabled": False})

    pieces, piece_of = cut_pieces(result, dets, cfg.window)
    if len(pieces) < 2:
        return TrackletResult(result=result,
                              info={"enabled": True, "n_pieces": len(pieces),
                                    "n_links": 0})

    durations = np.array([len(p.frames) for p in pieces], dtype=float)
    mass = durations / durations.sum()
    n = len(pieces)
    cost = np.full((n, n), np.inf)
    pairs: list[tuple[int, int]] = []
    for ai, A in enumerate(pieces):
        for bi, B in enumerate(pieces):
            if A is B or B.parent or B.begin <= A.end:
                continue
            gap = B.begin - A.end
            if gap < 1 or gap > cfg.max_gap or A.terminal_by_division:
                continue
            d1 = float(np.linalg.norm(B.start_xy - A.end_xy))
            if d1 > ccfg.r_max * gap:
                continue
            # 跨空洞串联的前置条件（同 §2.0.1 的桥接边）：**中间帧该节点确实缺失**。
            # 否则"gap>1 的合并"会凭空造出空洞——实测在真实数据上产生 3487 个
            # 假空洞（中间帧其实有检测），既不符合论文语义，也让 CTC 格式必须靠
            # 补画兜底。判据：把 A→B 的预测位置线性插值到中间帧，若存在 r_max 内的
            # 检测，则该节点没缺，只允许 gap==1 的合并。
            if gap > 1:
                missing = True
                for f in range(A.end + 1, B.begin):
                    frac = (f - A.end) / gap
                    pred = A.end_xy + frac * (B.start_xy - A.end_xy)
                    xy = dets.centroid(f)
                    if xy.size == 0:
                        continue
                    diff = xy - pred[None, :]
                    if spacing is not None:
                        diff = diff * np.asarray(spacing, dtype=float)[None, :]
                    if float(np.min(np.linalg.norm(diff, axis=1))) <= ccfg.r_max:
                        missing = False
                        break
                if not missing:
                    continue
            pred = A.end_xy + A.velocity * gap
            d2 = float(np.linalg.norm(B.start_xy - pred))
            cost[ai, bi] = ccfg.alpha * d1 ** 2 + ccfg.alpha_pred * d2 ** 2
            pairs.append((ai, bi))

    info: dict = {"enabled": True, "n_pieces": n, "n_candidates": len(pairs),
                  "window": int(cfg.window), "tau": cfg.tau}
    if not pairs:
        return TrackletResult(result=result, info={**info, "n_links": 0})

    finite = cost[np.isfinite(cost)]
    eps = (float(ccfg.eps) if ccfg.eps is not None
           else float(ccfg.eps_rel if ccfg.eps_rel is not None else 0.1)
           * float(np.median(finite)))
    coupling = sinkhorn_log(cost, mass, mass, eps=max(eps, 1e-9),
                            tau_a=cfg.tau, tau_b=cfg.tau,
                            n_iter=ccfg.sinkhorn_iters)

    parent_of = list(range(n))          # union-find（时间前向，无环）

    def find(x: int) -> int:
        while parent_of[x] != x:
            parent_of[x] = parent_of[parent_of[x]]
            x = parent_of[x]
        return x

    claimed: set[int] = set()
    links = 0
    for ai in range(n):
        if not np.any(np.isfinite(cost[ai])):
            continue
        for bi in (int(b) for b in np.argsort(-coupling[ai])):
            if bi in claimed or not np.isfinite(cost[ai, bi]):
                continue
            if coupling[ai, bi] < cfg.theta_link * mass[ai]:
                break
            ra, rb = find(ai), find(bi)
            if ra == rb:
                continue
            parent_of[max(ra, rb)] = min(ra, rb)   # 代表 = 时间更早的片段
            claimed.add(bi)
            links += 1
            break

    root_of = {p.pid: find(p.pid) for p in pieces}
    groups: dict[int, list[_Piece]] = {}
    for p in pieces:
        groups.setdefault(root_of[p.pid], []).append(p)
    # 轨迹 id 必须 ≥ 1（CTC 的 0 是背景）：按各组成员的起始帧重新编号
    ordered = sorted(groups, key=lambda g: (min(p.begin for p in groups[g]), g))
    renumber = {g: i + 1 for i, g in enumerate(ordered)}
    # 父子关系要跟着重编号一起搬：父轨迹 id（第一层口径）→ 它所属组的**新** id
    first_piece_of: dict[int, int] = {}
    for p in pieces:
        first_piece_of.setdefault(p.orig, p.pid)
    orig_group = {orig: renumber[root_of[pid]]
                  for orig, pid in first_piece_of.items()}
    new_assignment = {
        t: np.array([renumber[root_of[m[k]]] for k in sorted(m)], dtype=np.int64)
        for t, m in sorted(piece_of.items())}
    new_tracks: dict[int, Track] = {}
    for gid, members in ((renumber[g], groups[g]) for g in ordered):
        members = sorted(members, key=lambda p: p.begin)
        fr = sorted({f for p in members for f in p.frames})
        par_old = int(members[0].parent)
        parent = orig_group.get(par_old, 0) if par_old else 0
        if parent == gid:
            parent = 0                       # 自环保护
        new_tracks[gid] = Track(gid, fr[0], fr[-1], parent)

    assignment, tracks, norm_info = normalize_tracks(
        new_assignment, new_tracks, hole_policy=cfg.hole_policy)
    # 注意：这里必须**合并**上一阶段的 meta（决策阶段的桥接/分裂计数等诊断），
    # 不能让二层 tracklet 的结果把它覆盖掉（曾经丢过 n_bridge_used 等统计）。
    out = TrackResult(assignment=assignment, tracks=tracks,
                      meta={**result.meta,
                            "tracklet": {**info, "n_links": links, **norm_info}})
    return TrackletResult(result=out, table={"n_pieces": n}, coupling=coupling,
                          info={**info, "n_links": links, **norm_info})
