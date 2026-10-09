"""滑动窗口中的高置信 tracklet 与二层 OT（ideas.pdf §1.5、§1.6）。

原文先在 3–5 帧滑动窗口内筛出可靠关联，再以位置、平均速度和尺寸变化
汇总片段，用第二层 OT 关联。这里合并重叠窗口中一致的可靠链；低置信边和
分裂边划定片段边界。粗层沿用式(22)的位置/运动/尺寸项，具体权重和离散
接受比例属于工程参数。只为有时空候选的二部连通分量分配矩阵，避免全序列
片段数的平方内存；分量使用同一全局质量和 ε，等价于屏蔽分量间的运输。
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy.spatial import cKDTree

from celltracker.data.ctc import Track
from celltracker.ot.sinkhorn import sinkhorn_log
from celltracker.track.base import TrackResult
from ..representation.measure import masses
from ..reconstruction.tracks import normalize_tracks


@dataclass
class _Piece:
    pid: int
    orig: int
    frames: list[int]
    cents: list[np.ndarray]
    sizes: list[float] = field(default_factory=list)
    parent: int = 0
    terminal_by_division: bool = False

    @property
    def begin(self):
        return self.frames[0]

    @property
    def end(self):
        return self.frames[-1]

    @property
    def start_xy(self):
        return self.cents[0]

    @property
    def end_xy(self):
        return self.cents[-1]

    @property
    def velocity(self):
        if len(self.frames) < 2:
            return np.zeros_like(self.cents[0])
        return np.mean(np.diff(self.cents, axis=0)/np.diff(self.frames)[:, None], axis=0)

    @property
    def mean_size(self):
        return float(np.mean(self.sizes))

    @property
    def size_rate(self):
        if len(self.frames) < 2:
            return 0.0
        return float(np.mean(np.diff(self.sizes)/np.diff(self.frames)))


@dataclass
class TrackletResult:
    result: TrackResult
    table: dict = field(default_factory=dict)
    coupling: np.ndarray | None = None
    info: dict = field(default_factory=dict)


def cut_pieces(result, dets, window, confidence=0.8):
    """在滑窗中保留同一轨迹的高置信边，合并重叠可靠链。

    无边置信记录的外部轨迹仅将连续的既有硬关联视为可信；正常 pipeline
    使用重建阶段的 accepted_edges。背景编号 0 不生成片段。
    """
    if window < 2:
        raise ValueError('tracklet 窗口至少为两帧')
    score = {(int(t), int(i), int(tn), int(j)): float(p)
             for t, i, tn, j, p in result.meta.get('accepted_edges', [])}
    nodes_of = {}
    for t in sorted(result.assignment):
        for i, tid in enumerate(result.assignment[t]):
            if tid > 0:
                nodes_of.setdefault(int(tid), []).append((t, i))
    children = {int(tr.parent) for tr in result.tracks.values() if tr.parent}
    # 滑窗重叠处采用同一已接受关联；一个检测只归属一条可靠链。
    reliable = set()
    ts = dets.t_range
    for start in ts:
        stop = start + int(window)-1
        for tid, nodes in nodes_of.items():
            for (t, i), (tn, j) in zip(nodes[:-1], nodes[1:]):
                if start <= t < tn <= stop:
                    value = score.get((t, i, tn, j), 1.0 if not score and tn == t+1 else 0.0)
                    if value >= confidence:
                        reliable.add((t, i, tn, j))
    pieces, piece_of = [], {}
    for tid, nodes in sorted(nodes_of.items()):
        chunks, current = [], [nodes[0]]
        for previous, node in zip(nodes[:-1], nodes[1:]):
            if (*previous, *node) in reliable:
                current.append(node)
            else:
                chunks.append(current)
                current = [node]
        chunks.append(current)
        for index, chunk in enumerate(chunks):
            pid = len(pieces)
            pieces.append(_Piece(pid, tid, [t for t, _ in chunk],
                [dets.centroid(t)[i] for t, i in chunk],
                [float(dets.volume(t)[i]) for t, i in chunk],
                int(result.tracks[tid].parent) if index == 0 and tid in result.tracks else 0,
                index == len(chunks)-1 and tid in children))
            for t, i in chunk:
                piece_of.setdefault(t, {})[i] = pid
    return pieces, piece_of


def _candidate_costs(pieces, dets, cfg, ccfg, spacing):
    scale = np.asarray(spacing if spacing is not None else (1, 1, 1), dtype=float)
    volume_scale = float(np.prod(scale))
    sigma_size = ccfg.sigma_s or max(float(np.median([p.mean_size for p in pieces]))*volume_scale, 1e-12)
    starts = {}
    for p in pieces:
        if not p.parent or cfg.merge_division_children:
            starts.setdefault(p.begin, []).append(p.pid)
    trees = {t: cKDTree([pieces[i].start_xy*scale for i in ids]) for t, ids in starts.items()}
    pairs, costs = [], []
    for A in pieces:
        if A.terminal_by_division:
            continue
        for gap in range(1, cfg.max_gap+1):
            frame = A.end+gap
            if frame not in trees:
                continue
            for local in trees[frame].query_ball_point(A.end_xy*scale, ccfg.r_max*gap):
                B = pieces[starts[frame][local]]
                if gap > 1:
                    # 仅跨越真正缺检测的位置，避免把阈值拒绝造成的碎片误当作遮挡。
                    missing = True
                    for f in range(A.end+1, B.begin):
                        if f not in dets.frames:
                            continue
                        expected = A.end_xy+(B.start_xy-A.end_xy)*(f-A.end)/gap
                        distance = np.linalg.norm((dets.centroid(f)-expected)*scale, axis=1)
                        if len(distance) and np.min(distance) <= ccfg.r_max:
                            missing = False
                            break
                    if not missing:
                        continue
                d_current = np.linalg.norm((B.start_xy-A.end_xy)*scale)
                d_predicted = np.linalg.norm((B.start_xy-A.end_xy-A.velocity*gap)*scale)
                expected_size = max(A.mean_size+A.size_rate*gap, 0.0)
                size_delta = (B.mean_size-expected_size)*volume_scale/sigma_size
                pairs.append((A.pid, B.pid))
                costs.append(ccfg.alpha*d_current**2+cfg.velocity_weight*d_predicted**2
                             +cfg.size_weight*ccfg.beta*size_delta**2)
    return np.asarray(pairs, dtype=int).reshape(-1, 2), np.asarray(costs)


def _component_ot(pairs, costs, mass, eps, cfg, iterations):
    """全局禁止分量间传输时，各二部连通分量可以独立求解。"""
    parent = {}
    def find(node):
        parent.setdefault(node, node)
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]
    for a, b in pairs:
        ra, rb = find(('source', int(a))), find(('target', int(b)))
        parent[rb] = ra
    components = {}
    for index, (a, _) in enumerate(pairs):
        components.setdefault(find(('source', int(a))), []).append(index)
    flows = np.zeros(len(pairs))
    largest = 0
    for indices in components.values():
        src = sorted(set(pairs[indices, 0].tolist()))
        dst = sorted(set(pairs[indices, 1].tolist()))
        si, di = {x:i for i,x in enumerate(src)}, {x:i for i,x in enumerate(dst)}
        matrix = np.full((len(src), len(dst)), np.inf)
        rows = [si[int(pairs[k, 0])] for k in indices]
        cols = [di[int(pairs[k, 1])] for k in indices]
        matrix[rows, cols] = costs[indices]
        plan = sinkhorn_log(matrix, mass[src], mass[dst], eps=eps,
                            tau_a=cfg.tau, tau_b=cfg.tau, n_iter=iterations)
        flows[indices] = plan[rows, cols]
        largest = max(largest, matrix.size)
    return flows, len(components), largest


def link_tracklets(dets, result, cfg, ccfg, mcfg, spacing=None):
    if not cfg.enabled:
        return TrackletResult(result, info={'enabled': False})
    pieces, piece_of = cut_pieces(result, dets, cfg.window, cfg.local_confidence)
    info = {'enabled': True, 'n_pieces': len(pieces), 'window': cfg.window,
            'local_confidence': cfg.local_confidence, 'summary': 'mean_velocity_size_and_size_rate'}
    if len(pieces) < 2:
        return TrackletResult(result, info={**info, 'n_links': 0})
    pairs, costs = _candidate_costs(pieces, dets, cfg, ccfg, spacing)
    info['n_candidates'] = len(pairs)
    if not len(pairs):
        # 即使二层没有候选，也保留第一层可靠片段的边界。
        flows = np.empty(0)
        info.update(n_components=0, largest_matrix_entries=0)
    else:
        mass = masses(np.array([p.mean_size for p in pieces]), len(pieces), mcfg.mass_mode)
        positive = costs[costs > 0]
        eps = ccfg.eps if ccfg.eps is not None else (ccfg.eps_rel or 0.1)*(float(np.median(positive)) if len(positive) else 1.0)
        flows, components, largest = _component_ot(pairs, costs, mass, eps, cfg, ccfg.sinkhorn_iters)
        info.update(eps=float(eps), n_components=components, largest_matrix_entries=largest)
    successor, predecessor = {}, {}
    # 一入一出且时间严格前向，构成无环、不重叠的片段链。
    for index in np.argsort(-flows):
        a, b = map(int, pairs[index])
        if a in successor or b in predecessor or flows[index] < cfg.theta_link*mass[a]:
            continue
        successor[a], predecessor[b] = b, a
    group_of, groups = {}, {}
    for p in sorted(pieces, key=lambda p: (p.begin, p.pid)):
        if p.pid in predecessor:
            group = group_of[predecessor[p.pid]]
        else:
            group = len(groups)+1
            groups[group] = []
        group_of[p.pid] = group
        groups[group].append(p)
    # 子片段的 parent 指向原父轨迹的最后片段，而不是第一片段。
    terminal_group = {p.orig: group_of[p.pid] for p in sorted(pieces, key=lambda p:p.end)}
    assignment = {t: np.zeros(dets.n(t), dtype=np.int64) for t in dets.t_range}
    for t, mapping in piece_of.items():
        for i, pid in mapping.items():
            assignment[t][i] = group_of[pid]
    tracks = {}
    for group, members in groups.items():
        first = min(members, key=lambda p:p.begin)
        parent = terminal_group.get(first.parent, 0)
        tracks[group] = Track(group, first.begin, max(p.end for p in members), parent if parent != group else 0)
    assignment, tracks, normalization = normalize_tracks(assignment, tracks, cfg.hole_policy)
    info.update(n_links=len(successor), **normalization)
    output = TrackResult(assignment, tracks, meta={**result.meta, 'tracklet': info})
    return TrackletResult(output, table={'pairs': pairs, 'flow': flows}, info=info)
