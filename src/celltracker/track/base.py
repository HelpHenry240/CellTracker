"""追踪基础数据结构与经典连接器（基线）。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..data.ctc import Track


@dataclass
class Detections:
    """逐帧检测集合。

    `frames[t]` 是字典，至少包含 `centroid (n,d)` 与 `label (n,)`（检测来源的标签，
    用于回填结果掩码）；可选 `volume`、`intensity_mean`。
    """

    frames: dict[int, dict[str, np.ndarray]] = field(default_factory=dict)
    # 序列级元信息（如 `shape`：体数据尺寸，用于节点特征归一化）；
    # `from_h5` 会自动填入，便于下游阶段（如图构建）不需要额外的 h5 句柄。
    meta: dict = field(default_factory=dict)

    @property
    def t_range(self) -> list[int]:
        return sorted(self.frames)

    def n(self, t: int) -> int:
        return int(self.frames[t]["centroid"].shape[0])

    def centroid(self, t: int) -> np.ndarray:
        return np.asarray(self.frames[t]["centroid"], dtype=float)

    def label(self, t: int) -> np.ndarray:
        return np.asarray(self.frames[t]["label"], dtype=np.int64)

    def gt_label(self, t: int) -> np.ndarray:
        """检测对应的 GT 轨迹 id。

        预测检测的 h5 通过 `gt_label` 提供（未匹配到 GT 的检测为 0）；
        普通 GT 的 h5 里「标签即轨迹 id」，故回退到 `label`。
        训练 GNN 时用它生成边标签（同一轨迹 = 移动边，父子关系 = 分裂边）。
        """
        v = self.frames[t].get("gt_label")
        return np.asarray(v, dtype=np.int64) if v is not None else self.label(t)

    def volume(self, t: int) -> np.ndarray:
        v = self.frames[t].get("volume")
        return np.asarray(v, dtype=float) if v is not None else np.ones(self.n(t))

    @classmethod
    def from_h5(cls, h5_path, frames: list[int] | None = None) -> "Detections":
        """从内部 HDF5 读取检测（TRA 金标准即"完美分割"上界设定）。"""
        import h5py

        out: dict[int, dict[str, np.ndarray]] = {}
        meta: dict = {}
        with h5py.File(h5_path, "r") as f:
            if "shape" in f.attrs:
                meta["shape"] = np.asarray([int(x) for x in f.attrs["shape"]], dtype=float)
            meta["name"] = str(f.attrs.get("name", ""))
            meta["seq"] = str(f.attrs.get("seq", ""))
            keys = sorted(f["frames"].keys())
            for key in keys:
                t = int(key)
                if frames is not None and t not in frames:
                    continue
                g = f["frames"][key]
                entry = {
                    "label": np.asarray(g["label"], dtype=np.int64),
                    "centroid": np.asarray(g["centroid"], dtype=float),
                    "volume": np.asarray(g["volume"], dtype=float),
                    "intensity_mean": np.asarray(g["intensity_mean"], dtype=float),
                }
                # 预测检测的 h5 额外带 `gt_label`（每个检测对应的 GT 轨迹 id）
                if "gt_label" in g:
                    entry["gt_label"] = np.asarray(g["gt_label"], dtype=np.int64)
                out[t] = entry
        return cls(out, meta=meta)


@dataclass
class LinkerConfig:
    """连接器超参。"""

    method: str = "hungarian"          # "greedy" 或 "hungarian"
    max_dist: float = 30.0             # 候选门限 R_max（体素）
    use_velocity: bool = False         # 是否用匀速运动先验预测位置
    velocity_weight: float = 1.0       # 预测项权重 α'
    size_weight: float = 0.0           # 尺寸代价权重 β（0 表示关闭）
    size_sigma: float = 1.0            # 尺寸归一化 σ_s
    detect_division: bool = True       # 是否识别二分裂
    division_max_dist: float = 45.0    # 分裂时子细胞离父细胞的允许距离


@dataclass
class TrackResult:
    """追踪结果：逐帧的"检测 → 输出轨迹 id"赋值 + 轨迹表。"""

    assignment: dict[int, np.ndarray] = field(default_factory=dict)  # t -> out id (n,)
    tracks: dict[int, Track] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def n_tracks(self) -> int:
        return len(self.tracks)


def _pair_cost(cfg: LinkerConfig, c_src: np.ndarray, c_dst: np.ndarray,
               v_src: np.ndarray | None, s_src: np.ndarray, s_dst: np.ndarray
               ) -> np.ndarray:
    """代价矩阵：位移(+运动先验) + 尺寸差；门限外用 inf 屏蔽。"""
    d_cur = np.linalg.norm(c_src[:, None, :] - c_dst[None, :, :], axis=-1)
    cost = d_cur
    if cfg.use_velocity and v_src is not None:
        pred = c_src + v_src
        d_pred = np.linalg.norm(pred[:, None, :] - c_dst[None, :, :], axis=-1)
        cost = cost + cfg.velocity_weight * d_pred
    if cfg.size_weight > 0:
        ds = (s_src[:, None] - s_dst[None, :]) / (cfg.size_sigma * max(float(s_dst.mean()), 1e-6))
        cost = cost + cfg.size_weight * ds ** 2
    return np.where(d_cur <= cfg.max_dist, cost, np.inf)


def _match_greedy(cost: np.ndarray) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    used_r, used_c = set(), set()
    order = np.argsort(cost, axis=None)
    n_cols = cost.shape[1]
    for flat in order:
        i, j = divmod(int(flat), n_cols)
        if not np.isfinite(cost[i, j]):
            break
        if i in used_r or j in used_c:
            continue
        used_r.add(i)
        used_c.add(j)
        pairs.append((i, j))
    return pairs


def _match_hungarian(cost: np.ndarray) -> list[tuple[int, int]]:
    from scipy.optimize import linear_sum_assignment

    big = 1e9
    finite = np.where(np.isfinite(cost), cost, big)
    r, c = linear_sum_assignment(finite)
    return [(int(i), int(j)) for i, j in zip(r, c) if np.isfinite(cost[i, j])]


def run_tracking(dets: Detections, cfg: LinkerConfig | None = None) -> TrackResult:
    """经典连接器：贪心最近邻 / 匈牙利 + 可选匀速先验 + 二分裂判定。"""
    cfg = cfg or LinkerConfig()
    ts = dets.t_range
    res = TrackResult(meta={"config": cfg.__dict__})
    if not ts:
        return res

    # 每条轨迹的最近状态：位置、速度、最近一帧
    last_c: dict[int, np.ndarray] = {}
    velocity: dict[int, np.ndarray] = {}
    prev_c: dict[int, np.ndarray] = {}
    next_id = 1
    assignment: dict[int, np.ndarray] = {}

    # 第一帧：全部新建
    first = ts[0]
    ids = np.arange(next_id, next_id + dets.n(first))
    next_id += dets.n(first)
    assignment[first] = ids
    c0 = dets.centroid(first)
    for k, tid in enumerate(ids):
        last_c[int(tid)] = c0[k]
    active = set(int(x) for x in ids)

    for t_prev, t in zip(ts[:-1], ts[1:]):
        src_ids = assignment[t_prev]
        c_src, c_dst = dets.centroid(t_prev), dets.centroid(t)
        s_src, s_dst = dets.volume(t_prev), dets.volume(t)
        v_src = np.array([velocity.get(int(i), np.zeros_like(c_dst[0])) for i in src_ids])
        cost = _pair_cost(cfg, c_src, c_dst, v_src if cfg.use_velocity else None,
                          s_src, s_dst)
        pairs = _match_greedy(cost) if cfg.method == "greedy" else _match_hungarian(cost)

        out_ids = np.zeros(dets.n(t), dtype=np.int64)
        matched_src, matched_dst = set(), set()
        for i, j in pairs:
            tid = int(src_ids[i])
            out_ids[j] = tid
            matched_src.add(i)
            matched_dst.add(j)
            new_c = c_dst[j]
            if tid in last_c:
                velocity[tid] = new_c - last_c[tid]
            last_c[tid] = new_c

        # 未匹配的源检测：轨迹终止（可能因分裂而终结）
        unmatched_src = [i for i in range(dets.n(t_prev)) if i not in matched_src]
        unmatched_dst = [j for j in range(dets.n(t)) if j not in matched_dst]

        # 分裂：一个消失的父 + 两个邻近的新生
        if cfg.detect_division and unmatched_src and len(unmatched_dst) >= 2:
            for i in unmatched_src:
                parent = int(src_ids[i])
                p = c_src[i]
                d = np.linalg.norm(c_dst[unmatched_dst] - p[None, :], axis=1)
                near = [unmatched_dst[k] for k in np.argsort(d) if d[k] <= cfg.division_max_dist]
                if len(near) >= 2:
                    j1, j2 = near[0], near[1]
                    for j in (j1, j2):
                        child = next_id
                        out_ids[j] = child
                        last_c[child] = c_dst[j]
                        velocity[child] = c_dst[j] - p
                        res.tracks[child] = Track(child, t, t, parent)
                        next_id += 1
                    unmatched_dst = [j for j in unmatched_dst if j not in (j1, j2)]

        # 其余未匹配目标：新建轨迹（出生）
        for j in unmatched_dst:
            out_ids[j] = next_id
            last_c[next_id] = c_dst[j]
            res.tracks[next_id] = Track(next_id, t, t, 0)
            next_id += 1

        assignment[t] = out_ids

    assignment, tracks, info = finalize_tracks(assignment, res.tracks)
    res.assignment = assignment
    res.tracks = tracks
    res.meta.update(info)
    return res


def paint_result(labels_volume: np.ndarray, det_labels: np.ndarray,
                 out_ids: np.ndarray, dtype=np.uint16) -> np.ndarray:
    """把"检测标签 → 输出轨迹 id"的赋值画回体数据。

    输出默认用 uint16，避免 3D 大体积下 int64 造成的数十倍内存开销。
    """
    if det_labels.size == 0:
        return np.zeros(labels_volume.shape, dtype=dtype)
    max_label = int(max(det_labels.max(), int(labels_volume.max())))
    lut = np.zeros(max_label + 1, dtype=np.int64)
    lut[det_labels.astype(np.int64)] = out_ids.astype(np.int64)
    vol = np.asarray(labels_volume)
    return lut[vol].astype(dtype, copy=False)


def finalize_tracks(assignment: dict[int, np.ndarray],
                    tracks: dict[int, Track],
                    ) -> tuple[dict[int, np.ndarray], dict[int, Track], dict]:
    """把重建结果规范化为合法 CTC 提交（关键防御，官方评测要求）。

    规则：
      1. **丢弃幽灵轨迹**：出现在 `tracks` 但从未在任何帧被赋值的 id
         （官方 TRAMeasure 会直接报 "track not consistent with the image data"）；
      2. **起止帧以实际赋值为准**：`begin/end` 由该 id 真实出现的帧决定；
      3. **打断不连续**：若某 id 的出现帧不连续（中间缺帧），拆成多条连续轨迹，
         后段使用新的 id（保留跨度信息，避免整条被判非法）。
    """
    frames_of: dict[int, list[int]] = {}
    for t in sorted(assignment):
        for tid in np.unique(assignment[t]):
            frames_of.setdefault(int(tid), []).append(t)

    new_assignment = {t: arr.copy() for t, arr in assignment.items()}
    new_tracks: dict[int, Track] = {}
    next_id = max([*tracks.keys(), *frames_of.keys(), 0]) + 1
    dropped = split = 0

    for tid, tr in tracks.items():
        if tid not in frames_of:
            dropped += 1
            continue
        fr = sorted(frames_of[tid])
        segments: list[list[int]] = [[fr[0]]]
        for f in fr[1:]:
            if f == segments[-1][-1] + 1:
                segments[-1].append(f)
            else:
                segments.append([f])
        if len(segments) > 1:
            split += len(segments) - 1
        for k, seg in enumerate(segments):
            use_id = tid if k == 0 else next_id
            if k > 0:
                next_id += 1
                for t in seg:
                    arr = new_assignment[t]
                    arr[arr == tid] = use_id
            new_tracks[use_id] = Track(use_id, seg[0], seg[-1],
                                       tr.parent if k == 0 else 0)

    # 只在赋值里出现、却不在 tracks 里的 id（例如新出现的轨迹）
    for tid, fr in frames_of.items():
        if tid in new_tracks:
            continue
        fr = sorted(fr)
        new_tracks[tid] = Track(tid, fr[0], fr[-1], 0)

    # ---- 规范化父子关系（官方 TRAMeasure 硬性要求）----
    #  (a) 子轨迹起点必须等于父轨迹终点 + 1（分裂必须紧邻）
    #  (b) 一个父轨迹最多 2 个子节点
    by_parent: dict[int, list[int]] = {}
    for tid, tr in new_tracks.items():
        if tr.parent:
            by_parent.setdefault(tr.parent, []).append(tid)
    fixed_parent = dropped_parent = 0
    for parent, kids in by_parent.items():
        ptr = new_tracks.get(parent)
        kids_sorted = sorted(kids, key=lambda c: (new_tracks[c].begin, c))
        keep = set(kids_sorted[:2])
        if len(kids_sorted) > 2:
            dropped_parent += len(kids_sorted) - 2
        for c in kids_sorted:
            ctr = new_tracks[c]
            ok = (ptr is not None and c in keep and ctr.begin == ptr.end + 1)
            if not ok:
                new_tracks[c] = Track(c, ctr.begin, ctr.end, 0)
                fixed_parent += 1

    info = {"ghost_dropped": dropped, "segments_split": split,
            "parent_fixed": fixed_parent, "extra_children_removed": dropped_parent}
    return new_assignment, new_tracks, info
