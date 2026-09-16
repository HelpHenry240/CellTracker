"""用训练好的 GNN 做推理并重建轨迹（ideas.pdf §2.0.1 决策阶段）。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..data.ctc import Track
from ..track.base import Detections, TrackResult, finalize_tracks
from .data import PairDataset, collate
from .model import EdgeGNN, ModelConfig, N_CLASSES

__all__ = ["InferConfig", "predict_pairs", "reconstruct_tracks"]


@dataclass
class InferConfig:
    tau_move: float = 0.5      # 接受"移动边"的概率阈值
    tau_div: float = 0.5       # 接受"分裂边"的概率阈值
    max_children: int = 2
    device: str = "cpu"


@torch.no_grad()
def predict_pairs(checkpoint: str | Path, graph_root: str | Path,
                  batch_pairs: int = 16, device: str = "cpu") -> dict[int, dict]:
    """对图数据集里的每一对帧输出候选边的类别概率。"""
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = EdgeGNN(ModelConfig(**ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = PairDataset(graph_root, split="train", val_fraction=0.0)
    ds.sel = sorted(Path(graph_root).glob("pair_*.npz"))   # 全部帧对
    out: dict[int, dict] = {}
    for start in range(0, len(ds), batch_pairs):
        items = [ds[i] for i in range(start, min(start + batch_pairs, len(ds)))]
        batch = collate(items)
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        logits = model(batch["node_feat"], batch["edge_index"], batch["edge_feat"])
        prob = torch.softmax(logits[batch["cand_mask"]], dim=-1).cpu().numpy()
        cand = batch["cand_feat"].cpu().numpy()
        offset = 0
        for it in items:
            n_cand = it["cand_feat"].shape[0]
            t = it["meta"]["t"]
            d = np.load(Path(graph_root) / f"pair_{t:04d}.npz")
            out[t] = {
                "prob": prob[offset:offset + n_cand],
                "pairs": d["cand_edges"],
                "t_next": int(d["t_next"]),
                "n_src": int(d["n_src"]), "n_dst": int(d["n_dst"]),
                "cost": cand[offset:offset + n_cand, 3],
            }
            offset += n_cand
    return out


def reconstruct_tracks(dets: Detections, preds: dict[int, dict],
                       cfg: InferConfig | None = None) -> TrackResult:
    """按 GNN 边分类结果重建轨迹。

    规则：
      1. 每个源细胞的"移动后继"= P_move 最大且 ≥ τ_move 的目标；
      2. 若某源有 ≥2 个目标的 P_div ≥ τ_div，则判为分裂（取概率最高的两个）；
      3. 目标冲突时按置信度高者优先；未被占用的目标视为新生。
    """
    cfg = cfg or InferConfig()
    ts = dets.t_range
    res = TrackResult(meta={"config": cfg.__dict__, "source": "gnn"})
    next_id = 1
    assignment: dict[int, np.ndarray] = {}
    first = ts[0]
    ids0 = np.arange(next_id, next_id + dets.n(first))
    next_id += dets.n(first)
    assignment[first] = ids0

    for t in ts[:-1]:
        if t not in preds:
            assignment[t + 1] = np.arange(next_id, next_id + dets.n(t + 1))
            next_id += dets.n(t + 1)
            continue
        p = preds[t]
        prob, pairs = p["prob"], p["pairs"]
        n_src, n_dst = int(p["n_src"]), int(p["n_dst"])
        src_ids = assignment[t]
        out_ids = np.zeros(n_dst, dtype=np.int64)
        claimed: dict[int, float] = {}     # 目标 -> 置信度

        move_conf: dict[int, tuple[int, float]] = {}
        div_cands: dict[int, list[tuple[int, float]]] = {}
        used_as_parent: set[int] = set()
        for k in range(pairs.shape[0]):
            i, j = int(pairs[k, 0]), int(pairs[k, 1])
            pm, pd = float(prob[k, 1]), float(prob[k, 2])
            if pd >= cfg.tau_div and pd >= pm:
                if int(src_ids[i]) not in used_as_parent:
                    div_cands.setdefault(i, []).append((j, pd))
            elif pm >= cfg.tau_move:
                if i not in move_conf or pm > move_conf[i][1]:
                    move_conf[i] = (j, pm)

        # 先处理分裂（语义更强）：整对一起检查，避免覆盖已创建的轨迹记录
        for i, cands in div_cands.items():
            cands = sorted(cands, key=lambda x: -x[1])[: cfg.max_children]
            if len(cands) < 2:
                continue
            if any(j in claimed for j, _ in cands):
                continue          # 任一子目标已被占用 -> 放弃这次分裂（不覆盖）
            parent_tid = int(src_ids[i])
            used_as_parent.add(parent_tid)   # 分裂后父轨迹必须终止
            for j, conf in cands:
                out_ids[j] = next_id
                claimed[j] = conf
                res.tracks[next_id] = Track(next_id, t + 1, t + 1, parent_tid)
                next_id += 1

        # 再处理移动
        for i, (j, conf) in sorted(move_conf.items(), key=lambda kv: -kv[1][1]):
            if int(src_ids[i]) in used_as_parent:
                continue      # 已分裂的父轨迹不再延续（否则父子帧关系非法）
            if j in claimed and claimed[j] >= conf:
                continue
            if out_ids[j] != 0:
                continue
            out_ids[j] = int(src_ids[i])
            claimed[j] = conf

        # 其余目标：新生
        for j in range(n_dst):
            if out_ids[j] == 0:
                out_ids[j] = next_id
                res.tracks[next_id] = Track(next_id, t + 1, t + 1, 0)
                next_id += 1
        assignment[t + 1] = out_ids

    assignment, tracks, info = finalize_tracks(assignment, res.tracks)
    res.assignment = assignment
    res.tracks = tracks
    res.meta.update(info)
    return res
