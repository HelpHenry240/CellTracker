"""用训练好的 GNN 做推理并重建轨迹（ideas.pdf §2.0.1 决策阶段）。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..data.ctc import Track
from ..track.base import Detections, TrackResult, finalize_tracks
from .data import PairDataset, collate
from .fusion import conditional_division_prob, fused_existence, move_division_scores
from .model import EdgeGNN, ModelConfig, N_CLASSES

__all__ = ["InferConfig", "predict_pairs", "reconstruct_tracks"]


@dataclass
class InferConfig:
    """决策配置。

    默认走**论文口径**（§2.0.1）：OT 负责筛候选边（式26）与提供特征（式27），
    GNN 用自己的边分类概率做决策（式33）。`fusion=True` 时改用自创的
    λ 加权融合（消融项，原文没有此公式）。
    """

    tau_move: float = 0.5      # 接受"移动边"的概率阈值
    tau_div: float = 0.5       # 接受"分裂边"的概率阈值
    max_children: int = 2
    device: str = "cpu"
    fusion: bool = False       # 消融：显式加权融合 OT 先验与 GNN 输出
    lam: float = 0.5           # 仅 fusion=True 时有效
    ot_feat_cols: tuple[int, int] = (8, 9)   # cand_feat 中 (rnorm, is_argmax) 的列号


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
                "feat": cand[offset:offset + n_cand],
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

    两种模式：

    * **残差校正模式**（§2.0.1，"OT 先验 + GNN 校正"）：把 OT 的行归一化传输质量
      与 GNN 的类别概率线性融合（`λ` 控制残差强度），再统一决策：

          e_link(i→j) = (1-λ)·rnorm_ij + λ·P_move(i→j)
          e_div (i→j) = (1-λ)·rnorm_ij + λ·P_div(i→j)

      其中 `rnorm` 是 OT 传输计划的行归一化质量（可解释的局部软匹配先验），
      "两个目标的行归一化质量都显著" 正是 §1.6 中"一行对多个目标显著传输"
      的分裂判据。**λ=0 时完全退化为纯 OT 规则**（λ=0 下 `e_link = e_div = rnorm`，
      于是"有 ≥2 个目标质量超过阈值就分裂，否则连到 argmax"）。

    * **纯 GNN 模式**（λ=1，用于对照）：只看 GNN 概率，用 τ_move/τ_div 阈值。

    目标冲突时按置信度高者优先；未被占用的目标视为新生。
    """
    cfg = cfg or InferConfig()
    ts = dets.t_range
    res = TrackResult(meta={"config": cfg.__dict__, "source": "gnn"})
    next_id = 1
    assignment: dict[int, np.ndarray] = {}
    reject_stats = {"div_lt2": 0, "div_child_claimed": 0, "div_ok": 0}
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
        feat = p.get("feat")
        n_src, n_dst = int(p["n_src"]), int(p["n_dst"])
        src_ids = assignment[t]
        out_ids = np.zeros(n_dst, dtype=np.int64)
        claimed: dict[int, float] = {}     # 目标 -> 置信度

        # ---- 决策分数 ----
        if not cfg.fusion:
            # 论文口径（§2.0.1）：直接用 GNN 的边分类概率（式33）决策。
            # OT 的作用已体现在候选边（式26）与边特征（式27）中，
            # 不再另做加权融合（原文没有融合权重公式）。
            e_link = prob[:, 1].astype(np.float64)
            e_div = prob[:, 2].astype(np.float64)
        else:
            # 消融口径：自创的 λ 加权融合（P = 存在性 × 语义）
            rnorm = (feat[:, cfg.ot_feat_cols[0]].astype(np.float64)
                     if feat is not None else prob[:, 1] + prob[:, 2])
            e_link, e_div = move_division_scores(rnorm, prob[:, 1], prob[:, 2], cfg.lam)

        move_conf: dict[int, tuple[int, float]] = {}
        div_cands: dict[int, list[tuple[int, float]]] = {}
        used_as_parent: set[int] = set()
        diag = {"div_lt2": 0, "div_child_claimed": 0, "div_ok": 0}
        for k in range(pairs.shape[0]):
            i, j = int(pairs[k, 0]), int(pairs[k, 1])
            pm, pd = float(e_link[k]), float(e_div[k])
            # 两套候选独立收集：λ=0 时 e_link == e_div，若写成 if/elif 会让
            # 分裂分支"吃掉"所有边，导致移动分支永远拿不到候选。
            if pd >= cfg.tau_div and int(src_ids[i]) not in used_as_parent:
                div_cands.setdefault(i, []).append((j, pd))
            if pm >= cfg.tau_move:
                if i not in move_conf or pm > move_conf[i][1]:
                    move_conf[i] = (j, pm)

        # 先处理分裂（语义更强）：全局按置信度降序，避免"先到先得"式的误拒
        ordered_divs = sorted(
            ((max(c for _, c in v), i) for i, v in div_cands.items()),
            key=lambda x: -x[0])
        for _conf, i in ordered_divs:
            cands_all = sorted(div_cands[i], key=lambda x: -x[1])
            # 冲突时不再"整次放弃"，改为在未占用的目标里选最优的 max_children 个。
            # 理由（漏斗诊断）：直接放弃会造成两条缺失边（EA×1.5），
            # 而改选次优子目标最多损失一条边，代价更低。
            cands = [c for c in cands_all if c[0] not in claimed][: cfg.max_children]
            if len(cands) < 2:
                if len(cands_all) >= 2:
                    diag["div_child_claimed"] += 1
                else:
                    diag["div_lt2"] += 1
                continue
            parent_tid = int(src_ids[i])
            used_as_parent.add(parent_tid)   # 分裂后父轨迹必须终止
            diag["div_ok"] += 1
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
        reject_stats["div_lt2"] += diag["div_lt2"]
        reject_stats["div_child_claimed"] += diag["div_child_claimed"]
        reject_stats["div_ok"] += diag["div_ok"]

    assignment, tracks, info = finalize_tracks(assignment, res.tracks)
    res.assignment = assignment
    res.tracks = tracks
    res.meta.update(info)
    res.meta["division_reject"] = dict(reject_stats)
    return res
