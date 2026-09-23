"""§2.0.1 训练：式(29) 的边标签 + 式(34) 边级 BCE + 式(35) OT 一致性正则。

原文（R1 抄录）
--------------
式(34)  L_edge = − Σ_{e ∈ E_time} [ y_e log ŷ_e + (1−y_e) log(1−ŷ_e) ]
式(35)  L_OT-reg = Σ_{e ∈ E_time} ŷ_e C_e
        "该项鼓励被 GNN 判为'高置信真实边'的关联同时具有较低的 OT 代价"

数值归一化说明（ENG_SUPP，原文是"求和"）
----------------------------------------
原文两式都是对 E_time 求和。批训练时求和会让损失随帧内候选数线性增长，
故本实现取**均值**：`L_edge = mean over candidates`、`L_OT-reg = mean over candidates`。
因此 `lambda_ot` 的含义是"每条候选边的权重"，其量纲与 C（µm²）耦合——
`scripts/calibrate_params.py` 会打印 `median(C)` 供标定。
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# 复用：图批处理与磁盘图数据集（格式中立，与本包的 npz 键名对齐）
from celltracker.gnn.data import PairDataset, collate

from .model import EdgeGNN, ModelConfig

__all__ = ["TrainConfig", "train", "evaluate"]


@dataclass
class TrainConfig:
    epochs: int = 60
    batch_pairs: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden: int = 64
    layers: int = 3
    dropout: float = 0.1
    lambda_ot: float = 0.2        # 式(35) 的权重（CALIBRATED）
    residual: bool = False        # False = 严格按式(30)(32)
    class_weighted_ce: bool = False   # ENG_SUPP（非原文口径）
    val_fraction: float = 0.2
    seed: int = 20260923
    device: str = "cpu"
    out_dir: str = "paperpipe/runs/gnn"


def _pos_weight(ds: PairDataset, device) -> torch.Tensor | None:
    pos = neg = 0
    for f in ds.files:
        d = np.load(f)
        y = d["label"]
        pos += int((y > 0).sum())
        neg += int((y == 0).sum())
    pos, neg = max(pos, 1), max(neg, 1)
    return torch.tensor([neg / pos], dtype=torch.float32, device=device)


@torch.no_grad()
def evaluate(model: EdgeGNN, ds: PairDataset, device: str,
             batch_pairs: int = 8, tau: float = 0.5) -> dict:
    model.eval()
    tp = fp = fn = tn = 0
    for start in range(0, len(ds), batch_pairs):
        batch = collate([ds[i] for i in range(start, min(start + batch_pairs, len(ds)))])
        node = batch["node_feat"].to(device)
        eidx = batch["edge_index"].to(device)
        efeat = batch["edge_feat"].to(device)
        prob = torch.sigmoid(model(node, eidx, efeat))[batch["cand_mask"].to(device)]
        y = batch["label"].to(device)
        pred = prob >= tau
        tp += int((pred & (y > 0)).sum())
        fp += int((pred & (y == 0)).sum())
        fn += int((~pred & (y > 0)).sum())
        tn += int((~pred & (y == 0)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return {"precision": prec, "recall": rec, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def train(graph_root: str | Path, cfg: TrainConfig | None = None) -> dict:
    """式(34)(35) 的训练循环。图数据集由 `graph.build_dataset` 产出。"""
    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = PairDataset(graph_root, split="train", val_fraction=cfg.val_fraction,
                           seed=cfg.seed)
    val_ds = PairDataset(graph_root, split="val", val_fraction=cfg.val_fraction,
                         seed=cfg.seed)
    if len(train_ds) == 0:
        raise RuntimeError(f"{graph_root} 里没有 pair_*.npz 图数据集")
    device = torch.device(cfg.device)
    dims = np.load(train_ds.files[0])
    mcfg = ModelConfig(node_dim=int(dims["node_feat"].shape[1]),
                       edge_dim=int(dims["edge_feat"].shape[1]),
                       hidden=cfg.hidden, layers=cfg.layers,
                       dropout=cfg.dropout, residual=cfg.residual)
    model = EdgeGNN(mcfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    pos_weight = _pos_weight(train_ds, device) if cfg.class_weighted_ce else None
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    history: list[dict] = []
    best = -1.0
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for start in range(0, len(train_ds), cfg.batch_pairs):
            idx = np.random.permutation(len(train_ds))[start:start + cfg.batch_pairs]
            batch = collate([train_ds[int(i)] for i in idx])
            node = batch["node_feat"].to(device)
            eidx = batch["edge_index"].to(device)
            efeat = batch["edge_feat"].to(device)
            logits = model(node, eidx, efeat)[batch["cand_mask"].to(device)]
            y = batch["label"].to(device).float()
            loss = bce(logits, y)                                   # 式(34)
            if cfg.lambda_ot > 0:
                prob = torch.sigmoid(logits)
                cost = batch["cand_feat"][:, 0].to(device).clamp(min=0.0)
                loss = loss + cfg.lambda_ot * (prob * cost).mean()  # 式(35)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        m = evaluate(model, val_ds, device, cfg.batch_pairs)
        m.update({"epoch": epoch, "elapsed_s": time.time() - t0})
        history.append(m)
        print(f"[epoch {epoch:3d}] P={m['precision']:.4f} R={m['recall']:.4f} "
              f"F1={m['f1']:.4f}", flush=True)
        if m["f1"] > best:
            best = m["f1"]
            torch.save({"model": model.state_dict(), "model_config": asdict(mcfg),
                        "train_config": asdict(cfg), "epoch": epoch},
                       out_dir / "best.pt")
    torch.save({"model": model.state_dict(), "model_config": asdict(mcfg),
                "train_config": asdict(cfg), "epoch": cfg.epochs},
               out_dir / "last.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    return {"history": history, "out_dir": str(out_dir), "best_f1": best,
            "node_dim": mcfg.node_dim, "edge_dim": mcfg.edge_dim}
