"""GNN 训练循环（边 3 分类 + OT 成本一致性正则，式 34/35）。"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .data import PairDataset, collate
from .model import EdgeGNN, ModelConfig, N_CLASSES

__all__ = ["TrainConfig", "train"]


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_pairs: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden: int = 64
    layers: int = 3
    dropout: float = 0.1
    lambda_ot: float = 0.1        # L_OT-reg 权重（式 35）
    val_fraction: float = 0.2
    seed: int = 20260916
    device: str = "cpu"
    out_dir: str = "experiments/gnn_run"
    class_weights: tuple[float, ...] | None = None


def _class_weights(ds: PairDataset) -> torch.Tensor:
    counts = np.zeros(N_CLASSES, dtype=np.float64)
    for f in ds.files:
        d = np.load(f)
        counts += np.bincount(d["cand_label"], minlength=N_CLASSES)
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / (N_CLASSES * counts)
    return torch.tensor(w, dtype=torch.float32)


def train(graph_root: str | Path, cfg: TrainConfig | None = None) -> dict:
    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = PairDataset(graph_root, split="train", val_fraction=cfg.val_fraction,
                           seed=cfg.seed)
    val_ds = PairDataset(graph_root, split="val", val_fraction=cfg.val_fraction,
                         seed=cfg.seed)
    device = torch.device(cfg.device)

    model = EdgeGNN(ModelConfig(hidden=cfg.hidden, layers=cfg.layers,
                                dropout=cfg.dropout)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    weights = (_class_weights(train_ds) if cfg.class_weights is None
               else torch.tensor(cfg.class_weights, dtype=torch.float32)).to(device)
    ce = nn.CrossEntropyLoss(weight=weights)

    history = []
    best_f1 = -1.0
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        order = np.random.permutation(len(train_ds))
        losses = []
        for start in range(0, len(order), cfg.batch_pairs):
            batch = collate([train_ds[i] for i in order[start:start + cfg.batch_pairs]])
            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            logits = model(batch["node_feat"], batch["edge_index"], batch["edge_feat"])
            logits = logits[batch["cand_mask"]]
            loss = ce(logits, batch["label"])
            if cfg.lambda_ot > 0:
                # 式 (35)：被预测为"有关联"的边应具有较低 OT 成本
                cost = batch["cand_feat"][:, 3]
                p_link = torch.softmax(logits, dim=-1)[:, 1:].sum(dim=-1)
                loss = loss + cfg.lambda_ot * (p_link * cost.clamp(min=0)).mean() / 1000.0
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss))

        metrics = evaluate(model, val_ds, device, cfg.batch_pairs)
        metrics.update({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "elapsed_s": time.time() - t0})
        history.append(metrics)
        print(f"[epoch {epoch:3d}] loss={metrics['train_loss']:.4f} "
              f"val_acc={metrics['accuracy']:.4f} move_f1={metrics['f1_move']:.4f} "
              f"div_f1={metrics['f1_div']:.4f}", flush=True)
        if metrics["f1_div"] + metrics["f1_move"] > best_f1:
            best_f1 = metrics["f1_div"] + metrics["f1_move"]
            torch.save({"model": model.state_dict(), "config": asdict(cfg),
                        "model_config": asdict(model.cfg), "epoch": epoch},
                       out_dir / "best.pt")

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    torch.save({"model": model.state_dict(), "config": asdict(cfg),
                "model_config": asdict(model.cfg), "epoch": cfg.epochs},
               out_dir / "last.pt")
    return {"history": history, "out_dir": str(out_dir)}


@torch.no_grad()
def evaluate(model: EdgeGNN, ds: PairDataset, device, batch_pairs: int = 8) -> dict:
    model.eval()
    tp = np.zeros(N_CLASSES); fp = np.zeros(N_CLASSES); fn = np.zeros(N_CLASSES)
    correct = total = 0
    for start in range(0, len(ds), batch_pairs):
        batch = collate([ds[i] for i in range(start, min(start + batch_pairs, len(ds)))])
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        logits = model(batch["node_feat"], batch["edge_index"], batch["edge_feat"])
        pred = logits[batch["cand_mask"]].argmax(dim=-1).cpu().numpy()
        y = batch["label"].cpu().numpy()
        correct += int((pred == y).sum()); total += y.size
        for c in range(N_CLASSES):
            tp[c] += int(((pred == c) & (y == c)).sum())
            fp[c] += int(((pred == c) & (y != c)).sum())
            fn[c] += int(((pred != c) & (y == c)).sum())
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    return {"accuracy": correct / max(total, 1), "f1_none": float(f1[0]),
            "f1_move": float(f1[1]), "f1_div": float(f1[2]),
            "support_none": int((tp[0] + fn[0])), "support_move": int((tp[1] + fn[1])),
            "support_div": int((tp[2] + fn[2]))}
