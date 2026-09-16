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
from .fusion import conditional_division_prob, fused_existence
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
    lambda_ot: float = 0.2        # L_OT-reg 权重（式 35）
    fusion_loss: bool = False     # True = 使用自创的 λ 融合损失（消融项，非论文口径）
    lam: float = 0.5              # 仅 fusion_loss=True 时有效
    lam_random: bool = False      # 仅 fusion_loss=True 时有效
    label_smoothing: float = 0.0
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
    bce = nn.BCELoss()

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
            y = batch["label"]
            prob = torch.softmax(logits, dim=-1)

            # ================= 论文口径（式 34 + 式 35）=================
            # 式(34) 边级交叉熵。原文为二分类（边是否真实）；我们扩展为
            # 三分类以显式区分"移动"与"分裂"语义——对应 AOGM 里的 EC 项
            # （错误语义），这样模型才能像 §2.0.1 要求的那样校正 OT 的
            # "把分裂当移动"错误。
            loss = ce(logits, y)
            p_related = prob[:, 1] + prob[:, 2]        # ŷ_e：模型判为"真实边"的概率

            # 消融项（非论文口径）：把 OT 先验与模型输出显式加权融合后再监督。
            if cfg.fusion_loss:
                rnorm = batch["cand_feat"][:, 8].clamp(0.0, 1.0)
                lam = (float(np.random.uniform(0, 1)) if cfg.lam_random else cfg.lam)
                a_fused = fused_existence(rnorm, p_related, lam)
                loss = bce(a_fused.clamp(1e-6, 1 - 1e-6), (y > 0).float())

            if cfg.lambda_ot > 0:
                # 式(35)：L_OT-reg = Σ_e ŷ_e · C_e
                # 作用：让"高置信真实边"同时具有较低的 OT 代价，保持学习决策与
                # 显式代价骨架一致。C 是 cand_feat 的第 5 列（0=disp_x,1=disp_y,
                # 2=disp_z,3=|disp|,4=size_ratio,**5=cost**,6=d_pred,7=log_mass,
                # 8=rnorm,9=is_argmax,10=flag）。按平均代价归一以稳定量纲。
                cost = batch["cand_feat"][:, 5].clamp(min=0)
                cost_norm = cost / cost.mean().clamp(min=1e-6)
                loss = loss + cfg.lambda_ot * (p_related * cost_norm).mean()
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
              f"move_f1={metrics['f1_move']:.4f} div_f1={metrics['f1_div']:.4f} "
              f"sem_acc={metrics['sem_acc']:.4f}", flush=True)
        if metrics["f1_move"] + metrics["f1_div"] > best_f1:
            best_f1 = metrics["f1_move"] + metrics["f1_div"]
            torch.save({"model": model.state_dict(), "config": asdict(cfg),
                        "model_config": asdict(model.cfg), "epoch": epoch},
                       out_dir / "best.pt")

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    torch.save({"model": model.state_dict(), "config": asdict(cfg),
                "model_config": asdict(model.cfg), "epoch": cfg.epochs},
               out_dir / "last.pt")
    return {"history": history, "out_dir": str(out_dir)}


@torch.no_grad()
def evaluate(model: EdgeGNN, ds: PairDataset, device, batch_pairs: int = 8,
             lams: tuple[float, ...] = ()) -> dict:
    """论文口径评估：模型自身的边分类（式33）。

    - `move_f1` / `div_f1`：模型 argmax 的三分类结果（主指标）
    - `exist_precision/recall`：把 move/div 合并为"存在边"后的二分类指标
      （对应原文式(29)(34) 的二分类设定）
    - `sem_acc`：在真实边上区分"移动 vs 分裂"的准确率
    - `exist_f1_lam*`：仅在消融时给出（λ 融合口径）
    """
    model.eval()
    tp = np.zeros(N_CLASSES); fp = np.zeros(N_CLASSES); fn = np.zeros(N_CLASSES)
    correct = total = 0
    stats = {lam: {"tp": 0, "fp": 0, "fn": 0} for lam in lams}
    sem_correct = sem_total = 0
    for start in range(0, len(ds), batch_pairs):
        batch = collate([ds[i] for i in range(start, min(start + batch_pairs, len(ds)))])
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        logits = model(batch["node_feat"], batch["edge_index"], batch["edge_feat"])
        logits = logits[batch["cand_mask"]]
        prob = torch.softmax(logits, dim=-1).cpu().numpy()
        rnorm = batch["cand_feat"][:, 8].cpu().numpy()
        y = batch["label"].cpu().numpy()
        pred = prob.argmax(-1)
        correct += int((pred == y).sum()); total += y.size
        for c in range(N_CLASSES):
            tp[c] += int(((pred == c) & (y == c)).sum())
            fp[c] += int(((pred == c) & (y != c)).sum())
            fn[c] += int(((pred != c) & (y == c)).sum())
        exist = y > 0
        p_rel = prob[:, 1] + prob[:, 2]
        pred_exist = (p_rel >= 0.5)
        e_tp = int((pred_exist & exist).sum()); e_fp = int((pred_exist & ~exist).sum())
        e_fn = int((~pred_exist & exist).sum())
        for lam in lams:
            a = fused_existence(rnorm, p_rel, lam)
            pl = a >= 0.5
            s = stats[lam]
            s["tp"] += int((pl & exist).sum()); s["fp"] += int((pl & ~exist).sum())
            s["fn"] += int((~pl & exist).sum())
        # 语义（移动 vs 分裂）仅在真实边上评估
        if exist.any():
            cond = np.stack([prob[exist, 1], prob[exist, 2]], axis=-1)
            sem_correct += int((cond.argmax(-1) == (y[exist] == 2).astype(int)).sum())
            sem_total += int(exist.sum())

    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    prec = e_tp / max(e_tp + e_fp, 1)
    rec = e_tp / max(e_tp + e_fn, 1)
    out = {"accuracy": correct / max(total, 1), "f1_none": float(f1[0]),
           "f1_move": float(f1[1]), "f1_div": float(f1[2]),
           "exist_precision": prec, "exist_recall": rec,
           "exist_f1": 2 * prec * rec / max(prec + rec, 1e-9),
           "support_move": int(tp[1] + fn[1]), "support_div": int(tp[2] + fn[2])}
    for lam in lams:
        s = stats[lam]
        p_ = s["tp"] / max(s["tp"] + s["fp"], 1)
        r_ = s["tp"] / max(s["tp"] + s["fn"], 1)
        out[f"exist_lam{str(lam).replace('.', '')}_f1"] = 2 * p_ * r_ / max(p_ + r_, 1e-9)
    out["sem_acc"] = sem_correct / max(sem_total, 1)
    return out
