"""§2.0.1 训练：式(29) 的边标签 + 式(34) 边级 BCE + 式(35) OT 一致性正则。

论文公式
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

训练/验证切分（重要协议）
------------------------
原仓库复用的 `celltracker.gnn.data.PairDataset` 是**随机**切 train/val，
相邻帧对会同时出现在训练与验证里（帧 t−1,t 与 t,t+1 共享节点）→ 验证 F1 虚高。
本模块改用**按时间块切分**（默认前 80% 帧对训练、后 20% 验证），
并把"跨序列留出"（在 seq01 上训练、在 seq02 上评测）作为真正的泛化证据。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# 复用：图批处理与磁盘图数据集（格式中立，与本包的 npz 键名对齐）
from celltracker.gnn.data import collate

from .model import EdgeGNN, ModelConfig
from ..graph.build import EDGE_COST

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
    lambda_ot: float = 0.2        # 式(35) 的候选权重；部署值须结合实际 C 分布验证
    residual: bool = False        # False = 严格按式(30)(32)
    class_weighted_ce: bool = False   # ENG_SUPP（非原文口径）
    val_fraction: float = 0.2
    # "block" = 按时间块切分（默认，避免相邻帧泄漏）；"random" = 原仓库口径（对照）
    split_mode: str = "block"
    seed: int = 20260923
    device: str = "cpu"
    out_dir: str = "data/interim/gnn"
    purge_overlap: bool = True
    resume: bool = False


class GraphDataset:
    """按**时间块**切分的图数据集（默认口径；`split_mode="random"` 可回到原仓库口径）。

    样本按 `pair_%04d.npz` 里的帧号排序，前 `1-val_fraction` 的帧对作训练、
    其余作验证 —— 时间上相邻的帧对不会被切成"一半训练一半验证"。
    """

    def __init__(self, root: str | Path, split: str = "train",
                 val_fraction: float = 0.2, mode: str = "block", seed: int = 0,
                 purge_overlap: bool = True):
        self.root = Path(root)
        self.files = sorted(self.root.glob("pair_*.npz"))
        if not self.files:
            raise FileNotFoundError(f"{self.root} 里没有 pair_*.npz")
        if not 0 <= val_fraction < 1:
            raise ValueError('val_fraction 须在 [0,1)')
        n_val = max(1, int(len(self.files) * float(val_fraction))) if val_fraction > 0 else 0
        if mode == "random":
            rng = np.random.default_rng(seed)
            idx = rng.permutation(len(self.files))
            val_idx = set(idx[:n_val].tolist())
            self.sel = [f for i, f in enumerate(self.files)
                        if (i in val_idx) == (split == "val")]
        elif mode == "block":
            cut = len(self.files) - n_val
            self.sel = (self.files[:cut] if split == "train" else self.files[cut:])
            if split == "train" and n_val and purge_overlap:
                with np.load(self.files[cut]) as first_validation:
                    boundary = int(first_validation.get("support_start", first_validation["t"]))
                kept = []
                for path in self.sel:
                    with np.load(path) as graph:
                        if int(graph.get("support_end",int(graph["t"])+1)) < boundary:
                            kept.append(path)
                self.sel = kept
        else:
            raise ValueError(f"未知 split_mode: {mode!r}")
        manifest = self.root / "manifest.json"
        self.meta = json.loads(manifest.read_text()) if manifest.exists() else {}
        self.cache = {}

    def __len__(self) -> int:
        return len(self.sel)

    def __getitem__(self, i: int) -> dict:
        if i in self.cache:
            return self.cache[i]
        with np.load(self.sel[i]) as stored:
            d = {key:stored[key] for key in stored.files}
        item = {
            "node_feat": torch.from_numpy(d["node_feat"]),
            "edge_index": torch.from_numpy(d["edge_index"].astype(np.int64)),
            "edge_feat": torch.from_numpy(d["edge_feat"]),
            "is_target": torch.from_numpy(d["is_target"]),
            "label": torch.from_numpy(d["label"]),
            "cand_feat": torch.from_numpy(d["cand_feat"]),
            "cand_edges": torch.from_numpy(d["cand_edges"].astype(np.int64)),
            "meta": {"t": int(d["t"]), "n_src": int(d["n_src"]),
                     "n_dst": int(d["n_dst"])},
        }
        self.cache[i] = item
        return item


def _pos_weight(ds, device) -> torch.Tensor | None:
    pos = neg = 0
    for f in ds.sel:
        with np.load(f) as d:
            y = d["label"]
        pos += int((y > 0).sum())
        neg += int((y == 0).sum())
    pos, neg = max(pos, 1), max(neg, 1)
    return torch.tensor([neg / pos], dtype=torch.float32, device=device)


@torch.no_grad()
def evaluate(model: EdgeGNN, ds, device: str,
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
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.use_deterministic_algorithms(True)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.glob('*.pt')) and not cfg.resume:
        raise FileExistsError("训练输出已存在；恢复请显式设置 resume=True")
    if cfg.resume and not (out_dir/'last.pt').exists():
        raise FileNotFoundError('恢复训练需要 last.pt')

    ds_kw = dict(val_fraction=cfg.val_fraction, mode=cfg.split_mode, seed=cfg.seed,
                 purge_overlap=cfg.purge_overlap)
    train_ds = GraphDataset(graph_root, split="train", **ds_kw)
    val_ds = GraphDataset(graph_root, split="val", **ds_kw)
    if len(train_ds) == 0:
        raise RuntimeError("排除重叠上下文后训练集为空；扩大数据范围，或仅冒烟时显式关闭验证划分")
    device = torch.device(cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("要求 CUDA 训练，但 GPU 不可用")
    dims = np.load(train_ds.sel[0])
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
    first_epoch = 1
    contract = train_ds.meta.get("input_contract")
    if cfg.resume and (out_dir / "last.pt").exists():
        saved = torch.load(out_dir / "last.pt",map_location=device,weights_only=False)
        if saved.get("input_contract") != contract:
            raise ValueError("恢复检查点的数据契约不匹配")
        previous = saved['train_config']
        for key,value in asdict(cfg).items():
            if key not in {'resume','epochs','device','out_dir'} and previous.get(key) != value:
                raise ValueError(f'恢复训练配置不一致：{key}')
        model.load_state_dict(saved["model"])
        opt.load_state_dict(saved["optimizer"])
        first_epoch = saved["epoch"]+1
        best = saved["best_f1"]
        history = saved["history"]
        np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"].cpu())
        if device.type == "cuda" and saved.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    def checkpoint(epoch):
        return {"model":model.state_dict(),"model_config":asdict(mcfg),"train_config":asdict(cfg),
                "epoch":epoch,"input_contract":contract,"optimizer":opt.state_dict(),
                "best_f1":best,"history":history,"numpy_rng":np.random.get_state(),
                "torch_rng":torch.get_rng_state(),
                "cuda_rng":torch.cuda.get_rng_state_all() if device.type=="cuda" else None}
    t0 = time.time()
    for epoch in range(first_epoch, cfg.epochs + 1):
        model.train()
        order = np.random.permutation(len(train_ds))
        epoch_loss, seen = 0.0, 0
        for start in range(0, len(train_ds), cfg.batch_pairs):
            idx = order[start:start + cfg.batch_pairs]
            batch = collate([train_ds[int(i)] for i in idx])
            node = batch["node_feat"].to(device)
            eidx = batch["edge_index"].to(device)
            efeat = batch["edge_feat"].to(device)
            logits = model(node, eidx, efeat)[batch["cand_mask"].to(device)]
            y = batch["label"].to(device).float()
            if len(y) == 0:
                continue
            loss = bce(logits, y)                                   # 式(34)
            if cfg.lambda_ot > 0:
                prob = torch.sigmoid(logits)
                cost = batch["cand_feat"][:, EDGE_COST].to(device)
                if torch.any(cost < 0):
                    raise ValueError("原始特征代价 C 不应为负")
                loss = loss + cfg.lambda_ot * (prob * cost).mean()  # 式(35)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            epoch_loss += float(loss.detach())*len(idx)
            seen += len(idx)
        m = evaluate(model, val_ds if len(val_ds) else train_ds, device, cfg.batch_pairs)
        m['selection_split'] = 'validation' if len(val_ds) else 'train_smoke_only'
        m.update({"epoch": epoch, "elapsed_s": time.time() - t0,
                  "graphs_seen":seen,"train_loss":epoch_loss/max(seen,1)})
        history.append(m)
        print(f"[epoch {epoch:3d}] P={m['precision']:.4f} R={m['recall']:.4f} "
              f"F1={m['f1']:.4f}", flush=True)
        if m["f1"] > best:
            best = m["f1"]
            torch.save(checkpoint(epoch),out_dir / "best.pt")
        temporary = out_dir / "last.pending.pt"
        torch.save(checkpoint(epoch),temporary)
        temporary.replace(out_dir / "last.pt")
        (out_dir / "history.json").write_text(json.dumps(history,indent=2))
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    return {"history": history, "out_dir": str(out_dir), "best_f1": best,
            "node_dim": mcfg.node_dim, "edge_dim": mcfg.edge_dim,
            "split_mode": cfg.split_mode, "n_train": len(train_ds),
            "n_val": len(val_ds)}


@torch.no_grad()
def evaluate_graphs(checkpoint: str | Path, graph_root: str | Path,
                    device: str = "cpu", tau: float = 0.5,
                    batch_pairs: int = 8) -> dict:
    """在**另一个**图数据集上评测（跨序列留出用：seq01 训练 → seq02 评测）。"""
    from .infer import load_model

    model = load_model(checkpoint, device)
    ds = GraphDataset(graph_root, split="train", val_fraction=0.0)
    from ..runtime.contracts import check_contract
    check_contract(model.input_contract,ds.meta.get('input_contract'))
    out = evaluate(model, ds, device, batch_pairs, tau)
    out["n_graphs"] = len(ds)
    return out
