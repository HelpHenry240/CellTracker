"""把 `graph/build.py` 产出的图数据集包装成训练用批次。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

__all__ = ["PairDataset", "collate"]


@dataclass
class PairDataset:
    root: Path
    split: str = "train"          # train / val
    val_fraction: float = 0.2
    seed: int = 20260916

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.files = sorted(self.root.glob("pair_*.npz"))
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(len(self.files))
        n_val = max(1, int(len(self.files) * self.val_fraction))
        val_idx = set(idx[:n_val].tolist())
        self.sel = [f for i, f in enumerate(self.files)
                    if (i in val_idx) == (self.split == "val")]
        meta = self.root / "meta.json"
        self.meta = json.loads(meta.read_text()) if meta.exists() else {}

    def __len__(self) -> int:
        return len(self.sel)

    def __getitem__(self, i: int) -> dict:
        d = np.load(self.sel[i])
        n_src = int(d["n_src"])
        cand = d["cand_edges"]

        # 统一边表：候选跨帧边（type=1）+ 帧内边（type=0）
        intra = d["intra_edges"]
        edge_index = np.concatenate([intra, cand + np.array([0, n_src])], axis=0) \
            if intra.size else (cand + np.array([0, n_src]))
        n_intra = intra.shape[0]
        cand_feat = d["cand_feat"]
        # 统一特征维度：候选边特征 + 1 个"是否帧内边"标志位
        feat_dim = cand_feat.shape[1] + 1
        cand_feat_p = np.concatenate(
            [cand_feat, np.zeros((cand_feat.shape[0], 1), dtype=np.float32)], axis=1)
        intra_feat = np.zeros((n_intra, feat_dim), dtype=np.float32)
        if n_intra:
            intra_feat[:, -1] = 1.0        # 帧内边标志
            # 帧内边的前 3 维复用为位移分量（此处用 0 表示同帧）
        edge_feat = np.concatenate([intra_feat, cand_feat_p], axis=0)

        return {
            "node_feat": torch.from_numpy(d["node_feat"]),
            "edge_index": torch.from_numpy(edge_index.T.astype(np.int64)),
            "edge_feat": torch.from_numpy(edge_feat),
            "n_intra": n_intra,
            "label": torch.from_numpy(d["cand_label"]),
            "cand_feat": torch.from_numpy(cand_feat_p),
            "meta": {"t": int(d["t"]), "n_src": n_src, "n_dst": int(d["n_dst"])},
        }


def collate(items: list[dict]) -> dict:
    """把多张图拼成一个批次（节点/边偏移）。"""
    node_feats, edge_indices, edge_feats, labels, cand_feats, metas = [], [], [], [], [], []
    is_cand = []
    n_off = e_off = 0
    for it in items:
        n = it["node_feat"].shape[0]
        node_feats.append(it["node_feat"])
        edge_indices.append(it["edge_index"] + n_off)
        edge_feats.append(it["edge_feat"])
        is_cand.append(torch.cat([torch.zeros(it["n_intra"], dtype=torch.bool),
                                  torch.ones(it["cand_feat"].shape[0], dtype=torch.bool)]))
        labels.append(it["label"])
        cand_feats.append(it["cand_feat"])
        metas.append(it["meta"])
        n_off += n
        e_off += it["edge_index"].shape[1]
    return {
        "node_feat": torch.cat(node_feats, 0),
        "edge_index": torch.cat(edge_indices, 1),
        "edge_feat": torch.cat(edge_feats, 0),
        "label": torch.cat(labels, 0),
        "cand_feat": torch.cat(cand_feats, 0),
        "cand_mask": torch.cat(is_cand, 0),   # 覆盖"全部边"的掩码
        "metas": metas,
        "n_nodes": n_off,
    }
