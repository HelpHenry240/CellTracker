"""§2.0.1 推理：用训练好的边分类器（式33）产出边决策，再交给 §1.6 重建轨迹。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .graph import EDGE_DIM, build_graph, node_dim_for
from .model import EdgeGNN, ModelConfig

__all__ = ["load_model", "predict_graph", "make_edge_decider"]


def load_model(checkpoint: str | Path, device: str = "cpu") -> EdgeGNN:
    ckpt = torch.load(str(checkpoint), map_location=device, weights_only=False)
    model = EdgeGNN(ModelConfig(**ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


@torch.no_grad()
def predict_graph(model: EdgeGNN, g: dict, device: str = "cpu") -> dict:
    """对单张图里的候选边输出 ŷ_e（式33 的 sigmoid）。"""
    item = _to_tensors(g)
    node = item["node_feat"].to(device)
    eidx = item["edge_index"].to(device)
    efeat = item["edge_feat"].to(device)
    logits = model(node, eidx, efeat)
    prob = torch.sigmoid(logits[item["is_target"].to(device)]).cpu().numpy()
    return {"pairs": g["cand_edges"], "score": prob, "gap": g["cand_gap"]}


def _to_tensors(g: dict) -> dict:
    return {
        "node_feat": torch.from_numpy(np.ascontiguousarray(g["node_feat"])),
        "edge_index": torch.from_numpy(np.ascontiguousarray(g["edge_index"], dtype=np.int64)),
        "edge_feat": torch.from_numpy(np.ascontiguousarray(g["edge_feat"])),
        "is_target": torch.from_numpy(np.ascontiguousarray(g["is_target"])),
    }


def make_edge_decider(checkpoint: str | Path, cfg, device: str = "cpu",
                      mcfg=None, spacing=None):
    """生成 `decider(dets, couplings, jump) -> {t: 决策}`，可直接插进 pipeline。"""
    model = load_model(checkpoint, device)
    exp_node_dim = node_dim_for(cfg.node.f_source,
                               encoder_dim=0 if cfg.node.f_source == "intensity" else -1)

    def _decider(dets, couplings, jump, encoder_feats=None, r_max: float = 3.0):
        ts = dets.t_range
        dec: dict[int, dict] = {}
        for pos, t in enumerate(ts[:-1]):
            g = build_graph(dets, t, ts, couplings.get(pos), jump.get(t), cfg.graph,
                            mcfg, encoder_feats=encoder_feats, spacing=spacing,
                            r_max=r_max)
            if not g or g["cand_edges"].shape[0] == 0:
                continue
            dec[t] = predict_graph(model, g, device)
        return dec

    return _decider
