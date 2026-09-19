"""B0：把训练好的 GNN 接进 `pipeline.runner` 的适配层。

`run_pipeline(..., gnn=callable)` 期望的可调用对象签名为
`(dets, couplings, cfg) -> (TrackResult, graph_dir | None)`。
本模块提供 `make_gnn_runner` 生成这样的可调用对象：

  耦合(内存) → 图(内存) → GNN 边分类 → 轨迹重建

这样 GNN 就真正进入了论文的主链路（候选边由式(26) 筛、特征含 Γ 与 C、
决策用式(33) 的边分类概率），而不是走早先"磁盘图数据集 + 独立脚本"的旁路。
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from ..graph.build import GraphConfig
from ..track.base import Detections, TrackResult
from .infer import InferConfig, predict_from_couplings, reconstruct_tracks

__all__ = ["make_gnn_runner", "graph_cfg_from_pipeline"]


def graph_cfg_from_pipeline(cfg) -> GraphConfig:
    """把 pipeline 的配置段映射到图构建配置（保持单一配置源）。"""
    g = cfg.graph
    return GraphConfig(
        r_max=cfg.ot.r_max,
        knn=g.intra_knn,
        use_ot=True,
        cand_from_ot=g.cand_from_ot,
        theta_gamma=g.theta_gamma,
        theta_c=g.theta_c,
        cand_topk=g.cand_topk,
        window=g.window,
        eta=cfg.ot.eta,
        eps=cfg.ot.eps,
        eps_rel=cfg.ot.eps_rel,
    )


def make_gnn_runner(checkpoint: str | Path, infer_cfg: InferConfig | None = None,
                    device: str = "cpu", artifacts_dir: str | Path | None = None):
    """构造可直接传给 `run_pipeline(gnn=...)` 的可调用对象。"""
    infer_cfg = infer_cfg or InferConfig()
    art_dir = Path(artifacts_dir) if artifacts_dir else None

    def _runner(dets: Detections, couplings, cfg) -> tuple[TrackResult, Path | None]:
        graph_cfg = graph_cfg_from_pipeline(cfg)
        preds = predict_from_couplings(dets, couplings, checkpoint,
                                       graph_cfg=graph_cfg, device=device)
        if art_dir is not None:
            art_dir.mkdir(parents=True, exist_ok=True)
            import numpy as np
            np.savez_compressed(
                art_dir / "gnn_preds.npz",
                **{f"prob_{t}": p["prob"] for t, p in preds.items()})
        # 决策阈值来自 pipeline 的重建配置（单一配置源）
        infer = replace(infer_cfg,
                        tau_move=cfg.reconstruct.tau_move,
                        tau_div=cfg.reconstruct.tau_div,
                        max_children=cfg.reconstruct.max_children)
        result = reconstruct_tracks(dets, preds, infer)
        return result, art_dir

    return _runner
