"""B0 适配层测试：GNN 能否作为 `run_pipeline(gnn=...)` 的决策阶段接入。"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from celltracker.gnn.adapter import graph_cfg_from_pipeline, make_gnn_runner
from celltracker.gnn.model import EdgeGNN, ModelConfig
from celltracker.gnn.infer import InferConfig
from celltracker.pipeline import run_pipeline
from celltracker.pipeline.config import OTConfig, PipelineConfig


@pytest.fixture()
def fake_h5(tmp_path):
    """最小内部 HDF5（5 帧 × 6 细胞匀速右移）。"""
    import h5py

    path = tmp_path / "fake.h5"
    n_cells = 6
    with h5py.File(path, "w") as f:
        f.attrs.update(name="Fake", seq="01", ndim=3,
                       shape=np.array([4, 64, 64], dtype=np.int32), n_frames=5)
        arr = np.zeros(n_cells, dtype=[("label", "i4"), ("begin", "i4"),
                                       ("end", "i4"), ("parent", "i4")])
        arr["label"] = np.arange(1, n_cells + 1)
        arr["begin"], arr["end"] = 0, 4
        f.create_dataset("tracks", data=arr)
        f.create_dataset("seg_frames", data=np.array([0], dtype=np.int32))
        gf = f.create_group("frames")
        for t in range(5):
            g = gf.create_group(f"{t:04d}")
            xy = np.stack([np.arange(n_cells) * 9.0 + 2.0 * t,
                           np.arange(n_cells) * 3.0,
                           np.zeros(n_cells)], axis=1)
            g.create_dataset("labels", data=np.zeros((4, 64, 64), dtype=np.uint16))
            g.create_dataset("label", data=np.arange(1, n_cells + 1, dtype=np.int32))
            g.create_dataset("centroid", data=xy.astype(np.float32))
            g.create_dataset("volume", data=np.full(n_cells, 100, dtype=np.int32))
            g.create_dataset("bbox_min", data=np.zeros((n_cells, 3), dtype=np.int16))
            g.create_dataset("bbox_max", data=np.zeros((n_cells, 3), dtype=np.int16))
            g.create_dataset("intensity_mean", data=np.full(n_cells, 100.0, np.float32))
            g.create_dataset("intensity_std", data=np.full(n_cells, 5.0, np.float32))
    return path


@pytest.fixture()
def ckpt(tmp_path):
    """随机初始化的模型 checkpoint（只验证链路，不验证精度）。"""
    model = EdgeGNN(ModelConfig(node_dim=7, edge_dim=11))
    path = tmp_path / "model.pt"
    torch.save({"model": model.state_dict(), "model_config": model.cfg.__dict__,
                "config": {}, "epoch": 0}, path)
    return path


def test_graph_cfg_maps_from_pipeline_config():
    """图构建配置必须来自 pipeline 配置段（单一配置源，避免口径漂移）。"""
    cfg = PipelineConfig(ot=OTConfig(r_max=25.0, eta=0.3, eps_rel=0.05))
    cfg.graph.theta_gamma = 0.03
    g = graph_cfg_from_pipeline(cfg)
    assert g.r_max == 25.0 and g.eta == 0.3 and g.eps_rel == 0.05
    assert g.theta_gamma == 0.03 and g.cand_from_ot is True


def test_gnn_runner_plugs_into_pipeline(fake_h5, ckpt, tmp_path):
    """把 GNN 作为决策阶段接入 runner，应产出合法轨迹并落盘预测。"""
    cfg = PipelineConfig(ot=OTConfig(r_max=30.0, eps_rel=0.1))
    runner = make_gnn_runner(ckpt, InferConfig(tau_move=0.5, tau_div=0.5),
                             device="cpu", artifacts_dir=tmp_path / "gnn_art")
    run = run_pipeline(fake_h5, cfg, artifacts_dir=tmp_path / "art", gnn=runner)

    assert run.info["decision"] == "gnn"
    assert run.track_result.n_tracks() >= 1
    assert (tmp_path / "gnn_art" / "gnn_preds.npz").exists()
    # CTC 格式：轨迹在起止帧内连续
    present: dict[int, set[int]] = {}
    for t, arr in run.track_result.assignment.items():
        for tid in np.unique(arr):
            present.setdefault(int(tid), set()).add(t)
    for tid, tr in run.track_result.tracks.items():
        for t in range(tr.begin, tr.end + 1):
            assert t in present.get(tid, set()), f"轨迹 {tid} 在帧 {t} 缺失"


def test_gnn_ablation_switch_falls_back_to_ot_rule(fake_h5, ckpt, tmp_path):
    """`--ablate gnn` 时必须回退到 §1.6 OT 规则路径（消融对照）。"""
    from celltracker.pipeline import apply_ablation

    cfg = apply_ablation(PipelineConfig(ot=OTConfig(r_max=30.0, eps_rel=0.1)), {"gnn"})
    runner = make_gnn_runner(ckpt, InferConfig(), device="cpu")
    run = run_pipeline(fake_h5, cfg, gnn=runner)
    assert run.info["decision"] == "ot_rule"
    assert "gnn" in run.info["ablated"]
