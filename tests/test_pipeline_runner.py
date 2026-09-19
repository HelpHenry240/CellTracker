"""统一 runner 的端到端测试（小规模合成数据，不依赖真实数据集）。"""

from __future__ import annotations

import numpy as np
import pytest

from celltracker.pipeline import apply_ablation, run_pipeline
from celltracker.pipeline.config import (MeasureConfig, OTConfig, PipelineConfig,
                                         ReconstructConfig, TrackletConfig)


@pytest.fixture()
def fake_h5(tmp_path):
    """构造一个最小的内部 HDF5（4 帧 × 5 细胞，匀速右移 + 一次分裂）。"""
    import h5py

    path = tmp_path / "fake.h5"
    n_cells = 5
    with h5py.File(path, "w") as f:
        f.attrs.update(name="Fake", seq="01", ndim=3,
                       shape=np.array([4, 64, 64], dtype=np.int32),
                       n_frames=4, n_frames_total=4)
        arr = np.zeros(4, dtype=[("label", "i4"), ("begin", "i4"),
                                 ("end", "i4"), ("parent", "i4")])
        arr["label"] = np.arange(1, 5)
        arr["begin"] = 0
        arr["end"] = 3
        f.create_dataset("tracks", data=arr)
        f.create_dataset("seg_frames", data=np.array([0], dtype=np.int32))
        gf = f.create_group("frames")
        for t in range(4):
            g = gf.create_group(f"{t:04d}")
            xy = np.stack([np.arange(n_cells) * 8.0 + 2.0 * t,
                           np.arange(n_cells) * 3.0,
                           np.zeros(n_cells)], axis=1)
            vol = np.full(n_cells, 100.0)
            labels_vol = np.zeros((4, 64, 64), dtype=np.uint16)
            for k in range(n_cells):
                z, y, x = int(xy[k, 2]), int(xy[k, 1]), int(xy[k, 0])
                labels_vol[z, y, y:y + 2] = k + 1
            g.create_dataset("labels", data=labels_vol)
            g.create_dataset("label", data=np.arange(1, n_cells + 1, dtype=np.int32))
            g.create_dataset("centroid", data=xy.astype(np.float32))
            g.create_dataset("volume", data=vol.astype(np.int32))
            g.create_dataset("bbox_min", data=np.zeros((n_cells, 3), dtype=np.int16))
            g.create_dataset("bbox_max", data=np.zeros((n_cells, 3), dtype=np.int16))
            g.create_dataset("intensity_mean", data=np.full(n_cells, 100.0, np.float32))
            g.create_dataset("intensity_std", data=np.full(n_cells, 5.0, np.float32))
    return path


def test_runner_end_to_end(fake_h5, tmp_path):
    """默认配置（GNN 关闭 → OT 规则）：应产出合法轨迹表与中间产物。"""
    cfg = PipelineConfig(
        measure=MeasureConfig(),
        ot=OTConfig(r_max=30.0, eps_rel=0.1),
        reconstruct=ReconstructConfig(tau_move=0.5, tau_div=0.5),
        tracklet=TrackletConfig(enabled=False),
    )
    run = run_pipeline(fake_h5, cfg, artifacts_dir=tmp_path / "art")
    assert run.info["decision"] == "ot_rule"
    assert run.track_result.n_tracks() >= 1
    assert (tmp_path / "art" / "couplings.npz").exists()
    assert (tmp_path / "art" / "run_info.json").exists()


def test_runner_ablation_switches_are_effective(fake_h5, tmp_path):
    """消融开关必须真的改变运行配置（fgw/unbalanced/motion/multiscale/tracklet）。"""
    base = PipelineConfig(ot=OTConfig(r_max=30.0, eps_rel=0.1, eta=0.5,
                                      tau_a=1.0, tau_b=1.0, alpha_pred=1.0))
    ab = apply_ablation(base, {"fgw", "unbalanced", "motion"})
    assert ab.ot.eta == 0.0 and ab.ot.tau_a is None and ab.ot.alpha_pred == 0.0

    run = run_pipeline(fake_h5, ab, artifacts_dir=tmp_path / "art2")
    assert run.info["ablated"] == sorted(["fgw", "unbalanced", "motion"])
    # 运动先验被消融后不应有速度相关统计
    assert "velocity_live" not in run.info


def test_runner_multiscale_and_tracklet_paths(fake_h5, tmp_path):
    """开启多尺度与 tracklet 阶段时应正常执行并记录信息。"""
    from celltracker.pipeline.config import MultiscaleConfig

    cfg = PipelineConfig(
        ot=OTConfig(r_max=30.0, eps_rel=0.1),
        multiscale=MultiscaleConfig(enabled=True, ks=(2,), lambda_temp=0.1, n_rounds=1),
        tracklet=TrackletConfig(enabled=True, max_gap=2),
    )
    run = run_pipeline(fake_h5, cfg, artifacts_dir=tmp_path / "art3")
    assert run.ms_result is not None and run.ms_result.info["enabled"] is True
    assert "multiscale" in run.info
    assert run.tracklet_result is not None
    assert "tracks_final" in run.info
