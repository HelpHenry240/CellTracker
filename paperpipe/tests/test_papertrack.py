"""paperpipe 的回归测试（合成数据，秒级；不依赖任何真实数据集/云）。

覆盖点：
  §1.2 式(1) 质量归一化、式(4)-(7) kNN 结构（W 必须真的非零/非死代码）
  §1.3 式(12)(14) 平衡/非平衡 OT 的边际性质
  §2.0.1 式(25)-(29) 图维度与桥接边（Δt=2）、式(29) 标签
  §1.6 式(23)(24) 的 OT 规则重建 + 桥接跨空洞
  §2.0.1 式(33)-(35) GNN 前向/训练冒烟
  §1.6/§2.0.1 跨空洞轨迹的 CTC 导出与格式校验
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# 包根 = 本文件所在目录的上一级（仓库内是 `<repo>/paperpipe/`，
# 单独解包后是 `<解包目录>/paperpipe_<日期>/`）——两种布局都能跑。
PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / "src"))

import papertrack  # noqa: E402,F401  —— 触发 _paths：把 vendor 里的 celltracker 副本加入 sys.path

from celltracker.track.base import Detections                       # noqa: E402  (vendor 副本)
from papertrack.config import GraphConfig, MeasureConfig, PipelineConfig  # noqa: E402
from papertrack.coupling import solve_coupling, solve_jump_coupling  # noqa: E402
from papertrack.graph import EDGE_DIM, build_graph                 # noqa: E402
from papertrack.reconstruction.rules import (reconstruct_from_edges,  # noqa: E402
                                             reconstruct_from_ot,
                                             volume_consistent)
from papertrack.reconstruction.tracks import holes_of, normalize_tracks  # noqa: E402
from papertrack.representation.measure import knn_structure, masses  # noqa: E402


# ---------------------------------------------------------------------------
# 合成数据
# ---------------------------------------------------------------------------

def _dets(volumes: list[list[float]], cents: list[list[list[float]]],
          gt: list[list[int]] | None = None, shape=(16, 32, 32)) -> Detections:
    frames = {}
    for t, (xy, s) in enumerate(zip(cents, volumes)):
        xy = np.asarray(xy, dtype=float).reshape(-1, 3)
        d = {"centroid": xy, "volume": np.asarray(s, dtype=float),
             "label": np.arange(1, len(xy) + 1, dtype=np.int64),
             "intensity_mean": np.full(len(xy), 100.0),
             "intensity_std": np.full(len(xy), 10.0)}
        if gt is not None:
            d["gt_label"] = np.asarray(gt[t], dtype=np.int64)
        frames[t] = d
    return Detections(frames, meta={"shape": np.asarray(shape, dtype=float)})


def _mcfg() -> MeasureConfig:
    # 合成数据用各向同性"µm"（1 体素 = 1 µm），便于手算位移
    return MeasureConfig(mass_mode="volume", knn_k=2, spacing_zyx=(1.0, 1.0, 1.0))


def _ccfg(**kw):
    from papertrack.config import CouplingConfig

    base = dict(alpha=1.0, beta=0.0, r_max=3.0, eta=0.0, eps=0.1,
                eps_rel=None, tau_a=None, tau_b=None)
    base.update(kw)
    return CouplingConfig(**base)


def _straight_motion() -> Detections:
    # 3 帧、单细胞匀速移动；第 1 帧缺失（空洞）
    return _dets(volumes=[[100.0], [], [100.0]],
                 cents=[[[4, 5, 5]], [], [[4, 9, 9]]])


# ---------------------------------------------------------------------------
# §1.2
# ---------------------------------------------------------------------------

def test_masses_follow_size():
    a = masses(np.array([1.0, 3.0]), 2, "volume")
    assert a.sum() == pytest.approx(1.0)
    assert a[1] / a[0] == pytest.approx(3.0)          # 式(1)：质量 ∝ 尺寸
    assert masses(None, 4, "uniform") == pytest.approx(np.full(4, 0.25))


def test_self_contained_vendor(tmp_path):
    """paperpipe 必须自包含：只给 `paperpipe/src` 时，`import celltracker` 应命中 vendor 副本。

    用**子进程 + 干净 PYTHONPATH + 仓库外的 cwd** 验证，避免污染同一次 pytest 会话里
    仓库自身测试对 `celltracker` 的导入（早先的 conftest 改全局 sys.path 就踩过这个坑）。
    """
    import os
    import subprocess
    import sys

    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = str(PKG / "src")
    code = ("import papertrack, celltracker;"
            "print(celltracker.__file__);"
            "from papertrack.runtime import run_pipeline;"
            "from papertrack.reconstruction import export_ctc")
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(tmp_path), env=env,
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "vendor" in proc.stdout, proc.stdout


def test_knn_structure_uses_eq6_weight_and_eq7_distance():
    xy = np.array([[0.0, 0, 0], [1.0, 0, 0], [5.0, 0, 0]])
    mcfg = _mcfg()
    mcfg.knn_k = 1                                    # 只连最近邻 → 0 与 2 不相邻
    D, W, adj = knn_structure(xy, np.array([1.0, 1.0, 1.0]), mcfg)
    assert adj[0, 1] and adj[1, 0] and not adj[0, 2]
    assert W[0, 1] > 0.0 and W[0, 2] == 0.0           # 式(6) 的 W 真的被算出来
    assert W[0, 1] <= 1.0
    assert D[0, 1] == pytest.approx(1.0)              # 式(7) 的物理距离
    assert D[0, 2] == 0.0                             # 非邻接处为 0（§1.7 稀疏近似）


# ---------------------------------------------------------------------------
# §1.3
# ---------------------------------------------------------------------------

def test_balanced_coupling_marginals():
    src = np.array([[2.0, 5, 5], [2.0, 20, 20]])
    dst = np.array([[2.5, 5, 5], [2.5, 20, 20]])
    art = solve_coupling(src, dst, np.array([1.0, 1.0]), np.array([1.0, 1.0]),
                         None, _ccfg(), _mcfg(), spacing=(1.0, 1.0, 1.0))
    assert np.allclose(art.plan.sum(axis=1), art.mass_a, atol=1e-6)   # 式(10)
    assert np.allclose(art.plan.sum(axis=0), art.mass_b, atol=1e-6)
    assert art.info["mass_mode"] == "volume"


def test_unbalanced_coupling_allows_mass_loss():
    src = np.array([[2.0, 5, 5], [2.0, 20, 20]])
    dst = np.array([[2.5, 5, 5]])                    # 第二个细胞无后继
    art = solve_coupling(src, dst, np.array([1.0, 1.0]), np.array([1.0]),
                         None, _ccfg(tau_a=0.1, tau_b=0.1), _mcfg(),
                         spacing=(1.0, 1.0, 1.0))
    assert art.row_sum[1] < art.mass_a[1]            # 式(14)：允许质量流失（死亡）


# ---------------------------------------------------------------------------
# §2.0.1 图与标签
# ---------------------------------------------------------------------------

def test_graph_shapes_and_eq29_labels():
    dets = _dets(volumes=[[100.0], [100.0], [100.0]],
                 cents=[[[4, 5, 5]], [[4, 6, 6]], [[4, 7, 7]]],
                 gt=[[7], [7], [7]])
    ts = dets.t_range
    c01 = solve_coupling(dets.centroid(0), dets.centroid(1), dets.volume(0),
                         dets.volume(1), None, _ccfg(), _mcfg(), (1.0, 1.0, 1.0))
    c02 = solve_jump_coupling(dets.centroid(0), dets.centroid(2), 2, _ccfg(),
                              _mcfg(), dets.volume(0), dets.volume(2),
                              (1.0, 1.0, 1.0))
    # bridge_scope="all" 才允许"直接候选存在时也生成桥接边"（对照口径）；
    # 默认 missing_only 对应原文"节点缺失"的前置条件（下一个测试覆盖）。
    g = build_graph(dets, 0, ts, c01, c02,
                    GraphConfig(theta_gamma_frac=0.0, bridge=True,
                                bridge_scope="all"),
                    _mcfg(), gt_parent={}, spacing=(1.0, 1.0, 1.0), r_max=3.0)
    assert g["edge_feat"].shape[1] == EDGE_DIM
    assert g["node_feat"].shape[1] == 7                       # 式(25)
    assert g["cand_gap"].max() == 2                           # 桥接边 Δt=2
    assert set(g["cand_label"].tolist()) <= {0, 1}
    # 同 GT 轨迹 → 式(29) 的正样本
    pos = g["cand_label"] == 1
    assert pos.any()
    # 注意：Δt=2 的桥接边若连接同一 GT 轨迹，按 §2.0.1 的"跨一帧平滑连接"语义
    # 同样是式(29) 的正样本（y_e=1 ⇔ 真实前驱关系成立）。
    assert (g["cand_gap"][pos] == 1).any()
    assert set(g["cand_gap"][pos].tolist()) == {1, 2}


def test_bridge_edges_only_when_middle_node_missing():
    """原文前置条件：只有帧 t+1 的节点缺失（R_max 内无候选）才生成桥接边。"""
    dets = _dets(volumes=[[100.0], [100.0], [100.0]],
                 cents=[[[4, 5, 5]], [[4, 6, 6]], [[4, 7, 7]]])
    ts = dets.t_range
    c01 = solve_coupling(dets.centroid(0), dets.centroid(1), dets.volume(0),
                         dets.volume(1), None, _ccfg(), _mcfg(), (1.0, 1.0, 1.0))
    c02 = solve_jump_coupling(dets.centroid(0), dets.centroid(2), 2, _ccfg(),
                              _mcfg(), dets.volume(0), dets.volume(2),
                              (1.0, 1.0, 1.0))
    g = build_graph(dets, 0, ts, c01, c02,
                    GraphConfig(theta_gamma_frac=0.0, bridge=True,
                                bridge_scope="missing_only"),
                    _mcfg(), spacing=(1.0, 1.0, 1.0), r_max=3.0)
    assert g["cand_gap"].tolist() == [1]              # 有直接候选 → 不生成桥接边


def test_graph_bridge_marks_bridge_edges():
    dets = _straight_motion()
    ts = dets.t_range
    c02 = solve_jump_coupling(dets.centroid(0), dets.centroid(2), 2,
                              _ccfg(r_max=6.0), _mcfg(), dets.volume(0),
                              dets.volume(2), (1.0, 1.0, 1.0))
    g = build_graph(dets, 0, ts, None, c02,
                    GraphConfig(theta_gamma_frac=0.0, bridge=True, ctx_window=0),
                    _mcfg(), spacing=(1.0, 1.0, 1.0), r_max=6.0)
    assert g["cand_gap"].tolist() == [2]
    assert g["edge_feat"][g["is_target"], 9].max() == 1.0     # bridge 标志位


# ---------------------------------------------------------------------------
# §1.6 重建（OT 规则 + 桥接）
# ---------------------------------------------------------------------------

def test_volume_consistency_criterion():
    assert volume_consistent(100.0, [50.0, 50.0], 0.5)
    assert not volume_consistent(100.0, [50.0, 10.0], 0.2)


def test_ot_rule_reconstruct_links_simple_motion():
    dets = _dets(volumes=[[100.0], [100.0], [100.0]],
                 cents=[[[4, 5, 5]], [[4, 6, 6]], [[4, 7, 7]]])
    ts = dets.t_range
    couplings = {0: solve_coupling(dets.centroid(0), dets.centroid(1), dets.volume(0),
                                   dets.volume(1), None, _ccfg(), _mcfg(),
                                   (1.0, 1.0, 1.0)),
                 1: solve_coupling(dets.centroid(1), dets.centroid(2), dets.volume(1),
                                   dets.volume(2), None, _ccfg(), _mcfg(),
                                   (1.0, 1.0, 1.0))}
    cfg = PipelineConfig().reconstruct
    res = reconstruct_from_ot(dets, couplings, cfg, r_max=3.0)
    assert res.n_tracks() == 1
    assert res.tracks[1].begin == 0 and res.tracks[1].end == 2


def test_ot_rule_bridge_crosses_hole():
    dets = _straight_motion()                       # 中间帧缺失
    ts = dets.t_range
    jump = {0: solve_jump_coupling(dets.centroid(0), dets.centroid(2), 2,
                                   _ccfg(r_max=6.0), _mcfg(), dets.volume(0),
                                   dets.volume(2), (1.0, 1.0, 1.0))}
    cfg = PipelineConfig().reconstruct
    res = reconstruct_from_ot(dets, {}, cfg, r_max=6.0, jump=jump, bridge=True)
    assert res.n_tracks() == 1                       # 轨迹跨过空洞（不截断）
    assert res.tracks[next(iter(res.tracks))].begin == 0
    holes = holes_of(res.assignment)
    assert list(holes.values()) == [[1]]             # 空洞帧被登记


def test_edge_path_uses_bridge_only_when_direct_missing():
    dets = _dets(volumes=[[100.0], [100.0], [100.0]],
                 cents=[[[4, 5, 5]], [[4, 6, 6]], [[4, 7, 7]]])
    cfg = PipelineConfig().reconstruct
    # 直接边与桥接边都被判为高分 → 直接边优先（Δt=1 胜出）
    dec = {0: {"pairs": np.array([[0, 0], [0, 0]]),
               "score": np.array([0.9, 0.9]),
               "gap": np.array([1, 2])}}
    res = reconstruct_from_edges(dets, dec, cfg)
    # Δt=1 的直接边胜出：帧 1 继承帧 0 的 id；帧 2 没有入边 → 新轨迹；
    # 因为用不到桥接，所以不该出现空洞。
    assert res.assignment[1][0] == res.assignment[0][0]
    assert res.assignment[2][0] != res.assignment[0][0]
    assert not holes_of(res.assignment)


def test_edge_path_bridges_when_direct_rejected():
    dets = _straight_motion()
    cfg = PipelineConfig().reconstruct
    dec = {0: {"pairs": np.array([[0, 0]]), "score": np.array([0.9]),
               "gap": np.array([2])}}
    res = reconstruct_from_edges(dets, dec, cfg)
    assert res.n_tracks() == 1
    assert holes_of(res.assignment) == {1: [1]}


def test_hole_policy_split_cuts_the_track():
    dets = _straight_motion()
    cfg = PipelineConfig().reconstruct
    cfg.hole_policy = "split"
    dec = {0: {"pairs": np.array([[0, 0]]), "score": np.array([0.9]),
               "gap": np.array([2])}}
    res = reconstruct_from_edges(dets, dec, cfg)
    assert res.n_tracks() == 2                       # CTC 合法但被截断
    assert not holes_of(res.assignment)


# ---------------------------------------------------------------------------
# §2.0.1 GNN（式30-35）
# ---------------------------------------------------------------------------

def test_gnn_forward_and_train_smoke(tmp_path):
    import torch

    from papertrack.gnn.model import EdgeGNN, ModelConfig
    from papertrack.gnn.train import TrainConfig, train
    from papertrack.graph.build import build_dataset

    dets = _dets(volumes=[[100.0, 90.0], [100.0, 90.0], [100.0, 90.0]],
                 cents=[[[4, 5, 5], [4, 20, 20]],
                        [[4, 6, 6], [4, 21, 21]],
                        [[4, 7, 7], [4, 22, 22]]],
                 gt=[[3, 4], [3, 4], [3, 4]])
    ts = dets.t_range
    couplings, jump = {}, {}
    for pos, (t, tn) in enumerate(zip(ts[:-1], ts[1:])):
        couplings[pos] = solve_coupling(dets.centroid(t), dets.centroid(tn),
                                        dets.volume(t), dets.volume(tn), None,
                                        _ccfg(), _mcfg(), (1.0, 1.0, 1.0))
        if t + 2 <= ts[-1]:
            jump[t] = solve_jump_coupling(dets.centroid(t), dets.centroid(t + 2), 2,
                                          _ccfg(r_max=6.0), _mcfg(), dets.volume(t),
                                          dets.volume(t + 2), (1.0, 1.0, 1.0))
    stats = build_dataset(dets, couplings, jump,
                          GraphConfig(theta_gamma_frac=0.0), _mcfg(),
                          tmp_path / "graphs", gt_parent={}, spacing=(1.0, 1.0, 1.0),
                          r_max=3.0)
    assert stats["n_graphs"] >= 1

    model = EdgeGNN(ModelConfig(node_dim=7, edge_dim=EDGE_DIM))
    g = build_graph(dets, 0, ts, couplings[0], jump.get(0),
                    GraphConfig(theta_gamma_frac=0.0), _mcfg(),
                    spacing=(1.0, 1.0, 1.0), r_max=3.0)
    logits = model(torch.from_numpy(g["node_feat"]),
                   torch.from_numpy(g["edge_index"]),
                   torch.from_numpy(g["edge_feat"]))
    assert logits.shape == (g["edge_feat"].shape[0],)

    res = train(tmp_path / "graphs",
                TrainConfig(epochs=2, batch_pairs=1, val_fraction=0.0,
                            out_dir=str(tmp_path / "run")))
    assert (tmp_path / "run" / "best.pt").exists()
    assert res["history"][-1]["f1"] >= 0.0


# ---------------------------------------------------------------------------
# 端到端（合成 h5 → pipeline → CTC 导出 → 格式校验）
# ---------------------------------------------------------------------------

def _write_h5(path: Path, dets: Detections, vol_shape=(16, 32, 32)) -> Path:
    import h5py

    with h5py.File(path, "w") as f:
        f.attrs["shape"] = np.asarray(vol_shape, dtype=np.int32)
        f.attrs["spacing_zyx"] = np.asarray([1.0, 1.0, 1.0], dtype=np.float32)
        f.attrs["seq"] = "01"
        f.attrs["name"] = "synthetic"
        fr = f.create_group("frames")
        for t in dets.t_range:
            g = fr.create_group(f"{t:04d}")
            lab = np.zeros(vol_shape, dtype=np.uint16)
            for k, c in enumerate(dets.centroid(t)):
                z, y, x = (int(round(v)) for v in c)
                lab[max(z - 1, 0):z + 1, max(y - 1, 0):y + 1,
                    max(x - 1, 0):x + 1] = k + 1
            g.create_dataset("labels", data=lab)
            d = dets.frames[t]
            for key in ("label", "centroid", "volume", "intensity_mean",
                        "intensity_std"):
                if key in d:
                    g.create_dataset(key, data=np.asarray(d[key]))
    return path


def test_end_to_end_synthetic(tmp_path):
    from papertrack.config import PipelineConfig as PC
    from papertrack.reconstruction import export_ctc
    from papertrack.runtime import run_pipeline
    from papertrack.runtime.validate import validate_ctc_dir

    dets = _dets(volumes=[[100.0], [], [100.0]],
                 cents=[[[4, 5, 5]], [], [[4, 9, 9]]])
    h5 = _write_h5(tmp_path / "seq.h5", dets)
    cfg = PC()
    cfg.detection_source = "synthetic"
    cfg.measure = _mcfg()
    cfg.coupling = _ccfg(r_max=6.0)
    cfg.multiscale.enabled = False
    cfg.tracklet.enabled = False
    cfg.motion.enabled = False
    cfg.graph.theta_gamma_frac = 0.0

    run = run_pipeline(h5, cfg)
    assert run.spacing == (1.0, 1.0, 1.0)
    assert run.result.n_tracks() == 1
    assert run.info["holes"]["n_hole_frames"] == 1

    res_dir = tmp_path / "01_RES"
    stats = export_ctc(run, h5, res_dir, seq="01")
    assert stats["hole_frames_filled"] == 1
    check = validate_ctc_dir(res_dir)
    assert check["ok"], check["errors"]
