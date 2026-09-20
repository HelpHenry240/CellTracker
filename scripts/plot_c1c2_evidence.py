#!/usr/bin/env python3
"""Phase C 留痕图：把 C1（推理）/ C2（实例拆分）/ C5.0（间距修复重标定）的证据画成论文可用图。

输入都是**已在本地留存的中间产物**，不重跑推理：
  - `data/interim/pred_report_01.json`：逐帧实例数 / GT 标记数 / 召回 / 精确 / 独立性
  - `experiments/C2_instance_split/metrics*.json`：标定扫描表

输出：
  - `experiments/C1_nnunet_infer/figures/instances_vs_markers.{png,pdf}`
  - `experiments/C1_nnunet_infer/metrics.json`
  - `experiments/C2_instance_split/figures/calibration_tradeoff.{png,pdf}`
  - `experiments/C5.0_spacing_fix/figures/spacing_recalibration.{png,pdf}`

用法::

    python scripts/plot_c1c2_evidence.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from celltracker.viz import PALETTE, savefig, setup  # noqa: E402

setup()
import matplotlib.pyplot as plt  # noqa: E402,F401


def plot_instance_profile(report_path: Path, fig_dir: Path, metrics_path: Path) -> dict:
    """C1 证据：逐帧实例数 vs GT 标记数 + 每实例覆盖细胞数。"""
    rep = json.loads(report_path.read_text())
    per = rep["per_frame"]
    t = np.array([r["t"] for r in per])
    n_pred = np.array([r["n_pred"] for r in per], dtype=float)
    n_gt = np.array([r["n_gt"] for r in per], dtype=float)
    recall = np.array([r["recall"] for r in per], dtype=float)
    excl = np.array([r["exclusive"] for r in per], dtype=float)
    ratio = n_pred / np.maximum(n_gt, 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    ax = axes[0]
    ax.plot(t, n_gt, color=PALETTE["grey"], lw=1.2, label="GT markers")
    ax.plot(t, n_pred, color=PALETTE["blue"], lw=1.2, label="predicted instances")
    ax.set_xlabel("frame index")
    ax.set_ylabel("objects per frame")
    ax.set_title("Detections vs GT markers (nnU-Net, seq01)")
    ax.legend(frameon=False)

    ax = axes[1]
    ax.plot(t, ratio, color=PALETTE["red"], lw=1.2, label="instances / marker")
    ax.axhline(1.0, color=PALETTE["grey"], ls="--", lw=1.0)
    ax.plot(t, excl, color=PALETTE["green"], lw=1.2, label="marker exclusivity")
    ax.set_xlabel("frame index")
    ax.set_ylim(0, 1.6)
    ax.set_title("Under-segmentation grows with density")
    ax.legend(frameon=False)
    savefig(fig, fig_dir / "instances_vs_markers")

    agg = rep["aggregate"]
    metrics = {
        "seq": "01",
        "n_frames": agg["n_frames"],
        "instances_per_frame_mean": agg["mean_instances"],
        "markers_per_frame_mean": agg["mean_gt_markers"],
        "instances_per_marker": agg["mean_instances"] / agg["mean_gt_markers"],
        "detection_recall": agg["detection_recall"],
        "detection_precision": agg["detection_precision"],
        "instance_config": agg["config"],
        "instances_per_marker_first_50_frames": float(ratio[t < 50].mean()),
        "instances_per_marker_last_50_frames": float(ratio[t >= t.max() - 49].mean()),
        "marker_exclusivity_last_50_frames": float(excl[t >= t.max() - 49].mean()),
        "source": str(report_path),
    }
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics


def plot_calibration_tradeoff(files: list[Path], fig_dir: Path) -> dict:
    """C2 证据：把"实例数/标记数"与 F1 放一起，暴露 F1 判据奖励欠分割。"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))

    ax = axes[0]
    fixed = json.loads(files[0].read_text())["sweep"]
    ratio = [r["instances_per_marker"] for r in fixed]
    f1 = [r["f1"] for r in fixed]
    ax.scatter(ratio, f1, s=28, color=PALETTE["blue"],
               label="fixed-scale seeds (grid)")
    for r in fixed:
        if r["min_distance"] in (3, 6, 10) and r["min_volume"] == 100:
            ax.annotate(f"md={r['min_distance']}",
                        (r["instances_per_marker"], r["f1"]),
                        textcoords="offset points", xytext=(4, -8), fontsize=7)
    ax.set_xlabel("instances per marker  (1.0 = correct count)")
    ax.set_ylabel("detection F1 vs markers")
    ax.set_title("F1 rewards under-segmentation")

    ax = axes[1]
    adaptive = json.loads(files[1].read_text())["sweep"]
    xs = [r["h_frac"] for r in adaptive]
    ax.plot(xs, [r["instances_per_marker"] for r in adaptive], "o-",
            color=PALETTE["red"], label="instances / marker")
    ax.plot(xs, [r["exclusive"] for r in adaptive], "s-",
            color=PALETTE["green"], label="marker exclusivity")
    ax.plot(xs, [r["f1"] for r in adaptive], "^-",
            color=PALETTE["blue"], label="detection F1")
    ax.axvline(0.35, color=PALETTE["grey"], ls="--", lw=1.0)
    ax.annotate("deployed\nh_frac=0.35", (0.35, 0.15), fontsize=7,
                color=PALETTE["grey"])
    ax.set_xlabel("h_frac (adaptive seed threshold)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Adaptive seeds: fewer instances, 'better' F1")
    ax.legend(frameon=False, fontsize=7)
    savefig(fig, fig_dir / "calibration_tradeoff")

    return {
        "fixed_scale_sweep": fixed,
        "adaptive_sweep": adaptive,
        "note": "F1 在欠分割时单调升高，不能作为实例拆分的目标函数（见 notes.md）",
    }


def main() -> None:
    c1_dir = ROOT / "experiments/C1_nnunet_infer"
    c2_dir = ROOT / "experiments/C2_instance_split"
    c5_dir = ROOT / "experiments/C5.0_spacing_fix"
    c1_metrics = plot_instance_profile(
        ROOT / "data/interim/pred_report_01.json",
        c1_dir / "figures", c1_dir / "metrics.json")
    c2_metrics = plot_calibration_tradeoff(
        [c2_dir / "metrics.json", c2_dir / "metrics_adaptive.json"], c2_dir / "figures")

    (c2_dir / "summary.json").write_text(
        json.dumps({
            "deployed_config": {"h_frac": 0.35, "min_volume": 100, "watershed": True,
                                "min_distance": "不生效（h_frac>0 时由 h-maxima 决定）"},
            "deployed_mean_instances_per_frame": c1_metrics["instances_per_frame_mean"],
            "deployed_mean_markers_per_frame": c1_metrics["markers_per_frame_mean"],
            "deployed_instances_per_marker": c1_metrics["instances_per_marker"],
            "n_configs_swept": len(c2_metrics["fixed_scale_sweep"])
            + len(c2_metrics["adaptive_sweep"]),
        }, indent=2, ensure_ascii=False), encoding="utf-8")

    if (c5_dir / "metrics_sweep.json").exists():
        plot_spacing_recalibration(
            [c5_dir / "metrics_bracket.json", c5_dir / "metrics_sweep.json"],
            c5_dir / "figures")
    print(json.dumps(c1_metrics, indent=2, ensure_ascii=False))


def plot_spacing_recalibration(files: list[Path], fig_dir: Path) -> None:
    """C5.0 证据：物理间距 EDT 之后，用"实例/标记 + exclusive"重新标定 h_frac。

    刻意**不**画 F1：C2 已证明 F1 在欠分割时单调升高（奖励错误方向）。
    """
    rows = []
    for p in files:
        if p.exists():
            rows.extend(json.loads(p.read_text())["sweep"])
    if not rows:
        return
    rows.sort(key=lambda r: r["h_frac"])
    hf = [r["h_frac"] for r in rows]
    ratio = [r["instances_per_marker"] for r in rows]
    excl = [r["exclusive"] for r in rows]
    prec = [r["precision"] for r in rows]

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot(hf, ratio, "o-", color=PALETTE["red"], label="instances / marker")
    ax.plot(hf, excl, "s-", color=PALETTE["green"], label="marker exclusivity")
    ax.plot(hf, prec, "^-", color=PALETTE["blue"], label="detection precision")
    ax.axhline(1.0, color=PALETTE["grey"], ls="--", lw=1.0)
    for x, y in zip(hf, ratio):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                    xytext=(0, 6), fontsize=7, color=PALETTE["red"], ha="center")
    ax.axvline(0.10, color=PALETTE["grey"], ls=":", lw=1.0)
    ax.annotate("chosen\nh_frac=0.10", (0.10, 1.12), fontsize=7,
                color=PALETTE["grey"])
    ax.axvline(0.35, color="k", ls=":", lw=1.0)
    ax.annotate("previously\ndeployed 0.35", (0.35, 0.30), fontsize=7)
    ax.set_xlabel("h_frac (h-maxima threshold, physical spacing EDT)")
    ax.set_ylim(0, 1.45)
    ax.set_title("Re-calibration after fixing EDT sampling (7 frames, seq01)")
    ax.legend(frameon=False, fontsize=7, loc="center right")
    savefig(fig, fig_dir / "spacing_recalibration")


if __name__ == "__main__":
    main()
