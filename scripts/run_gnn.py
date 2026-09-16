#!/usr/bin/env python3
"""P4：动态图 GNN 的构建 / 训练 / 推理 / 评测一条龙。

示例::

    # 1) 构图
    python scripts/run_gnn.py build --h5 data/interim/Fluo-N3DH-CE_01.h5 \
        --out data/interim/graphs_ce01
    # 2) 训练
    python scripts/run_gnn.py train --graphs data/interim/graphs_ce01 \
        --exp-id E4.1_gnn_ce01 --epochs 40
    # 3) 推理 + 评测（本地指标，可选官方）
    python scripts/run_gnn.py infer --graphs data/interim/graphs_ce01 \
        --ckpt experiments/E4.1_gnn_ce01/best.pt --h5 data/interim/Fluo-N3DH-CE_01.h5 \
        --dataset Fluo-N3DH-CE --seq 01 --exp-id E4.2_gnn_ce01_infer --official
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_baseline import _plot_diagnostics, _plot_tracks, paint_and_write_stream  # noqa: E402

from celltracker.eval.local_metrics import seg_measure  # noqa: E402
from celltracker.experiment import Experiment  # noqa: E402
from celltracker.gnn.infer import InferConfig, predict_pairs, reconstruct_tracks  # noqa: E402
from celltracker.gnn.train import TrainConfig, train  # noqa: E402
from celltracker.graph import GraphConfig, build_dataset  # noqa: E402
from celltracker.track import Detections  # noqa: E402


def cmd_build(args) -> None:
    exp = Experiment(args.exp_id, purpose="P4 构图",
                     params={"h5": args.h5, "r_max": args.r_max, "eta": args.eta})
    cfg = GraphConfig(r_max=args.r_max, knn=args.knn, use_ot=True, eta=args.eta,
                      eps=args.eps)
    build_dataset(args.h5, args.out, cfg)
    stats = {"pairs": len(list(Path(args.out).glob("pair_*.npz")))}
    labels = np.zeros(3, dtype=np.int64)
    for f in Path(args.out).glob("pair_*.npz"):
        labels += np.bincount(np.load(f)["cand_label"], minlength=3)
    stats["label_counts"] = {"none": int(labels[0]), "move": int(labels[1]),
                             "division": int(labels[2])}
    stats["pos_rate_move"] = float(labels[1] / max(labels.sum(), 1))
    stats["pos_rate_div"] = float(labels[2] / max(labels.sum(), 1))
    exp.save_metrics(stats)
    exp.log(str(stats))
    exp.finish()


def cmd_train(args) -> None:
    exp = Experiment(args.exp_id, purpose="P4 GNN 训练",
                     params={"graphs": args.graphs, "epochs": args.epochs,
                             "hidden": args.hidden, "layers": args.layers,
                             "lambda_ot": args.lambda_ot})
    cfg = TrainConfig(epochs=args.epochs, hidden=args.hidden, layers=args.layers,
                      dropout=args.dropout, lambda_ot=args.lambda_ot,
                      batch_pairs=args.batch_pairs, lr=args.lr, device=args.device,
                      out_dir=str(exp.dir / "artifacts" / "model"))
    result = train(args.graphs, cfg)
    hist = result["history"]
    exp.save_metrics({"final": hist[-1], "best_move_f1": max(h["f1_move"] for h in hist),
                      "best_div_f1": max(h["f1_div"] for h in hist), "history": hist})
    _plot_gnn_history(exp, hist)
    exp.finish(summary=f"best move_f1={max(h['f1_move'] for h in hist):.3f} "
                       f"div_f1={max(h['f1_div'] for h in hist):.3f}")


def cmd_infer(args) -> None:
    exp = Experiment(args.exp_id, purpose="P4 GNN 推理与评测",
                     params={"ckpt": args.ckpt, "graphs": args.graphs,
                             "tau_move": args.tau_move, "tau_div": args.tau_div,
                             "official_eval": args.official})
    dets = Detections.from_h5(args.h5)
    ts = dets.t_range
    preds = predict_pairs(args.ckpt, args.graphs, device=args.device)
    exp.log(f"预测完成: {len(preds)} 对帧")
    result = reconstruct_tracks(dets, preds,
                                InferConfig(tau_move=args.tau_move, tau_div=args.tau_div))
    exp.log(f"重建轨迹 {result.n_tracks()} 条")

    res_dir = exp.artifact_dir("submission") / f"{args.seq}_RES"
    diag, mip = paint_and_write_stream(Path(args.h5), ts, dets, result.assignment,
                                       res_dir, result.tracks,
                                       want_mip_frame=ts[len(ts) // 2])
    gt_seg_dir = Path("data/raw") / args.dataset / f"{args.seq}_GT" / "SEG"
    sego = seg_measure(gt_seg_dir, res_dir) if gt_seg_dir.is_dir() else float("nan")
    with h5py.File(args.h5, "r") as f:
        gt_tracks = np.asarray(f["tracks"]) if "tracks" in f else np.zeros(0, dtype=[("parent", "i4")])
    metrics = {**diag, "experiment": args.exp_id, "dataset": args.dataset,
               "seq": args.seq, "n_tracks_pred": result.n_tracks(),
               "gt_tracks": {"n_tracks": int(len(gt_tracks)),
                             "n_divisions": int(np.count_nonzero(gt_tracks["parent"] > 0))},
               "local_SEG": sego, "tau_move": args.tau_move, "tau_div": args.tau_div}
    exp.save_metrics(metrics)
    exp.log(f"matched={diag['matched']} IDsw={diag['id_switches']} "
            f"frag={diag['fragmentation']} divP={diag.get('division_precision')} "
            f"divR={diag.get('division_recall')}")
    _plot_diagnostics(exp, result, diag, args.dataset, args.seq)
    _plot_tracks(exp, dets, result.assignment, args.dataset, args.seq, mip=mip)

    if args.official:
        import subprocess
        cmd = [sys.executable, str(ROOT / "scripts" / "cloud_eval.py"),
               "--res-dir", str(res_dir), "--dataset", args.dataset, "--seq", args.seq,
               "--out", str(exp.dir / f"metrics_official_{args.seq}.json")]
        exp.log("官方指标: " + " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        exp.log((proc.stdout or "")[-1500:])
        if proc.returncode != 0:
            exp.log("官方指标失败: " + (proc.stderr or "")[-800:])
    exp.finish(summary=f"tracks={result.n_tracks()} IDsw={diag['id_switches']} "
                       f"frag={diag['fragmentation']}")


def _plot_gnn_history(exp: Experiment, hist: list[dict]) -> None:
    import matplotlib.pyplot as plt
    from celltracker.viz import PALETTE, savefig

    fig, axes = plt.subplots(1, 3, figsize=(11, 2.9))
    ep = [h["epoch"] for h in hist]
    axes[0].plot(ep, [h["train_loss"] for h in hist], color=PALETTE["blue"])
    axes[0].set_title("train loss"); axes[0].set_xlabel("epoch")
    axes[1].plot(ep, [h["f1_move"] for h in hist], color=PALETTE["green"], label="move")
    axes[1].plot(ep, [h["f1_div"] for h in hist], color=PALETTE["red"], label="division")
    axes[1].set_title("validation F1"); axes[1].set_xlabel("epoch"); axes[1].legend()
    axes[2].plot(ep, [h["accuracy"] for h in hist], color=PALETTE["purple"])
    axes[2].set_title("validation accuracy"); axes[2].set_xlabel("epoch")
    savefig(fig, exp.figure_path("training_history"))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--h5", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--exp-id", required=True)
    b.add_argument("--r-max", type=float, default=30.0)
    b.add_argument("--knn", type=int, default=4)
    b.add_argument("--eps", type=float, default=1.0)
    b.add_argument("--eta", type=float, default=0.0)
    b.set_defaults(func=cmd_build)

    t = sub.add_parser("train")
    t.add_argument("--graphs", required=True)
    t.add_argument("--exp-id", required=True)
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--hidden", type=int, default=64)
    t.add_argument("--layers", type=int, default=3)
    t.add_argument("--dropout", type=float, default=0.1)
    t.add_argument("--lambda-ot", type=float, default=0.1)
    t.add_argument("--batch-pairs", type=int, default=8)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--device", default="cpu")
    t.set_defaults(func=cmd_train)

    i = sub.add_parser("infer")
    i.add_argument("--graphs", required=True)
    i.add_argument("--ckpt", required=True)
    i.add_argument("--h5", required=True)
    i.add_argument("--dataset", required=True)
    i.add_argument("--seq", required=True)
    i.add_argument("--exp-id", required=True)
    i.add_argument("--tau-move", type=float, default=0.5)
    i.add_argument("--tau-div", type=float, default=0.5)
    i.add_argument("--device", default="cpu")
    i.add_argument("--official", action="store_true")
    i.set_defaults(func=cmd_infer)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
