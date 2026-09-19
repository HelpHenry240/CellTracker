#!/usr/bin/env python3
"""B2 消融矩阵驱动：对每个消融项自动完成
「重建图 → 重训 GNN → 官方评测 → 汇总成表」。

为什么每项都要重训：本 pipeline 里几乎每个模块都会改变 GNN 的**输入分布**
（候选边集合或边特征），若沿用原模型，测到的是"分布漂移"而不是模块贡献。

用法::

    python scripts/run_ablation_matrix.py --h5 data/interim/Fluo-N3DH-CE_01.h5 \
        --dataset Fluo-N3DH-CE --seq 01 \
        --ablations ot_cand,cand_topk,fgw,unbalanced,motion \
        --epochs 60 --official
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from celltracker.experiment import Experiment  # noqa: E402
from celltracker.pipeline import (PipelineConfig, apply_ablation, load_config,  # noqa: E402
                                  run_pipeline, save_config)

PY = sys.executable


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--ablations", default="",
                    help="逗号分隔的“关掉”实验，如 ot_cand,cand_topk")
    ap.add_argument("--experiments", default="",
                    help="“打开/改值”实验，格式 name:key=val,key=val;name2:... "
                         "例如 fgw:ot.eta=0.3;unbalanced:ot.tau_a=1.0,ot.tau_b=1.0")
    ap.add_argument("--exp-prefix", default="B2")
    ap.add_argument("--config", default=str(ROOT / "configs" / "pipeline_default.yaml"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--frames", default=None)
    ap.add_argument("--official", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    frames = None
    if args.frames:
        lo, _, hi = args.frames.partition(":")
        frames = list(range(int(lo), int(hi) + 1))

    base = load_config(args.config)
    names = [s for s in args.ablations.split(",") if s]
    # 解析 --experiments（每个实验一组 --set 覆盖）
    exp_specs: list[tuple[str, list[str]]] = []
    for spec in [s for s in args.experiments.split(";") if s.strip()]:
        name, _, kv = spec.partition(":")
        exp_specs.append((name.strip(), [k for k in kv.split(",") if k]))
    if not names and not exp_specs:
        raise SystemExit("需要 --ablations 或 --experiments 至少一项")
    exp = Experiment(f"{args.exp_prefix}_ablation_matrix",
                     purpose="消融矩阵（每项重建图+重训GNN+官方评测）",
                     params={"h5": args.h5, "ablations": names,
                             "epochs": args.epochs, "frames": args.frames})
    save_config(base, exp.dir / "base_config.yaml")

    rows = []
    jobs = [(n, None) for n in names] + exp_specs
    for name, overrides in jobs:
        train_id = f"{args.exp_prefix}_{name}_train"
        eval_id = f"{args.exp_prefix}_{name}_eval"
        ckpt = ROOT / "experiments" / train_id / "artifacts" / "model" / "best.pt"
        if args.skip_existing and (ROOT / "experiments" / eval_id /
                                   f"metrics_official_{args.seq}.json").exists():
            exp.log(f"[{name}] 已存在，跳过")
            continue

        if overrides:
            from run_pipeline import _set_dotted
            cfg = _set_dotted(base, overrides)
        else:
            cfg = apply_ablation(base, {name})
        graph_dir = ROOT / "data" / "interim" / f"graphs_{args.seq}_{name}"
        exp.log(f"[{name}] 1/3 重建图 → {graph_dir}")
        run_pipeline(args.h5, cfg, frames=frames, dump_graphs=graph_dir)

        exp.log(f"[{name}] 2/3 重训 GNN → {train_id}")
        proc = subprocess.run(
            [PY, str(ROOT / "scripts" / "run_gnn.py"), "train",
             "--graphs", str(graph_dir), "--exp-id", train_id,
             "--epochs", str(args.epochs)],
            capture_output=True, text=True)
        if proc.returncode != 0:
            exp.log(f"[{name}] 训练失败: {proc.stderr[-500:]}")
            continue
        train_metrics = json.loads((ROOT / "experiments" / train_id /
                                    "metrics.json").read_text())

        exp.log(f"[{name}] 3/3 评测 → {eval_id}")
        cmd = [PY, str(ROOT / "scripts" / "eval_pipeline.py"),
               "--h5", args.h5, "--dataset", args.dataset, "--seq", args.seq,
               "--exp-id", eval_id, "--config", args.config,
               "--ckpt", str(ckpt)]
        if overrides:
            cmd += ["--set"] + overrides
        else:
            cmd += ["--ablate", name]
        if args.frames:
            cmd += ["--frames", args.frames]
        if args.official:
            cmd += ["--official"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        off_path = (ROOT / "experiments" / eval_id /
                    f"metrics_official_{args.seq}.json")
        tra = det = None
        if off_path.exists():
            off = json.loads(off_path.read_text())
            tra, det = off.get("TRA"), off.get("DET")
        loc_path = ROOT / "experiments" / eval_id / "metrics.json"
        loc = json.loads(loc_path.read_text()) if loc_path.exists() else {}
        row = {"ablation": name,
               "overrides": ",".join(overrides) if overrides else f"ablate:{name}",
               "train_experiment": train_id,
               "eval_experiment": eval_id, "checkpoint": str(ckpt.relative_to(ROOT)),
               "best_move_f1": round(train_metrics.get("best_move_f1", float("nan")), 4),
               "best_div_f1": round(train_metrics.get("best_div_f1", float("nan")), 4),
               "tracks": loc.get("n_tracks_pred"), "IDsw": loc.get("id_switches"),
               "frag": loc.get("fragmentation"), "DET": det, "TRA": tra}
        rows.append(row)
        exp.log(f"[{name}] 结果: {row}")

    # 汇总表
    keys = ["ablation", "overrides", "train_experiment", "best_move_f1", "best_div_f1",
            "tracks", "IDsw", "frag", "DET", "TRA"]
    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(k, "—")) for k in keys) + " |")
    table = "\n".join(lines)
    (exp.dir / "ablation_table.md").write_text(table + "\n")
    exp.save_metrics({"rows": rows})
    exp.log("\n" + table)
    exp.finish(summary=f"{len(rows)} 项消融完成")


if __name__ == "__main__":
    main()
