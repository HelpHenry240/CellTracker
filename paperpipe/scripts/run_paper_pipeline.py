#!/usr/bin/env python3
"""严格论文口径 pipeline 的入口（paperpipe）。

示例::

    # ① 只用 OT 规则（式23/24）跑通链路 + 落盘图数据集（训练 GNN 用）
    python paperpipe/scripts/run_paper_pipeline.py \
        --h5 data/interim/Fluo-N3DH-CE_01_pred_v2.h5 \
        --gt-h5 data/interim/Fluo-N3DH-CE_01.h5 \
        --seq 01 --exp-id P1_otrule_ce01 \
        --dump-graphs data/interim/pg_graphs_01

    # ② 用训练好的 GNN（§2.0.1 式33）决策 + 导出 CTC 结果 + 官方指标
    python paperpipe/scripts/run_paper_pipeline.py \
        --h5 ... --gt-h5 ... --seq 01 --exp-id P2_gnn_ce01 \
        --ckpt paperpipe/runs/gnn/best.pt --official

产物：experiments/<exp-id>/ 八件套；`artifacts/submission/<seq>_RES/` 为 CTC 提交目录。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "paperpipe" / "src"))
sys.path.insert(0, str(ROOT / "src"))

from celltracker.experiment import Experiment                      # noqa: E402
from papertrack.config import load_config, override, save_config   # noqa: E402
from papertrack.pipeline import export_ctc, run_pipeline           # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True, help="检测来源 h5（nnU-Net 预测实例）")
    ap.add_argument("--gt-h5", default=None, help="GT h5（提供血缘与评测真值）")
    ap.add_argument("--seq", required=True)
    ap.add_argument("--dataset", default="Fluo-N3DH-CE")
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--config", default=str(ROOT / "paperpipe" / "configs" /
                                            "paper_default.yaml"))
    ap.add_argument("--set", nargs="*", default=[], help="覆盖配置：段.字段=值")
    ap.add_argument("--ckpt", default=None, help="GNN 权重；不给则走 §1.6 OT 规则")
    ap.add_argument("--frames", default=None, help="帧范围（冒烟用），如 120:190")
    ap.add_argument("--dump-graphs", default=None, help="落盘图数据集目录（训练用）")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--official", action="store_true", help="云端官方 DET/SEG/TRA")
    ap.add_argument("--cloud-gt-root", default=None)
    ap.add_argument("--no-validate", action="store_true",
                    help="跳过 CTC 提交格式校验（默认执行，E2 要求）")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.set:
        cfg = override(cfg, args.set)

    frames = None
    if args.frames:
        lo, _, hi = args.frames.partition(":")
        frames = list(range(int(lo), int(hi) + 1))

    exp = Experiment(args.exp_id, purpose="paperpipe 严格论文口径 pipeline",
                     params={"h5": args.h5, "gt_h5": args.gt_h5, "seq": args.seq,
                             "ckpt": args.ckpt, "frames": args.frames,
                             "overrides": args.set})
    save_config(cfg, exp.dir / "pipeline_config.yaml")
    exp.log("配置已存 pipeline_config.yaml")

    run = run_pipeline(args.h5, cfg, gt_h5=args.gt_h5, ckpt=args.ckpt,
                       frames=frames, artifacts_dir=exp.artifact_dir("pipeline"),
                       dump_graphs=args.dump_graphs, device=args.device)
    exp.log(f"pipeline info: {json.dumps(run.info, ensure_ascii=False, default=str)}")

    # ---- 本地格式校验（E2：送官方评测前必须跑）----
    res_dir = exp.artifact_dir("submission") / f"{args.seq}_RES"
    stats = export_ctc(run, args.h5, res_dir, seq=args.seq,
                       gt_h5=args.gt_h5,
                       gt_seg_dir=(ROOT / "data" / "raw" / args.dataset /
                                   f"{args.seq}_GT" / "SEG"))
    exp.log(f"本地指标: {json.dumps(stats, ensure_ascii=False, default=str)}")

    if not args.no_validate:
        from papertrack.validate import validate_ctc_dir

        check = validate_ctc_dir(res_dir)
        stats["format_validation"] = check
        exp.log(f"CTC 格式校验: {json.dumps(check, ensure_ascii=False)}")
        if not check["ok"]:
            exp.log("⚠️ 格式校验未通过 → 不送官方评测（先修数据格式问题）")
            args.official = False
    exp.save_metrics(stats)

    if args.official:
        import subprocess

        cmd = [sys.executable, str(ROOT / "scripts" / "cloud_eval.py"),
               "--res-dir", str(res_dir), "--dataset", args.dataset,
               "--seq", args.seq,
               "--out", str(exp.dir / f"metrics_official_{args.seq}.json")]
        if args.cloud_gt_root:
            cmd += ["--cloud-gt-root", args.cloud_gt_root]
        exp.log("官方指标: " + " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        exp.log((proc.stdout or "")[-1500:])
        if proc.returncode != 0:
            exp.log("官方指标失败: " + (proc.stderr or "")[-800:])

    exp.finish(summary=f"tracks={run.result.n_tracks()} "
                       f"decision={run.info.get('decision')} "
                       f"holes={run.info['holes']['n_hole_frames']}")


if __name__ == "__main__":
    main()
