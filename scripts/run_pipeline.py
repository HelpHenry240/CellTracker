#!/usr/bin/env python3
"""统一 pipeline 入口：按论文顺序跑完整链路（含消融开关）。

示例::

    # 默认：OT 规则重建（GNN 关闭），带多尺度与第二层 tracklet
    python scripts/run_pipeline.py --h5 data/interim/Fluo-N3DH-CE_01.h5 \
        --exp-id A6_full_ot_rule --frames 120:190 \
        --set ot.eta=0.3 ot.tau_a=1.0 ot.tau_b=1.0 tracklet.enabled=true \
        --config configs/pipeline_default.yaml

    # 消融：关掉 FGW 与多尺度
    python scripts/run_pipeline.py --h5 ... --exp-id A6_ablate \
        --ablate fgw,multiscale
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from celltracker.experiment import Experiment  # noqa: E402
from celltracker.pipeline import (PipelineConfig, apply_ablation, load_config,  # noqa: E402
                                  run_pipeline, save_config)


def _coerce(v: str):
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    if v.lower() in ("none", "null"):
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


def _set_dotted(cfg: PipelineConfig, pairs: list[str]) -> PipelineConfig:
    """按 `段.字段=值` 覆盖配置（如 `ot.eta=0.3`）。"""
    from dataclasses import replace

    for kv in pairs:
        key, _, val = kv.partition("=")
        if not _:
            raise SystemExit(f"参数格式应为 段.字段=值，收到: {kv}")
        section, _, field = key.partition(".")
        if not field or not hasattr(cfg, section):
            raise SystemExit(f"未知配置段: {key}")
        cur = getattr(cfg, section)
        if not hasattr(cur, field):
            raise SystemExit(f"未知字段: {key}（可用: {list(vars(cur))}）")
        cfg = replace(cfg, **{section: replace(cur, **{field: _coerce(val)})})
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--dataset", default="Fluo-N3DH-CE")
    ap.add_argument("--seq", default="01")
    ap.add_argument("--frames", default=None,
                    help="帧范围，如 120:190（便于快速验证）")
    ap.add_argument("--config", default=None, help="从 yaml 载入基础配置")
    ap.add_argument("--set", nargs="*", default=[], help="覆盖配置，如 ot.eta=0.3")
    ap.add_argument("--ablate", default="", help="逗号分隔的消融项")
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else PipelineConfig()
    if args.set:
        cfg = _set_dotted(cfg, args.set)
    if args.ablate:
        cfg = apply_ablation(cfg, [s for s in args.ablate.split(",") if s])

    frames = None
    if args.frames:
        lo, _, hi = args.frames.partition(":")
        frames = list(range(int(lo), int(hi) + 1))

    exp = Experiment(args.exp_id, purpose="pipeline 全链路运行",
                     params={"h5": args.h5, "frames": args.frames,
                             "ablate": args.ablate})
    save_config(cfg, exp.dir / "pipeline_config.yaml")
    run = run_pipeline(args.h5, cfg, frames=frames,
                       artifacts_dir=exp.artifact_dir("pipeline"))
    exp.log(f"pipeline info: {run.info}")
    exp.save_metrics({"info": run.info, "n_tracks": run.track_result.n_tracks()})
    exp.finish(summary=f"tracks={run.track_result.n_tracks()} "
                       f"decision={run.info.get('decision')}")


if __name__ == "__main__":
    main()
