#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
# 复现命令（自动记录）
/home/henry/ot_idea/CellTracker/scripts/eval_pipeline.py --h5 data/interim/Fluo-N3DH-CE_01.h5 --dataset Fluo-N3DH-CE --seq 01 --exp-id B2b_fgw_eval --config /home/henry/ot_idea/CellTracker/configs/pipeline_default.yaml --ablate fgw --ckpt /home/henry/ot_idea/CellTracker/experiments/B2b_fgw_train/artifacts/model/best.pt --official
