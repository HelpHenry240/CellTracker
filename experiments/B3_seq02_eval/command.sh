#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
# 复现命令（自动记录）
scripts/eval_pipeline.py --h5 data/interim/Fluo-N3DH-CE_02.h5 --dataset Fluo-N3DH-CE --seq 02 --exp-id B3_seq02_eval --ckpt experiments/B3_seq02_train/artifacts/model/best.pt --set tracklet.enabled=true tracklet.max_gap=3 --official
