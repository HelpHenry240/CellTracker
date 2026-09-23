#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
# 复现命令（自动记录）
scripts/eval_pipeline.py --h5 data/interim/Fluo-N3DH-CE_01_pred_v2rs13.h5 --gt-h5 data/interim/Fluo-N3DH-CE_01.h5 --dataset Fluo-N3DH-CE --seq 01 --exp-id C5.0f_k13_official --ckpt experiments/C5.0f_k13_train/artifacts/model/best.pt --official
