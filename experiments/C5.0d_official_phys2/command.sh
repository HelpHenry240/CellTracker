#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
# 复现命令（自动记录）
scripts/eval_pipeline.py --h5 data/interim/Fluo-N3DH-CE_01_pred_v2.h5 --gt-h5 data/interim/Fluo-N3DH-CE_01.h5 --dataset Fluo-N3DH-CE --seq 01 --exp-id C5.0d_official_phys2 --set ot.spacing_zyx=1.0,0.09,0.09 ot.r_max=3.0 --ckpt experiments/C5.0d_phys_units_train/artifacts/model/best.pt --official
