#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
# 复现命令（自动记录）
scripts/eval_pipeline.py --h5 data/interim/Fluo-N3DH-CE_01.h5 --dataset Fluo-N3DH-CE --seq 01 --exp-id B2_otrule_div0.5 --frames 120:190 --set reconstruct.div_ratio=0.5
