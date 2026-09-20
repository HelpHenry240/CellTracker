#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
# 复现命令（自动记录）
scripts/run_baseline.py --h5 data/interim/Fluo-N3DH-CE_01_pred.h5 --dataset Fluo-N3DH-CE --seq 01 --exp-id C4_pred_hungarian --method hungarian
