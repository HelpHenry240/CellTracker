#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
# 复现命令（自动记录）
scripts/run_ot.py --h5 data/interim/Fluo-N3DH-CE_01.h5 --dataset Fluo-N3DH-CE --seq 01 --exp-id A2_baseline_eta0_tauNone --eps 1.0 --eta 0 --r-max 30 --theta-gamma 0.05 --div-ratio 0.3
