#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
# 复现命令（自动记录）
scripts/run_ot.py --h5 data/interim/Fluo-N3DH-CE_01.h5 --dataset Fluo-N3DH-CE --seq 01 --exp-id E2.5c_CE01_ot_eps0.3 --eps 0.3 --eta 0.0 --r-max 30 --theta-gamma 0.05 --div-ratio 0.35 --div-sum-min 0.9 --official
