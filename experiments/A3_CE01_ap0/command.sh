#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
# 复现命令（自动记录）
scripts/run_ot.py --h5 data/interim/Fluo-N3DH-CE_01.h5 --dataset Fluo-N3DH-CE --seq 01 --exp-id A3_CE01_ap0 --eps 1.0 --no-eps-rel --eta 0 --r-max 30 --div-ratio 0.3
