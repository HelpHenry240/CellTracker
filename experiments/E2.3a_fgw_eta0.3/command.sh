#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
# 复现命令（自动记录）
scripts/run_ot.py --h5 data/interim/Fluo-N3DH-CHO_01.h5 --dataset Fluo-N3DH-CHO --seq 01 --exp-id E2.3a_fgw_eta0.3 --eps 1.0 --eta 0.3 --r-max 30 --official
