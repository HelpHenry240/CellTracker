#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
# 复现命令（自动记录）
scripts/run_sweep.py --h5 data/interim/Fluo-N3DH-CHO_01.h5 --dataset Fluo-N3DH-CHO --seq 01 --exp-id E2.3b_eta_eps_sweep_cho --x eta=0,0.3 --y eps=1,5
