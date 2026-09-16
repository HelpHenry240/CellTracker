#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
# 复现命令（自动记录）
scripts/run_sweep.py --h5 data/interim/Fluo-N3DH-CE_01.h5 --dataset Fluo-N3DH-CE --seq 01 --exp-id E2.5_CE01_theta_div_sweep --x theta_gamma=0.05,0.1,0.2 --y div_ratio=0.15,0.3 --fixed eps=1.0,eta=0,r_max=30
