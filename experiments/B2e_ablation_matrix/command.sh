#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
# 复现命令（自动记录）
scripts/run_ablation_matrix.py --h5 data/interim/Fluo-N3DH-CE_01.h5 --dataset Fluo-N3DH-CE --seq 01 --experiments unbalanced_weak:ot.tau_a=20.0,ot.tau_b=20.0;motion_mild:ot.alpha_pred=0.2 --exp-prefix B2e --official
