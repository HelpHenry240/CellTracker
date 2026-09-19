#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
# 复现命令（自动记录）
scripts/run_ablation_matrix.py --h5 data/interim/Fluo-N3DH-CE_01.h5 --dataset Fluo-N3DH-CE --seq 01 --experiments fgw:ot.eta=0.3;unbalanced:ot.tau_a=1.0,ot.tau_b=1.0;motion:ot.alpha_pred=1.0 --exp-prefix B2c --official
