#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
# 复现命令（自动记录）
scripts/run_ablation_matrix.py --h5 data/interim/Fluo-N3DH-CE_01.h5 --dataset Fluo-N3DH-CE --seq 01 --experiments fgw_mild:ot.eta=0.1;unbalanced_mid:ot.tau_a=5.0,ot.tau_b=5.0 --exp-prefix B2d --official
