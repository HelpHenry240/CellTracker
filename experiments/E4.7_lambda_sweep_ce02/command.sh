#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
# 复现命令（自动记录）
scripts/sweep_gnn_thresholds.py --graphs data/interim/graphs_ce02 --ckpt experiments/E4.6_gnn_ce02_insample/artifacts/model/best.pt --h5 data/interim/Fluo-N3DH-CE_02.h5 --exp-id E4.7_lambda_sweep_ce02 --lam 0,0.25,0.5,0.75,1.0 --dataset Fluo-N3DH-CE --seq 02
