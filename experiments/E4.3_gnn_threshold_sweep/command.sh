#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
# 复现命令（自动记录）
scripts/sweep_gnn_thresholds.py --graphs data/interim/graphs_ce01 --ckpt experiments/E4.1_gnn_ce01/artifacts/model/best.pt --h5 data/interim/Fluo-N3DH-CE_01.h5 --exp-id E4.3_gnn_threshold_sweep
