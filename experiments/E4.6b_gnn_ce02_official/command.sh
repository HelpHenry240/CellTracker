#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
# 复现命令（自动记录）
scripts/run_gnn.py infer --graphs data/interim/graphs_ce02 --ckpt experiments/E4.6_gnn_ce02_insample/artifacts/model/best.pt --h5 data/interim/Fluo-N3DH-CE_02.h5 --dataset Fluo-N3DH-CE --seq 02 --exp-id E4.6b_gnn_ce02_official --tau-move 0.5 --tau-div 0.5 --official
