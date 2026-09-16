#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
# 复现命令（自动记录）
scripts/run_gnn.py infer --graphs data/interim/graphs_ce01_w1 --ckpt experiments/E4.4_gnn_ctx1_ce01/artifacts/model/best.pt --h5 data/interim/Fluo-N3DH-CE_01.h5 --dataset Fluo-N3DH-CE --seq 01 --exp-id E4.4b_gnn_ctx1_official --tau-move 0.5 --tau-div 0.5 --official
