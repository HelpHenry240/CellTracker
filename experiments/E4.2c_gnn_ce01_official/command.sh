#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
# 复现命令（自动记录）
scripts/run_gnn.py infer --graphs data/interim/graphs_ce01 --ckpt experiments/E4.1_gnn_ce01/artifacts/model/best.pt --h5 data/interim/Fluo-N3DH-CE_01.h5 --dataset Fluo-N3DH-CE --seq 01 --exp-id E4.2c_gnn_ce01_official --official
