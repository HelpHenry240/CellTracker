#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
# 复现命令（自动记录）
paperpipe/scripts/run_paper_pipeline.py --h5 data/interim/Fluo-N3DH-CE_02_e2e.h5 --gt-h5 data/interim/Fluo-N3DH-CE_02.h5 --seq 02 --exp-id P4_final_gnn_ce02 --config paperpipe/configs/paper_e2e_ce.yaml --set reconstruct.tau_edge=0.35 --ckpt paperpipe/runs/pg_gnn_01/best.pt
