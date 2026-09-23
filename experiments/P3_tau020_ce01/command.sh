#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
# 复现命令（自动记录）
paperpipe/scripts/run_paper_pipeline.py --h5 data/interim/Fluo-N3DH-CE_01_pred_v2.h5 --gt-h5 data/interim/Fluo-N3DH-CE_01.h5 --seq 01 --exp-id P3_tau020_ce01 --config paperpipe/configs/paper_e2e_ce.yaml --set reconstruct.tau_edge=0.2 --ckpt paperpipe/runs/pg_gnn_01/best.pt
