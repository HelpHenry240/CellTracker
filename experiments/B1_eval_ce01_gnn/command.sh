#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
# 复现命令（自动记录）
scripts/eval_pipeline.py --h5 data/interim/Fluo-N3DH-CE_01.h5 --dataset Fluo-N3DH-CE --seq 01 --exp-id B1_eval_ce01_gnn --ckpt experiments/B1_gnn_paper_ce01/artifacts/model/best.pt --official
