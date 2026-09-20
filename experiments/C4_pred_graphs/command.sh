#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
# 复现命令（自动记录）
scripts/run_pipeline.py --h5 data/interim/Fluo-N3DH-CE_01_pred.h5 --exp-id C4_pred_graphs --dump-graphs data/interim/graphs_01_pred
