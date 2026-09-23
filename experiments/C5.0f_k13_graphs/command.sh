#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
# 复现命令（自动记录）
scripts/run_pipeline.py --h5 data/interim/Fluo-N3DH-CE_01_pred_v2rs13.h5 --gt-h5 data/interim/Fluo-N3DH-CE_01.h5 --exp-id C5.0f_k13_graphs --dump-graphs data/interim/graphs_v2rs13
