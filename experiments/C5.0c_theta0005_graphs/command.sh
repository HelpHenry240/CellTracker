#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
# 复现命令（自动记录）
scripts/run_pipeline.py --h5 data/interim/Fluo-N3DH-CE_01_pred_v2.h5 --gt-h5 data/interim/Fluo-N3DH-CE_01.h5 --exp-id C5.0c_theta0005_graphs --set graph.theta_gamma=0.005 --dump-graphs data/interim/graphs_v2b_theta0005
