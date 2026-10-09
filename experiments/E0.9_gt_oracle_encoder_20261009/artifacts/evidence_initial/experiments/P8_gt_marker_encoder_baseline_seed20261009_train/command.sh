#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
# 复现命令（自动记录）
/root/autodl-tmp/nnunet/venv/bin/python /root/CellTracker_oracle_encoder_20261009/paperpipe/scripts/train_paper_gnn.py --graphs /root/CellTracker_oracle_encoder_20261009/experiments/P8_gt_marker_encoder_matrix_20261009/graphs/baseline --out /root/CellTracker_oracle_encoder_20261009/experiments/P8_gt_marker_encoder_matrix_20261009/models/baseline/seed20261009 --config /root/CellTracker_oracle_encoder_20261009/experiments/P8_gt_marker_encoder_matrix_20261009/configs/baseline_calibrated.yaml --epochs 60 --seed 20261009 --device cuda --exp-id P8_gt_marker_encoder_baseline_seed20261009_train
