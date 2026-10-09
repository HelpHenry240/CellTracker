#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
# 复现命令（自动记录）
/root/autodl-tmp/nnunet/venv/bin/python /root/CellTracker_oracle_encoder_20261009/paperpipe/scripts/run_paper_pipeline.py --h5 data/interim/Fluo-N3DH-CE_01.h5 --gt-h5 data/interim/Fluo-N3DH-CE_01.h5 --seq 01 --exp-id P8_gt_marker_encoder_baseline_build01 --config /root/CellTracker_oracle_encoder_20261009/experiments/P8_gt_marker_encoder_matrix_20261009/configs/baseline_calibrated.yaml --cache-dir /root/CellTracker_oracle_encoder_20261009/experiments/P8_gt_marker_encoder_matrix_20261009/cache/baseline/01 --device cuda --set node.encoder_feat_path=data/interim/encoder_01_gt_marker_20261009.npz --dump-graphs /root/CellTracker_oracle_encoder_20261009/experiments/P8_gt_marker_encoder_matrix_20261009/graphs/baseline --build-only
