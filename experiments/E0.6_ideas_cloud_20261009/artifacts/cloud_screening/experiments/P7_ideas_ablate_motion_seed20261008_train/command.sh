#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
# 复现命令（自动记录）
/root/autodl-tmp/nnunet/venv/bin/python /root/autodl-tmp/CellTracker_rebuild_20261008/paperpipe/scripts/train_paper_gnn.py --graphs /root/autodl-tmp/CellTracker_rebuild_20261008/experiments/P7_ideas_ablation_matrix_20261009/graphs/motion --out /root/autodl-tmp/CellTracker_rebuild_20261008/experiments/P7_ideas_ablation_matrix_20261009/models/motion/seed20261008 --config /root/autodl-tmp/CellTracker_rebuild_20261008/experiments/P7_ideas_ablation_matrix_20261009/configs/motion_calibrated.yaml --epochs 60 --seed 20261008 --device cuda --exp-id P7_ideas_ablate_motion_seed20261008_train
