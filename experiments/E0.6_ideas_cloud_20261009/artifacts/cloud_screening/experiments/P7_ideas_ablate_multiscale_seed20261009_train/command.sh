#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
# 复现命令（自动记录）
/root/autodl-tmp/nnunet/venv/bin/python /root/autodl-tmp/CellTracker_rebuild_20261008/paperpipe/scripts/train_paper_gnn.py --graphs /root/autodl-tmp/CellTracker_rebuild_20261008/experiments/P7_ideas_ablation_matrix_20261009/graphs/multiscale --out /root/autodl-tmp/CellTracker_rebuild_20261008/experiments/P7_ideas_ablation_matrix_20261009/models/multiscale/seed20261009 --config /root/autodl-tmp/CellTracker_rebuild_20261008/experiments/P7_ideas_ablation_matrix_20261009/configs/multiscale_calibrated.yaml --epochs 60 --seed 20261009 --device cuda --exp-id P7_ideas_ablate_multiscale_seed20261009_train
