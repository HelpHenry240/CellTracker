#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
# 复现命令（自动记录）
/root/autodl-tmp/nnunet/venv/bin/python /root/autodl-tmp/CellTracker_rebuild_20261008/paperpipe/scripts/run_paper_pipeline.py --h5 data/interim/Fluo-N3DH-CE_01_rebuild.h5 --gt-h5 data/interim/Fluo-N3DH-CE_01.h5 --seq 01 --exp-id P7_ideas_v3_paper_all_k0_baseline_build01 --config /root/autodl-tmp/CellTracker_rebuild_20261008/experiments/P7_ideas_v3_paper_all_k0_matrix_20261009/configs/baseline_calibrated.yaml --cache-dir /root/autodl-tmp/CellTracker_rebuild_20261008/experiments/P7_ideas_v3_paper_all_k0_matrix_20261009/cache/baseline/01 --device cuda --set node.encoder_feat_path=data/interim/encoder_01_rebuild.npz --dump-graphs /root/autodl-tmp/CellTracker_rebuild_20261008/experiments/P7_ideas_v3_paper_all_k0_matrix_20261009/graphs/baseline --build-only
