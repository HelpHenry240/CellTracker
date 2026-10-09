#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
# 复现命令（自动记录）
/root/autodl-tmp/nnunet/venv/bin/python /root/autodl-tmp/CellTracker_rebuild_20261008/paperpipe/scripts/run_paper_pipeline.py --h5 data/interim/Fluo-N3DH-CE_02_rebuild.h5 --gt-h5 data/interim/Fluo-N3DH-CE_02.h5 --seq 02 --exp-id P7_ideas_v3_paper_all_k0_baseline_seed20261009_eval02 --config /root/autodl-tmp/CellTracker_rebuild_20261008/experiments/P7_ideas_v3_paper_all_k0_matrix_20261009/configs/baseline_calibrated.yaml --cache-dir /root/autodl-tmp/CellTracker_rebuild_20261008/experiments/P7_ideas_v3_paper_all_k0_matrix_20261009/cache/baseline/02 --device cuda --set node.encoder_feat_path=data/interim/encoder_02_rebuild.npz --ckpt /root/autodl-tmp/CellTracker_rebuild_20261008/experiments/P7_ideas_v3_paper_all_k0_matrix_20261009/models/baseline/seed20261009/best.pt --official --official-tools /root/EvaluationSoftware/Linux --official-gt-dir /root/autodl-tmp/ctc/raw/Fluo-N3DH-CE/02_GT
