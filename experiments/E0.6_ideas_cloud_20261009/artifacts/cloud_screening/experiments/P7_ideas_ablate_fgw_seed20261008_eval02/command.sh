#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
# 复现命令（自动记录）
/root/autodl-tmp/nnunet/venv/bin/python /root/autodl-tmp/CellTracker_rebuild_20261008/paperpipe/scripts/run_paper_pipeline.py --h5 data/interim/Fluo-N3DH-CE_02_rebuild.h5 --gt-h5 data/interim/Fluo-N3DH-CE_02.h5 --seq 02 --exp-id P7_ideas_ablate_fgw_seed20261008_eval02 --config /root/autodl-tmp/CellTracker_rebuild_20261008/experiments/P7_ideas_ablation_matrix_20261009/configs/fgw_calibrated.yaml --cache-dir /root/autodl-tmp/CellTracker_rebuild_20261008/experiments/P7_ideas_ablation_matrix_20261009/cache/fgw/02 --device cuda --set node.encoder_feat_path=data/interim/encoder_02_rebuild.npz --ckpt /root/autodl-tmp/CellTracker_rebuild_20261008/experiments/P7_ideas_ablation_matrix_20261009/models/fgw/seed20261008/best.pt --official --official-tools /root/EvaluationSoftware/Linux --official-gt-dir /root/autodl-tmp/ctc/raw/Fluo-N3DH-CE/02_GT
