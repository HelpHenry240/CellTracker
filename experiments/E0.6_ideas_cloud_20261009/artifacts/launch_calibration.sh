#!/bin/bash
set -euo pipefail
cd /root/autodl-tmp/CellTracker_rebuild_20261008
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
mkdir -p experiments/P7_ideas_calibration01_v2_20261009 logs
nohup /root/autodl-tmp/nnunet/venv/bin/python -u paperpipe/scripts/calibrate_params.py \
 --h5 data/interim/Fluo-N3DH-CE_01_rebuild.h5 --gt-h5 data/interim/Fluo-N3DH-CE_01.h5 \
 --config paperpipe/configs/paper_e2e_ce.yaml --set node.encoder_feat_path=data/interim/encoder_01_rebuild.npz coupling.r_max=6.4 \
 --cache-dir data/interim/ideas_calibration_cache01_v2_20261009 \
 --out experiments/P7_ideas_calibration01_v2_20261009/distributions.json \
 --apply-out experiments/P7_ideas_calibration01_v2_20261009/calibrated.yaml \
 > logs/ideas_calibration01_v2_20261009.log 2>&1 < /dev/null &
printf '%s\n' "$!" > logs/ideas_calibration01_v2_20261009.pid
cat logs/ideas_calibration01_v2_20261009.pid
