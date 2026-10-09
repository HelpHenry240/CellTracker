#!/bin/bash
set -euo pipefail
cd /root/autodl-tmp/CellTracker_rebuild_20261008
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PY=/root/autodl-tmp/nnunet/venv/bin/python
CONFIG=experiments/P7_ideas_calibration01_v2_20261009/calibrated.yaml
while test ! -f "$CONFIG"; do
 if ! kill -0 "$(cat logs/ideas_calibration01_v2_20261009.pid)" 2>/dev/null; then
  echo 'Calibration stopped without producing configuration'; exit 1
 fi
 sleep 10
done
nvidia-smi > logs/baseline_gpu_before_20261009.txt
"$PY" -u paperpipe/scripts/run_ablation_matrix.py --config "$CONFIG" \
 --h5-01 data/interim/Fluo-N3DH-CE_01_rebuild.h5 --h5-02 data/interim/Fluo-N3DH-CE_02_rebuild.h5 \
 --gt-01 data/interim/Fluo-N3DH-CE_01.h5 --gt-02 data/interim/Fluo-N3DH-CE_02.h5 \
 --encoder01 data/interim/encoder_01_rebuild.npz --encoder02 data/interim/encoder_02_rebuild.npz \
 --out experiments/P7_ideas_baseline_matrix_20261009 --exp-prefix P7_ideas_full --ablations \
 --seeds 20261008 20261009 --epochs 60 --device cuda --official --execute
nvidia-smi > logs/baseline_gpu_after_20261009.txt
printf 'complete\n' > logs/ideas_baseline_20261009.complete
