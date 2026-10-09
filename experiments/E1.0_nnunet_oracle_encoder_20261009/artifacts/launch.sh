#!/bin/bash
set -euo pipefail
cd /root/CellTracker_nnunet_oracle_encoder_20261009
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
nohup /root/autodl-tmp/nnunet/venv/bin/python -u \
  experiments/E1.0_nnunet_oracle_encoder_20261009/artifacts/run_mask_oracle.py \
  --root /root/CellTracker_nnunet_oracle_encoder_20261009 \
  --original-interim /root/autodl-tmp/CellTracker_rebuild_20261008/data/interim \
  > logs/oracle_encoder_pipeline.log 2>&1 < /dev/null &
printf '%s\n' "$!" > logs/oracle_encoder_pipeline.pid
cat logs/oracle_encoder_pipeline.pid
