#!/bin/bash
set -euo pipefail
cd /root/CellTracker_oracle_encoder_conservative_20261009
nohup /root/autodl-tmp/nnunet/venv/bin/python -u \
  experiments/E0.9_gt_oracle_encoder_20261009/artifacts/postprocess_oracle.py \
  > logs/oracle_postprocess.log 2>&1 < /dev/null &
printf '%s\n' "$!" > logs/oracle_postprocess.pid
cat logs/oracle_postprocess.pid
