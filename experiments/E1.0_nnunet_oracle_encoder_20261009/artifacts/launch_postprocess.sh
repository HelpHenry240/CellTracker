#!/bin/bash
set -euo pipefail
cd /root/CellTracker_nnunet_oracle_encoder_20261009
nohup /root/autodl-tmp/nnunet/venv/bin/python -u \
  experiments/E1.0_nnunet_oracle_encoder_20261009/artifacts/postprocess_oracle.py \
  > logs/oracle_postprocess.log 2>&1 < /dev/null &
printf '%s\n' "$!" > logs/oracle_postprocess.pid
cat logs/oracle_postprocess.pid
