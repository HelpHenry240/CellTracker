#!/bin/bash
set -euo pipefail
cd /root/autodl-tmp/CellTracker_rebuild_20261008
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
nohup /root/autodl-tmp/nnunet/venv/bin/python -u run_fourth_profile_20261009.py > logs/ideas_final_factorial_20261009.log 2>&1 < /dev/null &
printf '%s\n' "$!" > logs/ideas_final_factorial_20261009.pid
cat logs/ideas_final_factorial_20261009.pid
