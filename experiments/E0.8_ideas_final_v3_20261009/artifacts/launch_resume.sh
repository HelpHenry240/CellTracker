#!/bin/bash
set -euo pipefail
cd /root/autodl-tmp/CellTracker_rebuild_20261008
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cp logs/ideas_frontend_prepare_20261009.pid logs/ideas_frontend_prepare_20261009_initial.pid
cp logs/ideas_final_v3_20261009.pid logs/ideas_final_v3_20261009_initial.pid
nohup /root/autodl-tmp/nnunet/venv/bin/python -u resume_encoder_20261009.py > logs/ideas_frontend_encoder_resume_20261009.log 2>&1 < /dev/null &
printf '%s\n' "$!" > logs/ideas_frontend_prepare_20261009.pid
nohup /root/autodl-tmp/nnunet/venv/bin/python -u resume_final_v3_20261009.py > logs/ideas_final_v3_resume_20261009.log 2>&1 < /dev/null &
printf '%s\n' "$!" > logs/ideas_final_v3_20261009.pid
cat logs/ideas_frontend_prepare_20261009.pid logs/ideas_final_v3_20261009.pid
