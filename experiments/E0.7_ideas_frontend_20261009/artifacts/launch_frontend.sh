#!/bin/bash
set -euo pipefail
cd /root/autodl-tmp/CellTracker_rebuild_20261008
nohup bash run_frontend_prepare_20261009.sh > logs/ideas_frontend_prepare_20261009.log 2>&1 < /dev/null &
printf '%s\n' "$!" > logs/ideas_frontend_prepare_20261009.pid
cat logs/ideas_frontend_prepare_20261009.pid
