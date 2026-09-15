#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
# 复现命令（自动记录）
scripts/run_baseline.py --h5 data/interim/Fluo-N3DH-CHO_01.h5 --dataset Fluo-N3DH-CHO --seq 01 --exp-id E1.1_baseline_hungarian --method hungarian --max-dist 30
