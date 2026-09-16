#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
# 复现命令（自动记录）
scripts/run_baseline.py --h5 data/interim/Fluo-N3DH-CE_02.h5 --dataset Fluo-N3DH-CE --seq 02 --exp-id E1.4_CE02_hungarian --method hungarian --max-dist 30 --official
