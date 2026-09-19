#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
# 复现命令（自动记录）
/home/henry/ot_idea/CellTracker/scripts/run_gnn.py train --graphs /home/henry/ot_idea/CellTracker/data/interim/graphs_01_ot_cand --exp-id B2_ot_cand_train --epochs 60
