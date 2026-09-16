#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
# 复现命令（自动记录）
scripts/run_gnn.py train --graphs data/interim/graphs_ce01_otcand3 --exp-id E4.9_gnn_top3_ce01 --epochs 40
