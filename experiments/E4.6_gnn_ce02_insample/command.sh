#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
# 复现命令（自动记录）
scripts/run_gnn.py train --graphs data/interim/graphs_ce02 --exp-id E4.6_gnn_ce02_insample --epochs 40
