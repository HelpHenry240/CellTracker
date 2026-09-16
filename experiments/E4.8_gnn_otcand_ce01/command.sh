#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
# 复现命令（自动记录）
scripts/run_gnn.py train --graphs data/interim/graphs_ce01_otcand --exp-id E4.8_gnn_otcand_ce01 --epochs 40
