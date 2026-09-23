#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
# 复现命令（自动记录）
scripts/run_gnn.py train --graphs data/interim/graphs_seq02_base --exp-id C5.0f_seq02_base_train --epochs 60
