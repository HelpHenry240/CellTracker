#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
# 复现命令（自动记录）
scripts/run_pipeline.py --h5 data/interim/Fluo-N3DH-CE_02.h5 --exp-id B3_seq02_graphs --config configs/pipeline_default.yaml --dump-graphs data/interim/graphs_02_base
