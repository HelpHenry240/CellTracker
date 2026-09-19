#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
# 复现命令（自动记录）
scripts/run_pipeline.py --h5 data/interim/Fluo-N3DH-CE_01.h5 --exp-id A6_e2e_smoke --frames 150:175 --config configs/pipeline_default.yaml --set multiscale.enabled=true tracklet.enabled=true ot.alpha_pred=1.0
