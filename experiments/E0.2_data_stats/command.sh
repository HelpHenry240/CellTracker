#!/usr/bin/env bash
# E0.2 复现命令
set -euo pipefail
cd "$(dirname "$0")/../.."

bash scripts/download_data.sh Fluo-N3DH-CHO
for s in 01 02; do
  PYTHONPATH=src python -m celltracker.data.build_dataset \
      --dataset data/raw/Fluo-N3DH-CHO --seq "$s" \
      --out "data/interim/Fluo-N3DH-CHO_${s}.h5"
  PYTHONPATH=src python -m celltracker.data.summarize \
      --h5 "data/interim/Fluo-N3DH-CHO_${s}.h5" \
      --figdir experiments/E0.2_data_stats/figures \
      --out "experiments/E0.2_data_stats/stats_CHO_${s}.json"
done
