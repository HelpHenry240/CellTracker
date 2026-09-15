#!/usr/bin/env bash
# E0.3 复现命令
set -euo pipefail
cd "$(dirname "$0")/../.."

# 1) 本地 SEG 实现自检（对比官方文档给出的基准值）
PYTHONPATH=src python -c "
from celltracker.eval.local_metrics import seg_measure
assert abs(seg_measure('tools/testing_dataset/01_GT/SEG','tools/testing_dataset/01_RES') - 0.232874) < 1e-6
assert abs(seg_measure('tools/testing_dataset/02_GT/SEG','tools/testing_dataset/02_RES') - 0.443686) < 1e-6
print('local SEG self-check OK')
"

# 2) 官方二进制（云端）自检
python scripts/cloud_eval.py --res-dir tools/testing_dataset/01_RES \
    --dataset ds --seq 01 --cloud-gt-root /root/ctc_testing
python scripts/cloud_eval.py --res-dir tools/testing_dataset/03_RES \
    --dataset ds --seq 03 --cloud-gt-root /root/ctc_testing
