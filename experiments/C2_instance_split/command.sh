#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
# 表 A：固定尺度种子（h_frac=0）的 min_distance × min_volume 网格
python scripts/calibrate_instance_split.py \
    --pred-dir data/interim/preds_nnunet \
    --img-root data/raw/Fluo-N3DH-CE --gt-root data/raw/Fluo-N3DH-CE --seq 01 \
    --frames 10,50,90,130,160,180,194 \
    --min-distance 3,4,5,6,8,10 --min-volume 50,100,200 --h-frac 0.0 \
    --save-json experiments/C2_instance_split/metrics.json \
    2>&1 | tee experiments/C2_instance_split/logs/calibration_sweep.log

# 表 B：尺度自适应种子（h-maxima），对应真正部署的 h_frac=0.35
python scripts/calibrate_instance_split.py \
    --pred-dir data/interim/preds_nnunet \
    --img-root data/raw/Fluo-N3DH-CE --gt-root data/raw/Fluo-N3DH-CE --seq 01 \
    --frames 10,50,90,130,160,180,194 \
    --min-distance 6 --min-volume 100 --h-frac 0.0,0.2,0.35,0.5 \
    --save-json experiments/C2_instance_split/metrics_adaptive.json \
    2>&1 | tee experiments/C2_instance_split/logs/calibration_adaptive.log

# 出图 + 汇总（读 pred_report_01.json，不重跑推理）
python scripts/plot_c1c2_evidence.py
