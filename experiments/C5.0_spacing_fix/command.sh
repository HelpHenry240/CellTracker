#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

# 1) 平滑尺度扫描（物理 EDT 已生效；sigma 单位 µm）
python scripts/calibrate_instance_split.py \
    --pred-dir data/interim/preds_nnunet \
    --img-root data/raw/Fluo-N3DH-CE --gt-root data/raw/Fluo-N3DH-CE --seq 01 \
    --frames 10,130,194 --min-distance 6 --min-volume 100 --h-frac 0.35 \
    --gaussian-sigma 0.0,0.3,0.9,2.0 2>&1 | tee experiments/C5.0_spacing_fix/logs/sigma.log

# 2) 种子阈值重标定（判据 = 实例/标记 → 1.0 且 exclusive 高；不用 F1）
python scripts/calibrate_instance_split.py \
    --pred-dir data/interim/preds_nnunet \
    --img-root data/raw/Fluo-N3DH-CE --gt-root data/raw/Fluo-N3DH-CE --seq 01 \
    --frames 10,50,90,130,160,180,194 --min-distance 6 --min-volume 100,300 \
    --h-frac 0.1,0.2,0.35 --gaussian-sigma 0.0 \
    --save-json experiments/C5.0_spacing_fix/metrics_sweep.json \
    2>&1 | tee experiments/C5.0_spacing_fix/logs/sweep.log
python scripts/calibrate_instance_split.py \
    --pred-dir data/interim/preds_nnunet \
    --img-root data/raw/Fluo-N3DH-CE --gt-root data/raw/Fluo-N3DH-CE --seq 01 \
    --frames 10,50,90,130,160,180,194 --min-distance 6 --min-volume 300 \
    --h-frac 0.03,0.05,0.15 --gaussian-sigma 0.0 \
    --save-json experiments/C5.0_spacing_fix/metrics_bracket.json \
    2>&1 | tee experiments/C5.0_spacing_fix/logs/bracket.log

# 3) 用选定配置重生成检测（约 20 分钟，本地 CPU；旧 v1 文件保留对照）
python scripts/predict_to_h5.py --pred-dir data/interim/preds_nnunet \
    --img-root data/raw/Fluo-N3DH-CE --seq 01 \
    --out data/interim/Fluo-N3DH-CE_01_pred_v2.h5 \
    --h-frac 0.10 --gaussian-sigma 0.0 --min-volume 300 \
    --gt-root data/raw/Fluo-N3DH-CE --report data/interim/pred_report_01_v2.json

# 4) 天花板 / 全链路 / 漏斗
python scripts/detection_ceiling.py \
    --pred-h5 data/interim/Fluo-N3DH-CE_01_pred_v2.h5 \
    --gt-h5 data/interim/Fluo-N3DH-CE_01.h5 \
    --out experiments/C5.0_spacing_fix/detection_ceiling_v2.json
python scripts/run_pipeline.py --h5 data/interim/Fluo-N3DH-CE_01_pred_v2.h5 \
    --exp-id C5.0_pred_v2_graphs --dump-graphs data/interim/graphs_01_pred_v2
python scripts/run_gnn.py train --graphs data/interim/graphs_01_pred_v2 \
    --exp-id C5.0_pred_v2_train --epochs 60
python scripts/eval_pipeline.py --h5 data/interim/Fluo-N3DH-CE_01_pred_v2.h5 \
    --gt-h5 data/interim/Fluo-N3DH-CE_01.h5 --dataset Fluo-N3DH-CE --seq 01 \
    --exp-id C5.0_pred_v2_local \
    --ckpt experiments/C5.0_pred_v2_train/artifacts/model/best.pt
python scripts/pipeline_funnel.py --h5 data/interim/Fluo-N3DH-CE_01_pred_v2.h5 \
    --gt-h5 data/interim/Fluo-N3DH-CE_01.h5 --graphs data/interim/graphs_01_pred_v2 \
    --seq 01 --ckpt experiments/C5.0_pred_v2_train/artifacts/model/best.pt \
    --out experiments/C5.0_spacing_fix/funnel_v2.json
