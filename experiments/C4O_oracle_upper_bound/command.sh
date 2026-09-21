#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

# 1) oracle 检测：GT 标记作分水岭种子，掩码仍是 nnU-Net 预测（约 20 分钟，本地）
python scripts/predict_to_h5.py --pred-dir data/interim/preds_nnunet \
    --img-root data/raw/Fluo-N3DH-CE --seq 01 \
    --out data/interim/Fluo-N3DH-CE_01_pred_oracle.h5 \
    --h-frac 0.10 --gaussian-sigma 0.0 --min-volume 300 \
    --gt-root data/raw/Fluo-N3DH-CE --oracle-markers \
    --report data/interim/pred_report_01_oracle.json

# 2) 天花板 / 建图 / 训练 / 漏斗
python scripts/detection_ceiling.py \
    --pred-h5 data/interim/Fluo-N3DH-CE_01_pred_oracle.h5 \
    --gt-h5 data/interim/Fluo-N3DH-CE_01.h5 --out /tmp/ceiling_oracle.json
python scripts/run_pipeline.py --h5 data/interim/Fluo-N3DH-CE_01_pred_oracle.h5 \
    --gt-h5 data/interim/Fluo-N3DH-CE_01.h5 --exp-id C4O_oracle_graphs \
    --dump-graphs data/interim/graphs_01_oracle
python scripts/run_gnn.py train --graphs data/interim/graphs_01_oracle \
    --exp-id C4O_oracle_train --epochs 60
python scripts/pipeline_funnel.py --h5 data/interim/Fluo-N3DH-CE_01_pred_oracle.h5 \
    --gt-h5 data/interim/Fluo-N3DH-CE_01.h5 --graphs data/interim/graphs_01_oracle \
    --seq 01 --ckpt experiments/C4O_oracle_train/artifacts/model/best.pt

# 3) 官方评测
python scripts/eval_pipeline.py --h5 data/interim/Fluo-N3DH-CE_01_pred_oracle.h5 \
    --gt-h5 data/interim/Fluo-N3DH-CE_01.h5 --dataset Fluo-N3DH-CE --seq 01 \
    --exp-id C4O_official_gnn \
    --ckpt experiments/C4O_oracle_train/artifacts/model/best.pt --official

# 4) 三档对比图
python scripts/plot_official_tiers.py \
    --out experiments/C4O_official_gnn/figures/official_tiers
