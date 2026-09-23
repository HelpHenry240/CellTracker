#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

# 规则 A：超大实例再切（需重建检测 h5，约 25 分钟）
python scripts/predict_to_h5.py --pred-dir data/interim/preds_nnunet \
    --img-root data/raw/Fluo-N3DH-CE --seq 01 \
    --out data/interim/Fluo-N3DH-CE_01_pred_v2rs.h5 \
    --h-frac 0.10 --gaussian-sigma 0.0 --min-volume 300 --resplit-k 1.6 \
    --gt-root data/raw/Fluo-N3DH-CE --report data/interim/pred_report_01_v2rs.json
python scripts/run_pipeline.py --h5 data/interim/Fluo-N3DH-CE_01_pred_v2rs.h5 \
    --gt-h5 data/interim/Fluo-N3DH-CE_01.h5 --exp-id C5.0e_resplit_graphs \
    --dump-graphs data/interim/graphs_v2rs
python scripts/run_gnn.py train --graphs data/interim/graphs_v2rs \
    --exp-id C5.0e_resplit_train --epochs 60

# 规则 B：孤立短轨迹过滤（直接用现有 v2 检测，无需重建）
python scripts/eval_pipeline.py --h5 data/interim/Fluo-N3DH-CE_01_pred_v2.h5 \
    --gt-h5 data/interim/Fluo-N3DH-CE_01.h5 --dataset Fluo-N3DH-CE --seq 01 \
    --exp-id C5.0e_dropiso1_local --set reconstruct.drop_isolated_len=1 \
    --ckpt experiments/C5.0b_pred_v2_train/artifacts/model/best.pt
python scripts/eval_pipeline.py --h5 data/interim/Fluo-N3DH-CE_01_pred_v2.h5 \
    --gt-h5 data/interim/Fluo-N3DH-CE_01.h5 --dataset Fluo-N3DH-CE --seq 01 \
    --exp-id C5.0e_dropiso_local --set reconstruct.drop_isolated_len=2 \
    --ckpt experiments/C5.0b_pred_v2_train/artifacts/model/best.pt
