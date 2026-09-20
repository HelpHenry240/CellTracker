#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

# 1) 用正确的 GT 血缘重导图（预测检测 + GT tracks）
python scripts/run_pipeline.py --h5 data/interim/Fluo-N3DH-CE_01_pred_v2.h5 \
    --gt-h5 data/interim/Fluo-N3DH-CE_01.h5 \
    --exp-id C5.0b_pred_v2_graphs --dump-graphs data/interim/graphs_01_pred_v2b

# 2) 重训 GNN
python scripts/run_gnn.py train --graphs data/interim/graphs_01_pred_v2b \
    --exp-id C5.0b_pred_v2_train --epochs 60

# 3) 漏斗 + 本地全链路
python scripts/pipeline_funnel.py --h5 data/interim/Fluo-N3DH-CE_01_pred_v2.h5 \
    --gt-h5 data/interim/Fluo-N3DH-CE_01.h5 --graphs data/interim/graphs_01_pred_v2b \
    --seq 01 --ckpt experiments/C5.0b_pred_v2_train/artifacts/model/best.pt \
    --out experiments/C5.0b_pred_v2_graphs/funnel_v2b.json
python scripts/eval_pipeline.py --h5 data/interim/Fluo-N3DH-CE_01_pred_v2.h5 \
    --gt-h5 data/interim/Fluo-N3DH-CE_01.h5 --dataset Fluo-N3DH-CE --seq 01 \
    --exp-id C5.0b_pred_v2_local \
    --ckpt experiments/C5.0b_pred_v2_train/artifacts/model/best.pt
