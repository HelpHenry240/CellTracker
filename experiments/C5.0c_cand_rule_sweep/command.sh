#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

# 变体重导图（预测检测 + GT 血缘）
python scripts/run_pipeline.py --h5 data/interim/Fluo-N3DH-CE_01_pred_v2.h5 \
    --gt-h5 data/interim/Fluo-N3DH-CE_01.h5 --exp-id C5.0c_topk5_graphs \
    --set graph.cand_topk=5 --dump-graphs data/interim/graphs_v2b_topk5
python scripts/run_pipeline.py --h5 data/interim/Fluo-N3DH-CE_01_pred_v2.h5 \
    --gt-h5 data/interim/Fluo-N3DH-CE_01.h5 --exp-id C5.0c_theta0005_graphs \
    --set graph.theta_gamma=0.005 --dump-graphs data/interim/graphs_v2b_theta0005
python scripts/run_pipeline.py --h5 data/interim/Fluo-N3DH-CE_01_pred_v2.h5 \
    --gt-h5 data/interim/Fluo-N3DH-CE_01.h5 --exp-id C5.0c_topk5_theta0005_graphs \
    --set graph.cand_topk=5 graph.theta_gamma=0.005 \
    --dump-graphs data/interim/graphs_v2b_topk5_theta0005

# 事件级漏斗（不带 --ckpt 只看 S1；带 --ckpt 看 S2/S3）
for v in topk5 theta0005 topk5_theta0005; do
  python scripts/pipeline_funnel.py --h5 data/interim/Fluo-N3DH-CE_01_pred_v2.h5 \
      --gt-h5 data/interim/Fluo-N3DH-CE_01.h5 \
      --graphs data/interim/graphs_v2b_${v} --seq 01
done

# 最优变体（topk5）的训练与本地评测
python scripts/run_gnn.py train --graphs data/interim/graphs_v2b_topk5 \
    --exp-id C5.0c_topk5_train --epochs 60
python scripts/pipeline_funnel.py --h5 data/interim/Fluo-N3DH-CE_01_pred_v2.h5 \
    --gt-h5 data/interim/Fluo-N3DH-CE_01.h5 --graphs data/interim/graphs_v2b_topk5 \
    --seq 01 --ckpt experiments/C5.0c_topk5_train/artifacts/model/best.pt
python scripts/eval_pipeline.py --h5 data/interim/Fluo-N3DH-CE_01_pred_v2.h5 \
    --gt-h5 data/interim/Fluo-N3DH-CE_01.h5 --dataset Fluo-N3DH-CE --seq 01 \
    --exp-id C5.0c_topk5_local --set graph.cand_topk=5 \
    --ckpt experiments/C5.0c_topk5_train/artifacts/model/best.pt
