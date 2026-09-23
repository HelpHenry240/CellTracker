#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

# 1) seq01：k=1.3 / k=2.0 的检测重建（k=1.6 见 C5.0e），三个重建可并行
for k in 1.3 2.0; do
  python -u scripts/predict_to_h5.py --pred-dir data/interim/preds_nnunet \
      --img-root data/raw/Fluo-N3DH-CE --seq 01 \
      --out "data/interim/Fluo-N3DH-CE_01_pred_v2rs${k/./}.h5" \
      --h-frac 0.10 --gaussian-sigma 0.0 --min-volume 300 --resplit-k "$k" \
      --gt-root data/raw/Fluo-N3DH-CE \
      --report "data/interim/pred_report_01_v2rs${k/./}.json"
done
# 2) seq02：留出集，用 seq01 选定的 k=1.6；同时重建"无再切"基线用于归因
python -u scripts/predict_to_h5.py --pred-dir data/interim/preds_nnunet \
    --img-root data/raw/Fluo-N3DH-CE --seq 02 \
    --out data/interim/Fluo-N3DH-CE_02_pred_v2.h5 \
    --h-frac 0.10 --gaussian-sigma 0.0 --min-volume 300 \
    --gt-root data/raw/Fluo-N3DH-CE --report data/interim/pred_report_02_v2.json
python -u scripts/predict_to_h5.py --pred-dir data/interim/preds_nnunet \
    --img-root data/raw/Fluo-N3DH-CE --seq 02 \
    --out data/interim/Fluo-N3DH-CE_02_pred_v2rs.h5 \
    --h-frac 0.10 --gaussian-sigma 0.0 --min-volume 300 --resplit-k 1.6 \
    --gt-root data/raw/Fluo-N3DH-CE --report data/interim/pred_report_02_v2rs.json

# 3) 建图 + 训练 + 官方评测（seq01 两个 k、seq02 两档）
for k in 13 20; do
  python scripts/run_pipeline.py --h5 "data/interim/Fluo-N3DH-CE_01_pred_v2rs${k}.h5" \
      --gt-h5 data/interim/Fluo-N3DH-CE_01.h5 --exp-id "C5.0f_k${k}_graphs" \
      --dump-graphs "data/interim/graphs_v2rs${k}"
  python scripts/run_gnn.py train --graphs "data/interim/graphs_v2rs${k}" \
      --exp-id "C5.0f_k${k}_train" --epochs 60
  CT_CLOUD_CONN=backup python scripts/eval_pipeline.py \
      --h5 "data/interim/Fluo-N3DH-CE_01_pred_v2rs${k}.h5" \
      --gt-h5 data/interim/Fluo-N3DH-CE_01.h5 --dataset Fluo-N3DH-CE --seq 01 \
      --exp-id "C5.0f_k${k}_official" \
      --ckpt "experiments/C5.0f_k${k}_train/artifacts/model/best.pt" --official
done
for v in base rs; do
  python scripts/run_pipeline.py --h5 "data/interim/Fluo-N3DH-CE_02_pred_v2${v/base/}.h5" \
      --gt-h5 data/interim/Fluo-N3DH-CE_02.h5 --exp-id "C5.0f_seq02_${v}_graphs" \
      --dump-graphs "data/interim/graphs_seq02_${v}"
  python scripts/run_gnn.py train --graphs "data/interim/graphs_seq02_${v}" \
      --exp-id "C5.0f_seq02_${v}_train" --epochs 60
  CT_CLOUD_CONN=backup python scripts/eval_pipeline.py \
      --h5 "data/interim/Fluo-N3DH-CE_02_pred_v2${v/base/}.h5" \
      --gt-h5 data/interim/Fluo-N3DH-CE_02.h5 --dataset Fluo-N3DH-CE --seq 02 \
      --exp-id "C5.0f_seq02_${v}_official" \
      --ckpt "experiments/C5.0f_seq02_${v}_train/artifacts/model/best.pt" --official
done

python scripts/plot_resplit_sweep.py
