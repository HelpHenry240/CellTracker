#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

# 物理单位变体：代价与 R_max 用 µm（r_max=3 µm ≈ 真实位移 p99.9=2.71 µm）
python scripts/run_pipeline.py --h5 data/interim/Fluo-N3DH-CE_01_pred_v2.h5 \
    --gt-h5 data/interim/Fluo-N3DH-CE_01.h5 \
    --exp-id C5.0d_phys_units_graphs \
    --set ot.spacing_zyx=1.0,0.09,0.09 ot.r_max=3.0 \
    --dump-graphs data/interim/graphs_v2_phys

python scripts/run_gnn.py train --graphs data/interim/graphs_v2_phys \
    --exp-id C5.0d_phys_units_train --epochs 60

python scripts/pipeline_funnel.py --h5 data/interim/Fluo-N3DH-CE_01_pred_v2.h5 \
    --gt-h5 data/interim/Fluo-N3DH-CE_01.h5 --graphs data/interim/graphs_v2_phys \
    --seq 01 --r-max 3.0 --spacing-zyx 1.0,0.09,0.09 \
    --ckpt experiments/C5.0d_phys_units_train/artifacts/model/best.pt \
    --out experiments/C5.0d_phys_units_graphs/funnel_phys.json

# 官方评测（需云端）：注意必须同时给 spacing 与 r_max
python scripts/eval_pipeline.py --h5 data/interim/Fluo-N3DH-CE_01_pred_v2.h5 \
    --gt-h5 data/interim/Fluo-N3DH-CE_01.h5 --dataset Fluo-N3DH-CE --seq 01 \
    --exp-id C5.0d_official_phys \
    --set ot.spacing_zyx=1.0,0.09,0.09 ot.r_max=3.0 \
    --ckpt experiments/C5.0d_phys_units_train/artifacts/model/best.pt --official
