#!/bin/bash
set -euo pipefail
cd /root/autodl-tmp/CellTracker_rebuild_20261008
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nnUNet_compile=false
PY=/root/autodl-tmp/nnunet/venv/bin/python
MODEL=/root/autodl-tmp/nnunet/nnUNet_results/Dataset501_CellTrackerCE/nnUNetTrainer__nnUNetPlans__3d_fullres
IMAGES=/root/autodl-tmp/nnunet/nnUNet_raw/Dataset501_CellTrackerCE/imagesTs_eval
for SEQ in 01 02; do
 "$PY" -u scripts/resplit_detections.py --input "data/interim/Fluo-N3DH-CE_${SEQ}_rebuild.h5" \
  --gt-h5 "data/interim/Fluo-N3DH-CE_${SEQ}.h5" --img-dir "/root/autodl-tmp/ctc/Fluo-N3DH-CE/${SEQ}" \
  --out "data/interim/Fluo-N3DH-CE_${SEQ}_resplit_v2.h5" --k 1.6
 done
printf 'ready\n' > logs/ideas_frontend_raw_20261009.complete
# CPU 数据准备与当前队列并行；GPU 特征前向等待训练/评测队列结束。
while ! test -f logs/ideas_control_20261009.complete; do
 if ! kill -0 "$(cat logs/ideas_control_20261009.pid)" 2>/dev/null; then
  echo 'Control queue stopped; inspect its logs before continuing'; exit 1
 fi
 sleep 10
done
for SEQ in 01 02; do
 nvidia-smi > "logs/frontend_encoder_${SEQ}_gpu_before.txt"
 "$PY" -u scripts/nnunet/export_encoder_features.py \
  --h5 "data/interim/Fluo-N3DH-CE_${SEQ}_resplit_v2.h5" --images "$IMAGES" --model "$MODEL" \
  --seq "$SEQ" --out "data/interim/encoder_${SEQ}_resplit_v2.npz" --device cuda &
 FEATURE_PID=$!
 printf '%s\n' "$FEATURE_PID" > "logs/frontend_encoder_${SEQ}.pid"
 while kill -0 "$FEATURE_PID" 2>/dev/null; do
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader >> "logs/frontend_encoder_${SEQ}_gpu_active.txt"
  sleep 10
 done
 wait "$FEATURE_PID"
 test -s "data/interim/encoder_${SEQ}_resplit_v2.npz"
 nvidia-smi > "logs/frontend_encoder_${SEQ}_gpu_after.txt"
done
printf 'complete\n' > logs/ideas_frontend_features_20261009.complete
