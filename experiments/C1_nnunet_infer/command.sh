#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
# C1 在云端 GPU 上执行。以下命令是**回填的等效命令**（按产物命名与 scripts/nnunet/*
# 推断），待 C4 开机后与云端 shell 历史/日志核对；核对前不要当作原始记录引用。
#
# --- 云端（/root/autodl-tmp/nnunet/venv 环境变量已在 cloud_train.sh 中导出）---
# export nnUNet_raw=/root/autodl-tmp/nnunet/nnUNet_raw
# export nnUNet_preprocessed=/root/autodl-tmp/nnunet/nnUNet_preprocessed
# export nnUNet_results=/root/autodl-tmp/nnunet/nnUNet_results
#
# 1) 生成推理输入（只转有 GT 标注的帧，间距 (0.09,0.09,1.0) µm）
# python scripts/nnunet/make_infer_input.py \
#     --src /root/autodl-tmp/ctc/Fluo-N3DH-CE \
#     --out "$nnUNet_raw/Dataset501_CellTrackerCE/imagesTs_eval"
#
# 2) 推理（3d_fullres，fold 0，best 权重，TTA 默认）
# nohup "$nnUNet /bin/nnUNetv2_predict" -i "$nnUNet_raw/Dataset501_CellTrackerCE/imagesTs_eval" \
#     -o /root/autodl-tmp/preds_nnunet -d 501 -c 3d_fullres -f 0 \
#     -chk checkpoint_best > /root/autodl-tmp/logs/predict.log 2>&1 &
#
# 3) 回传（本地执行）
# python scripts/cloud_run.py --download /root/autodl-tmp/preds_nnunet data/interim/preds_nnunet
echo "C1 命令为回填记录，见上方注释；原始云端日志待 C4 核对补齐。"
