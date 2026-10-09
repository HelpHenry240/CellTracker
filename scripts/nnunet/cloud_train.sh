#!/usr/bin/env bash
# 在云服务器上：解压数据 -> 转 nnU-Net 数据集 -> plan/preprocess -> 训练。
# 用法（云端）: bash /root/autodl-tmp/nnunet/cloud_train.sh [stage]
#   stage = all | data | preprocess | train
set -euo pipefail

STAGE="${1:-all}"
BASE="${BASE:-/root/autodl-tmp/nnunet}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="$BASE/venv"
DATASET=501
DS_NAME="Dataset${DATASET}_CellTrackerCE"
SRC=/root/autodl-tmp/ctc/Fluo-N3DH-CE

export nnUNet_raw="$BASE/nnUNet_raw"
export nnUNet_preprocessed="$BASE/nnUNet_preprocessed"
export nnUNet_results="$BASE/nnUNet_results"
export nnUNet_n_proc_DA=24          # 128 核机器，留足余量给数据加载
mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"

echo "=== [$(date +%H:%M:%S)] stage=$STAGE ==="

if [[ "$STAGE" == "all" || "$STAGE" == "data" ]]; then
  if [[ ! -d "$SRC/01" ]]; then
    echo "解压数据 ..."
    mkdir -p "$SRC"
    tar xf /root/autodl-tmp/ctc/ce_src.tar -C "$SRC"
  fi
  if [[ ! -f "$nnUNet_raw/$DS_NAME/dataset.json" ]]; then
    echo "转换为 nnU-Net 格式 ..."
    "$VENV/bin/python" "$REPO_ROOT/scripts/nnunet/build_dataset.py" --src "$SRC" \
        --out "$nnUNet_raw/$DS_NAME"
  fi
fi

if [[ "$STAGE" == "all" || "$STAGE" == "preprocess" ]]; then
  if [[ ! -d "$nnUNet_preprocessed/$DS_NAME" ]]; then
    echo "plan & preprocess (仅 3d_fullres) ..."
    "$VENV/bin/nnUNetv2_plan_and_preprocess" -d "$DATASET" -c 3d_fullres \
        -np "${NP:-24}" --verify_dataset_integrity
  fi
fi

if [[ "$STAGE" == "all" || "$STAGE" == "train" ]]; then
  echo "开始训练 3d_fullres fold 0 ${TRAIN_ARGS:-} ..."
  # TRAIN_ARGS="--c" 表示从 checkpoint_latest.pth 续训（不要改 --num_epochs，
  # 否则余弦学习率调度会与 checkpoint 内的总 epoch 不一致）
  "$VENV/bin/nnUNetv2_train" "$DATASET" 3d_fullres 0 ${TRAIN_ARGS:-}
fi

echo "=== [$(date +%H:%M:%S)] stage=$STAGE 完成 ==="
