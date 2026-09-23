#!/usr/bin/env bash
# 云端：nnU-Net 推理（Fluo-N3DH-CE 两个序列的**有标注帧**）→ 供端到端链路使用的掩码。
#
# 背景：C1 那次推理没有把命令落盘（只在 notes 里回填了等效命令）。本脚本把
# "输入转换 + 推理"固化下来，便于复现与查错（AGENTS.md E11：长任务要有命令与日志）。
#
# 用法（云端，后台跑）::
#   bash scripts/cloud_run.sh "bash /root/autodl-tmp/nnunet/cloud_infer_eval.sh \
#       > /root/autodl-tmp/nnunet/logs/infer_eval.log 2>&1 & echo \$!"
#
# 关键口径（与训练一致，改动会让 nnU-Net 的重采样决策错位）：
#   * 输入体素间距 (0.09, 0.09, 1.0) µm —— 由 `make_infer_input.py` 写入
#   * 3d_fullres / fold 0 / TTA 开（nnUNetv2_predict 默认）
#   * 权重默认 checkpoint_best.pth（C1/E5.1 用的那个，验证 Dice 0.9605）
set -euo pipefail

BASE=/root/autodl-tmp/nnunet
DS_NAME=Dataset501_CellTrackerCE
DATASET=501

export nnUNet_raw="$BASE/nnUNet_raw"
export nnUNet_preprocessed="$BASE/nnUNet_preprocessed"
export nnUNet_results="$BASE/nnUNet_results"
export nnUNet_n_proc_DA=24

# 备用2 实例上图像与 GT 分开放：图像在 ctc/Fluo-N3DH-CE/{01,02}，
# 标注在 ctc/raw/Fluo-N3DH-CE/{01,02}_GT —— 必须用 --gt-root 指过去，
# 否则 make_infer_input 会匹配到 0 帧（实测踩过）。
SRC="${SRC:-/root/autodl-tmp/ctc/Fluo-N3DH-CE}"           # 原始 tif 序列（含 01/ 02/）
GT_ROOT="${GT_ROOT:-/root/autodl-tmp/ctc/raw/Fluo-N3DH-CE}"  # 标注根（含 01_GT/TRA）
OUT="${OUT:-$BASE/preds_eval}"                      # 预测输出目录（*.nii.gz）
CKPT="${CKPT:-checkpoint_best.pth}"                 # best / final
INPUT="$nnUNet_raw/$DS_NAME/imagesTs_eval"

mkdir -p "$INPUT" "$OUT" "$BASE/logs"

echo "=== [$(date +%H:%M:%S)] 1/2 生成推理输入 ==="
if [[ -f "$INPUT/CE02_f189_0000.nii.gz" && "${FORCE_INPUT:-0}" != "1" ]]; then
  echo "输入已存在（$(ls "$INPUT" | wc -l) 个），跳过转换"
else
  "$BASE/venv/bin/python" "$BASE/make_infer_input.py" \
      --src "$SRC" --gt-root "$GT_ROOT" --out "$INPUT" --sequences 01,02
fi

echo "=== [$(date +%H:%M:%S)] 2/2 nnUNetv2_predict（$CKPT，TTA 默认开）==="
md5sum "$nnUNet_results/$DS_NAME/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/$CKPT"
"$BASE/venv/bin/nnUNetv2_predict" \
    -i "$INPUT" -o "$OUT" -d "$DATASET" -c 3d_fullres -f 0 -chk "$CKPT"

echo "=== [$(date +%H:%M:%S)] 完成：$(ls "$OUT" | wc -l) 个预测文件 -> $OUT ==="
