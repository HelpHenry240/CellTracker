#!/usr/bin/env bash
# 下载 CTC 训练数据集（默认 3D 为主）。
# 用法:
#   bash scripts/download_data.sh                # 默认: Fluo-N3DH-CHO Fluo-N3DH-CE
#   bash scripts/download_data.sh Fluo-N2DL-HeLa # 指定数据集
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
base="https://data.celltrackingchallenge.net/training-datasets"
raw="$root/data/raw"
mkdir -p "$raw"

declare -A SIZE_HINT=(
  [Fluo-N3DH-CE]="3.1 GB"
  [Fluo-N3DH-CHO]="98 MB"
  [Fluo-N2DL-HeLa]="66 MB"
  [DIC-C2DH-HeLa]="5.6 MB"
)

datasets=("$@")
if [[ ${#datasets[@]} -eq 0 ]]; then
  datasets=(Fluo-N3DH-CHO Fluo-N3DH-CE)
fi

for d in "${datasets[@]}"; do
  zip="$raw/$d.zip"
  echo "=== [$d] (约 ${SIZE_HINT[$d]:-未知}) ==="
  if [[ -f "$zip" ]] && unzip -tq "$zip" >/dev/null 2>&1; then
    echo "已存在且校验通过，跳过下载: $zip"
  else
    if [[ -f "$zip" ]]; then
      echo "已有文件校验失败（可能被中断），重新下载: $zip"
      rm -f "$zip"
    fi
    rm -f "$zip.partial"
    bash "$root/scripts/pdownload.sh" "$base/$d.zip" "$zip" "${CHUNKS:-16}"
    unzip -tq "$zip" >/dev/null && echo "zip 完整性校验通过" || { echo "zip 损坏: $zip" >&2; exit 1; }
  fi
  if [[ ! -d "$raw/$d" ]]; then
    echo "解压中..."
    unzip -q -o "$zip" -d "$raw"
  else
    echo "已解压，跳过: $raw/$d"
  fi
  echo "$d done"
done

echo "=== 记录校验和 ==="
cd "$raw"
{
  echo "# CTC 数据集校验和 生成时间: $(date -Is)"
  for d in "${datasets[@]}"; do
    [[ -f "$d.zip" ]] && md5sum "$d.zip"
  done
} > CHECKSUMS.txt
cat CHECKSUMS.txt
echo "原始数据目录: $raw"
