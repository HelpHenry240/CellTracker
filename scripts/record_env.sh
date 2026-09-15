#!/usr/bin/env bash
# 刷新某个实验目录的 env.txt
set -euo pipefail
exp_id="${1:?用法: record_env.sh <实验目录名>}"
root="$(cd "$(dirname "$0")/.." && pwd)"
dir="$root/experiments/$exp_id"
{
  echo "date: $(date -Is)"
  echo "host: $(hostname)"
  echo "platform: $(uname -srm)"
  echo "cpu_cores: $(nproc)"
  echo "python: $(python3 -V 2>&1)"
  echo "gpu: $( (nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null) || echo none )"
  echo "--- pip freeze ---"
  python3 -m pip freeze 2>/dev/null | head -150 || true
} > "$dir/env.txt"
echo "已刷新 $dir/env.txt"
