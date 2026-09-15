#!/usr/bin/env bash
# Phase 1+2 套件：在指定数据集/序列上跑基线 + OT 变体 + 参数扫描，并出汇总。
#
# 用法: bash scripts/run_phase12_suite.sh Fluo-N3DH-CE 01
set -euo pipefail

dataset="${1:-Fluo-N3DH-CE}"
seq="${2:-01}"
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

PY=/home/henry/miniconda3/envs/celltracker/bin/python
export PYTHONPATH=src

h5="data/interim/${dataset}_${seq}.h5"
if [[ ! -f "$h5" ]]; then
  echo "=== 构建内部表示: $h5 ==="
  "$PY" -m celltracker.data.build_dataset --dataset "data/raw/${dataset}" \
      --seq "$seq" --out "$h5"
fi

echo "=== E1.3 基线（贪心 / 匈牙利 / 匈牙利+匀速） ==="
"$PY" scripts/run_baseline.py --h5 "$h5" --dataset "$dataset" --seq "$seq" \
    --exp-id "E1.3_${dataset}_${seq}_greedy" --method greedy --max-dist 30 --official
"$PY" scripts/run_baseline.py --h5 "$h5" --dataset "$dataset" --seq "$seq" \
    --exp-id "E1.3_${dataset}_${seq}_hungarian" --method hungarian --max-dist 30 --official
"$PY" scripts/run_baseline.py --h5 "$h5" --dataset "$dataset" --seq "$seq" \
    --exp-id "E1.3_${dataset}_${seq}_hungarian_vel" --method hungarian --max-dist 30 \
    --velocity --official

echo "=== E2.2 平衡 vs 非平衡 OT ==="
"$PY" scripts/run_ot.py --h5 "$h5" --dataset "$dataset" --seq "$seq" \
    --exp-id "E2.2_${dataset}_${seq}_balanced" --eps 1.0 --eta 0 --r-max 30 --official
"$PY" scripts/run_ot.py --h5 "$h5" --dataset "$dataset" --seq "$seq" \
    --exp-id "E2.2_${dataset}_${seq}_unbalanced_tau1" --eps 1.0 --eta 0 --tau 1.0 \
    --r-max 30 --official

echo "=== E2.3 η（结构项）扫描 ==="
"$PY" scripts/run_sweep.py --h5 "$h5" --dataset "$dataset" --seq "$seq" \
    --exp-id "E2.3_${dataset}_${seq}_eta_eps" --x "eta=0,0.1,0.3,0.6" \
    --y "eps=1,5" --fixed "r_max=30" --official

echo "=== 完成；结果见 experiments/ 与 INDEX.md ==="
