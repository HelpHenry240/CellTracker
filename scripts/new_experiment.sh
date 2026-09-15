#!/usr/bin/env bash
# 新建一个实验记录目录，保证"留痕八件套"齐全。
# 用法: bash scripts/new_experiment.sh E2.3_sinkhorn_unbalanced "非平衡 OT 求解器"
set -euo pipefail

exp_id="${1:?用法: new_experiment.sh <实验目录名(如 E2.3_sinkhorn)> [描述]}"
desc="${2:-}"
root="$(cd "$(dirname "$0")/.." && pwd)"
dir="$root/experiments/$exp_id"

if [[ -e "$dir" ]]; then
  echo "错误: $dir 已存在。结果只增不改，请换新编号（如 -fix1）。" >&2
  exit 1
fi

mkdir -p "$dir"/{logs,artifacts,figures}

cat > "$dir/config.yaml" <<EOF
experiment_id: ${exp_id%%_*}
date: $(date +%F)
purpose: "$desc"
seed: 20260915
params: {}
status: running
EOF

cat > "$dir/command.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "\$(dirname "\$0")/../.."
# 在此固化本次实验的完整命令行
EOF
chmod +x "$dir/command.sh"

cat > "$dir/notes.md" <<EOF
# ${exp_id} ${desc}

## 目的

## 做了什么

## 观察（数字/现象）

## 结论

## 下一步
EOF

{
  echo "date: $(date -Is)"
  echo "host: $(hostname)"
  echo "platform: $(uname -srm)"
  echo "cpu_cores: $(nproc)"
  echo "python: $(python3 -V 2>&1)"
  echo "gpu: $( (nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null) || echo none )"
  echo "--- pip freeze ---"
  python3 -m pip freeze 2>/dev/null | head -100 || true
} > "$dir/env.txt"

git -C "$root" rev-parse HEAD > "$dir/git_commit.txt" 2>/dev/null || echo "no-commit" > "$dir/git_commit.txt"

echo "已创建: $dir"
ls -1 "$dir"
