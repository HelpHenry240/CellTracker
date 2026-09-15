#!/usr/bin/env bash
# 在云服务器上执行命令（密码登录，密码从 本地文档/ssh&key 读取）。
# 用法: bash scripts/cloud_run.sh "nvidia-smi"
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
exec /home/henry/miniconda3/bin/python3 "$root/scripts/cloud_run.py" "$@"
