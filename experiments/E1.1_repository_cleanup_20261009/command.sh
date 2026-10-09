#!/usr/bin/env bash
# 本次实际执行步骤的留痕。整理已完成，勿在当前目录盲目重跑一次性迁移。
# 若复演，先将00归档恢复到隔离目录，并拷贝本阶段整理工具；详见notes.md。
set -euo pipefail
python experiments/E1.1_repository_cleanup_20261009/artifacts/compact_history.py
python experiments/E1.1_repository_cleanup_20261009/artifacts/integrate_sources.py
python experiments/E1.1_repository_cleanup_20261009/artifacts/organize_documents.py
python experiments/E1.1_repository_cleanup_20261009/artifacts/verify_cleanup.py
python -m pytest -q -o addopts=''
# 干净Git归档导出后的完整pytest结果单独落日志；不启动云训练或官方评测。
