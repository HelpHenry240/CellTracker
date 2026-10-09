#!/bin/bash
set -euo pipefail
# 完整复现命令将保存在 artifacts 的后台作业脚本与云端计划中。

# 原后台作业、恢复与四档完整参数由 artifacts/*.py/*.sh 和各 plan.json 保存。
# 只复现已生成的统计/图，不重复训练或覆盖正式实验。
python experiments/E0.8_ideas_final_v3_20261009/artifacts/summarize_final.py \
  --root experiments/E0.8_ideas_final_v3_20261009/artifacts/cloud_final \
  --repository . --out /tmp/ideas_metrics_recomputed.json
python experiments/E0.8_ideas_final_v3_20261009/artifacts/plot_final.py \
  --summary experiments/E0.8_ideas_final_v3_20261009/metrics_final.json \
  --sources experiments/E0.7_ideas_frontend_20261009/artifacts/source_reports/ideas_detection_sources_20261009.json \
  --out /tmp/ideas_figures_recomputed
