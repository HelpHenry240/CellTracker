#!/bin/bash
set -euo pipefail
# 初版历史启动命令（已取消并归档，复核用）：
# bash experiments/E0.9_gt_oracle_encoder_20261009/artifacts/launch.sh
# 正式版使用通过分裂边安全审计的top-k=3配置，建新图并独立训练。
# 首次运行需准备备用3目录及输入，完整参数见artifacts/evidence_formal下的plan.json。
bash experiments/E0.9_gt_oracle_encoder_20261009/artifacts/launch_conservative.sh
# 完整结束后可按独立脚本启动汇总，保留后台PID和日志：
# bash experiments/E0.9_gt_oracle_encoder_20261009/artifacts/launch_postprocess.sh
