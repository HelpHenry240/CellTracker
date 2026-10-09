#!/bin/bash
set -euo pipefail
# 后台命令和完整计划保存在artifacts中；不覆盖旧的预测实例实验。
bash experiments/E0.9_gt_oracle_encoder_20261009/artifacts/launch.sh

# 正式配置采用通过分裂安全审计的已有top-k保底；初版后台保留为诊断。
bash experiments/E0.9_gt_oracle_encoder_20261009/artifacts/launch_conservative.sh
