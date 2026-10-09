#!/bin/bash
# 启动脚本已存 artifacts；远端的 SOURCE_VERSION.json 记录发布指纹。
CT_CLOUD_CONN=backup3 bash scripts/cloud_run.sh "bash /root/autodl-tmp/CellTracker_rebuild_20261008/launch_baseline_20261009.sh"
