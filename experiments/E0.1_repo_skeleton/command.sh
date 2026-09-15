#!/usr/bin/env bash
# E0.1 复现命令
set -euo pipefail
cd "$(dirname "$0")/../.."

git init -b main
git remote add origin https://github.com/HelpHenry240/CellTracker.git
mkdir -p src/celltracker/{data,detect,cost,ot,track,graph,gnn,eval,viz}
mkdir -p configs experiments figures tests scripts docs/reports
