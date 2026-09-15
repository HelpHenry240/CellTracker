# E0.1 仓库初始化与实验管理脚手架

## 目的

建立可复现、可留痕的工程骨架，确保后续每个实验都有：
config / 命令 / 环境 / git commit / 指标 / 日志 / 图 / 结论 八类产物。

## 做了什么

1. `git init -b main`，关联 remote `origin = https://github.com/HelpHenry240/CellTracker.git`
2. 建立目录骨架：`src/celltracker/{data,detect,cost,ot,track,graph,gnn,eval,viz}`、`configs/`、
   `experiments/`、`figures/`、`scripts/`、`tests/`、`docs/reports/`
3. `.gitignore`：排除凭据目录 `本地文档/`、数据、权重、缓存
4. `README.md`、`requirements.txt`、`environment.yml`
5. `experiments/INDEX.md` 作为实验总表

## 观察

- 本地 `.git` 目录在沙箱中为只读挂载，git 写操作需提权执行（已获批准 `git` 前缀规则）。
- 本机无 GPU；CPU 22 核 / 15 GB 内存，磁盘余量约 898 GB。
- 本机 Python 3.13，无 torch；计划用 conda 建 `celltracker`（py3.11）环境。

## 结论

骨架就绪，可进入 P0.2（数据管线）。

## 下一步

P0.2：CTC 3D 数据下载 + 校验 + 转内部格式 `dataset.h5`。
