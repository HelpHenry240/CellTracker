# CellTracker

基于**时序网络 + 最优传输（OT）**的显微细胞追踪方法实现与实验。

方法骨架：nnU-Net 分割 → 每帧检测构成经验测度 μᵗ + 帧内 kNN 图 →
相邻帧（非平衡）OT（C_feat + η·FGW 结构项 + ε 熵 + τ·KL）→ 多尺度时间一致性
→ 时间展开图 → GNN 边级校正 → 轨迹与谱系重建 → tracklet 二层 OT。

理论来源见 `ideas.pdf`（仓库根目录）。

## 当前状态

2026-10 重建入口为 `python scripts/run_ideas_pipeline.py`，使用 `paperpipe` 中的统一实现。
完整方法、配置、消融和恢复说明见 [paperpipe/README.md](paperpipe/README.md)。
本地公式与链路回归、真实双序列小样本已经通过；重建版本的全量 GNN、双序列官方指标
和消融正在单独验证。历史结果仍保存在 experiments，不作为新版本已完成的证据。

## 计划与约定

- 完整实验计划：[`docs/00_实验计划_v1.md`](docs/00_实验计划_v1.md)（v2，含两周日程）
- 数据集：**CTC，以 3D 为主**（主力 Fluo-N3DH-CE，其次 Fluo-N3DH-CHO，2D HeLa 作快速代理）
- 评测：一律使用 CTC 官方二进制计算 SEG / DET / TRA
- 每个实验的产物见 `experiments/<ID>_<slug>/`，总表见 `experiments/INDEX.md`
- 结果只增不改；修正实验另开 `-fix1` 目录
- 本地算力：22 核 CPU，无 GPU；云 GPU 仅用于训练/大规模重跑

## 快速开始

```bash
# 1. 环境
conda env create -f environment.yml   # 或 pip install -r requirements.txt
conda activate celltracker

# 2. 数据（需联网，CTC 官方地址）
bash scripts/download_data.sh

# 3. 自检：把 GT 当预测喂给官方评测器，SEG 必须 = 1.0
bash scripts/run_eval_selfcheck.sh
```

## 目录

```
src/celltracker/{data,detect,cost,ot,track,graph,gnn,eval,viz}
configs/            每个实验一个 yaml
experiments/        实验记录（config/命令/日志/指标/图/结论）
figures/            跨实验汇总图（论文用）
scripts/            数据下载、云机上下、评测封装
tests/              单测（OT 正确性、评测封装、格式转换）
```

## 凭据

云服务器与 GitHub 凭据存放于本地 `本地文档/`，**已加入 `.gitignore`，不入库**。
