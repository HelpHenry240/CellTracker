# CellTracker

按 `ideas.pdf` 实现的 3D 细胞追踪：预测实例 → 冻结 nnU-Net encoder → OT/FGW → 多帧图 → GNN → 轨迹重建 → 官方 CTC 评测。

源码统一位于 `src/`：`papertrack` 是当前论文主链路，`celltracker` 提供共享组件及明确保留的历史对照 API。已移除重复 vendor 与 `paperpipe/` 目录。

| 入口 | 内容 |
| --- | --- |
| [文档导航](docs/README.md) | 架构、运行、评测、决策与研究资料 |
| [项目进度](docs/status.md) | 已完成验证、结论边界及后续任务 |
| [运行说明](docs/guides/running.md) | 统一 CLI、训练、配置与消融 |
| [任务实验索引](experiments/INDEX.md) | 按进度排列的当前证据 |
| [合并实验记录](experiments/RECORDS.md) | 历史到当前的阶段结论 |
| [脚本清单](scripts/README.md) | 保留脚本的用途 |

```bash
conda activate celltracker
python -m pytest
python scripts/run_pipeline.py --list-modules
```

完整算法模板为 `configs/paper_default.yaml`；当前真实预测实例预设为 `configs/ce_calibrated_20261009.yaml`。数据、冻结 nnU-Net 大权重和完整历史压缩包不随 Git 提交；当前正式 GNN 小权重与逐项指纹保留在 `experiments/`。
