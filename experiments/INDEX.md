# 按任务进度整理的实验索引

当前阶段先读 [合并记录](RECORDS.md)；原始官方数值清单见 [ledger](ledger.json)。旧索引的完整原件在00归档，已按用户要求合并重排。

| 顺序 | 阶段 | 状态/用途 | 日常证据入口 |
| --- | --- | --- | --- |
| 1 | 仓库、数据、指标自校验 | 历史基础，保留摘要 | [E0.1](E0.1_repo_skeleton/notes.md)、[E0.2](E0.2_data_stats/notes.md)、[E0.3](E0.3_eval_selfcheck/notes.md) |
| 2 | 匈牙利/greedy基线、冻结nnU-Net训练 | 重要历史参照，完整记录归档 | [阶段记录](RECORDS.md)、[01归档](archives/README.md) |
| 3 | 早期A/B与E2–E4、P追踪 | 旧实现、偏离及探索，已合并归档 | [02归档](archives/README.md) |
| 4 | 早期C前端/真实检测/Oracle | 历史前端机制证据，已合并归档 | [03归档](archives/README.md) |
| 5 | E0.4–E0.6审计、重建、本地/云端探索 | 有已修复bug和被替代结果，不能作为新版消融 | [04归档](archives/README.md) |
| 6 | 冻结前端、encoder、检测映射 | 完成准备，无新增UNet训练 | [E0.7](E0.7_ideas_frontend_20261009/metrics_preparation_finished.json) |
| 7 | 真实预测实例正式双序列×双种子 | 当前可部署档，20次官方评测 | [E0.8最终指标](E0.8_ideas_final_v3_20261009/metrics_final.json)、[报告](../docs/reports/ideas_pipeline_rebuild_20261009.md) |
| 8 | GT marker Oracle + encoder | 隔离追踪贡献，5次官方评测 | [E0.9最终指标](E0.9_gt_oracle_encoder_20261009/metrics_final.json)、[报告](../docs/reports/gt_oracle_encoder_20261009.md) |
| 9 | NN掩码 + GT种子Oracle + encoder | 前端归属诊断，5次官方评测 | [E1.0最终指标](E1.0_nnunet_oracle_encoder_20261009/metrics_final.json)、[报告](../docs/reports/nnunet_oracle_encoder_20261009.md) |
| 10 | 项目文件整理 | 工程验收，无新增性能实验 | [E1.1](E1.1_repository_cleanup_20261009/notes.md) |

后续任务见 [项目进度](../docs/status.md)。正式证据只保留关键配置、模型指纹、官方日志、图和诊断；完整原件在05归档。所有本地压缩包、原commit和恢复方式见 [归档说明](archives/README.md)。
