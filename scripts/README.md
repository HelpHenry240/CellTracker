# 脚本职责

| 分类 | 脚本 | 用途 |
| --- | --- | --- |
| 主链路 | `run_pipeline.py` | 统一论文推理/监督建图/CTC导出/官方评测 |
| 学习 | `train_gnn.py` | 配置驱动的论文GNN训练与恢复 |
| 消融 | `run_ablation_matrix.py` | 双序列多种子计划、独立训练与恢复 |
| 标定 | `calibrate_params.py` | 完整上游的绝对阈值、物理分布与候选损失 |
| 实例前端 | `predict_to_h5.py`、`relabel_detections.py`、`resplit_detections.py`、`calibrate_instance_split.py` | 流式实例化、GT映射、再切与拆分诊断 |
| 冻结分割/特征 | `nnunet/` | 数据转换、冻结推理、encoder侧车、训练入口；重训另行确认 |
| 官方工具 | `cloud_eval.py`、`analyze_tra_log.py` | 官方评测与AOGM分解 |
| 云端连接 | `cloud_run.py`、`cloud_run.sh` | 执行与传输，凭据只从本地文档读取 |
| 对照 | `run_baseline.py` | 匈牙利/greedy历史基线 |
| 历史诊断 | `diagnostics/detection_ceiling.py` | 早期体素门限、单GT映射诊断；仅回归/历史用途，不定当前结论 |
| 数据/留痕 | `download_data.sh`、`pdownload.sh`、`new_experiment.sh`、`record_env.sh` | 已有数据获取与八件套辅助 |
| 归档 | `manage_archives.py` | 打包、逐文件校验及恢复到新目录 |

已删除的旧编排器、旧GNN/OT参数扫描、一次性作图、独立帧对旧候选扫描及旧paperpipe打包脚本，原件均在00归档。删除/迁移清单见 `experiments/E1.1_repository_cleanup_20261009/artifacts/source_migration.json`。实验内的一次性执行脚本同样完整归档，不再充当日常入口。
