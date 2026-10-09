# GT marker Oracle + encoder 的正式 GNN 推理权重

`gt_marker_encoder_conservative/` 中保存两个独立种子的推理权重，每份447738字节。
导出后逐张量验证与备用3原始best.pt完全一致；manifest及侧车JSON保留MD5、SHA256、
原训练断点散列、输入契约、epoch及训练源码版本500322cc。

使用 `artifacts/evidence_formal/experiments/P8_gt_marker_encoder_conservative_matrix_20261009/configs/baseline_calibrated.yaml`，
检测为GT marker H5，encoder侧车为本轮对应序列的NPZ；训练只来自seq01。
seq02仅更换检测和encoder入口，其他阈值及模型保持相同。完整命令见正式plan.json。
本权重不适用于预测实例输入，模型会检查来源与配置契约。

完整优化器与随机状态断点保留在云端：
`/root/CellTracker_oracle_encoder_conservative_20261009/experiments/P8_gt_marker_encoder_conservative_matrix_20261009/models/baseline/seed<seed>/`。
恢复训练使用原last.pt；这些小文件仅用于推理。nnU-Net始终冻结，没有重训。
