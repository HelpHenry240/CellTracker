# 正式 GNN 推理权重

四档配置各两个独立训练种子，共八份推理权重，原模型逐张量一致。
`manifest.json` 与各权重旁的 JSON 保留 MD5/SHA256、原断点散列、训练 epoch、
输入契约和正式源码版本。每份约434KB；正式数值源包是88bf5ca。

推荐：`conservative_k16/seed20261008_inference.pt`，配置为
`paperpipe/configs/ce_calibrated_20261009.yaml`；固定种子用于复现，另一种子用于波动检查。
seq01/seq02分别设置对应 encoder 侧车路径，阈值和模型保持相同。

这些文件仅供推理。完整的优化器/随机状态断点仍在备用3
`/root/autodl-tmp/CellTracker_rebuild_20261008/experiments/P7_ideas_v3_<profile>_matrix_20261009/models/baseline/seed<seed>/`。
训练恢复使用其中的 `last.pt`。导出脚本不会覆盖或删除原断点。
模型会拒绝不匹配的分割来源、上游开关、物理间距、encoder 指纹和特征布局。
