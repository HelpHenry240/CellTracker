# nnU-Net掩码Oracle + encoder的冻结GNN推理权重

两个种子各独立训练60轮，只使用seq01标定/训练。权重分别位于
`nnunet_oracle_encoder/seed20261008_inference.pt`和
`nnunet_oracle_encoder/seed20261009_inference.pt`，每份446586字节。

检测输入是保留全部nnU-Net预测前景的GT身份种子实例化；不是GT marker掩码。
种子转换版本为`preserve_label_ids_v2`；128维冻结stage2 encoder特征在Oracle实例质心重新采样。
输入契约防止与旧Oracle、普通预测实例或GT marker权重混用。

`manifest.json`及每份权重的JSON侧车记录md5、sha256、原检查点sha256、选择轮次、源码版本和输入契约。
两份权重在云端与原训练检查点逐张量核验，在本地CPU加载、有限值和文件指纹核验通过；见`../../logs/model_portability.json`。
该目录只含推理必需参数，优化器和完整训练断点仍在云端：
`/root/CellTracker_nnunet_oracle_encoder_20261009/experiments/P9_nnunet_oracle_encoder_matrix_20261009/models/baseline/seed<seed>/`。

训练源码：1d84b77569010e73fc0aeb32ecedde1bcfa86bb8；该版本180文件散列包含在权重payload中。
正式复现参数及任务命令见`../evidence_formal/experiments/P9_nnunet_oracle_encoder_matrix_20261009/`。
