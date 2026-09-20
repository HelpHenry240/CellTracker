# C1 云端日志：待补

C1 推理在云端直接执行，**日志未落盘到实验目录**（当时未记录，属留痕缺口，
C0 才发现）。已知的替代证据：

- 产物完整性与命名可核验：`data/interim/preds_nnunet/` 385 个 `*.nii.gz`
  （seq01 195 + seq02 190），单帧形状 `(35, 512, 708)`；
- 训练侧证据：`experiments/E5.1_nnunet_ce/notes.md`（每 epoch 26.5s、GPU 96%、
  Dice 0.9605）；
- 下游证据：`data/interim/pred_report_01.json`（逐帧实例/标记画像）。

**待办（C4 阶段云服务器开机时）**：核对云端 shell 历史/`logs/predict.log`，
把原始命令与日志回填到本目录，随后删除本文件。
