# 运行、配置与消融

在仓库根目录使用已有 `celltracker` 环境。源码部署须包含完整 `src/`、`scripts/` 与 `configs/`，不再单独打包旧 `paperpipe/`。

```bash
conda activate celltracker
python -m pytest
python scripts/run_pipeline.py --list-modules
python scripts/run_pipeline.py --config configs/ce_calibrated_20261009.yaml --list-modules
```

`paper_default.yaml` 是完整方法模板，需针对输入来源标定。`ce_calibrated_20261009.yaml` 是E0.8真实预测实例、k=1.6、保留孤立实例的现有预设；不能直接用于marker或新Oracle。`--set section.field=value`覆盖参数，`--ablate name`按当前状态反向切换。

已有前端产物的实例H5路径因本地/云端位置不同，可先按实际位置设置以下任务变量：

```bash
CT_DETECTIONS=data/interim/Fluo-N3DH-CE_01_pred_resplit_v2.h5
CT_TRUTH=data/interim/Fluo-N3DH-CE_01.h5
CT_ENCODER=data/interim/encoder_01_resplit_v2.npz
CT_CONFIG=configs/ce_calibrated_20261009.yaml
```

标定应先打印物理位移、原始C、Γ与候选召回分布；只用seq01设置阈值：

```bash
python scripts/calibrate_params.py --h5 "$CT_DETECTIONS" --gt-h5 "$CT_TRUTH"   --config "$CT_CONFIG" --set node.encoder_feat_path="$CT_ENCODER"   --out data/interim/new_calibration.json --apply-out data/interim/new_calibration.yaml
```

独立训练时先完整上游建图，再运行统一训练入口。以下为新实验命令模板；完整队列应按当前服务器实际耗时在云端后台执行，本地只做小样本：

```bash
python scripts/run_pipeline.py --h5 "$CT_DETECTIONS" --gt-h5 "$CT_TRUTH" --seq 01   --exp-id NEW_build --config "$CT_CONFIG" --set node.encoder_feat_path="$CT_ENCODER"   --dump-graphs data/interim/graphs_NEW --build-only
python scripts/train_gnn.py --graphs data/interim/graphs_NEW --out data/interim/model_NEW   --config "$CT_CONFIG" --epochs 60 --seed 20261008 --device cpu --exp-id NEW_train
python scripts/run_pipeline.py --h5 "$CT_DETECTIONS" --gt-h5 "$CT_TRUTH" --seq 01   --exp-id NEW_eval --config "$CT_CONFIG" --set node.encoder_feat_path="$CT_ENCODER"   --ckpt data/interim/model_NEW/best.pt --official
```

复用现有E0.8权重时保持同一前端及上游配置，仅替换数据与对应encoder位置；权重可在 `experiments/E0.8_ideas_final_v3_20261009/artifacts/models/conservative_k16/` 找到。两类Oracle的正式配置和权重分别在E0.9/E1.0中。改变来源或上游契约后必须独立重训。

矩阵入口 `scripts/run_ablation_matrix.py` 默认只生成计划，`--execute`才执行，`--resume`恢复已完成任务。传入 `--h5-01/02`、`--gt-01/02`、`--encoder01/02`、`--config`、`--out`、`--exp-prefix`，使用两个种子并独立训练每项变体。`--recalibrate`只在seq01标定各变体。具体参数以 `--help` 为准。

云端长任务使用 `nohup`、独立日志与PID；GPU启动前/运行中/结束后检查 `nvidia-smi`，完成后核验非零checkpoint。SSH连接由 `scripts/cloud_run.sh` 与本地凭据文档管理；备用连接使用 `CT_CLOUD_CONN=backup`，凭据不入库。nnU-Net重训需用户确认；本次整理未启动云任务。

模块开关与实现范围见 [公式映射](../architecture/formula_map.md)，结果口径见 [评测说明](evaluation.md)。
