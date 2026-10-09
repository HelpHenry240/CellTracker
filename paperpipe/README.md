# ideas pipeline

本目录是 `ideas.pdf` 方法的主实现。仓库入口 `scripts/run_ideas_pipeline.py` 与
`paperpipe/scripts/run_paper_pipeline.py` 调用同一实现。外层旧追踪脚本保留用于历史基线。

```text
nnU-Net 语义掩码 → 实例 H5 / 完整 GT 身份映射 / 冻结 encoder 特征
 → 测度与物理空间图（式1–7）
 → 两遍运动先验 / 相邻 FGW 或 OT（式8–14、20–22）
 → 多尺度时间精炼（式17–19）
 → 连通的时间图 / 边分类 GNN（式25–35）
 → 生死、分裂、冲突与不确定性重建（式23–24、§2.0.1）
 → 滑窗高置信 tracklet / 二层 OT（§1.5）
 → CTC 导出 / 格式校验 / 官方 DET、SEG、TRA
```

配置集中在 `configs/`，公式与实现对应见 [FORMULA_MAP.md](FORMULA_MAP.md)。
`paper_default.yaml` 和 `paper_e2e_ce.yaml` 是完整模块模板，需要设置 encoder 侧车路径并
标定新设置的绝对质量阈值。历史配置和历史指标不代表当前重建版本的性能。

## 数据准备

预测检测与训练检测来自同一分割流程；`--gt-h5` 只提供监督和评测真值。

```bash
# 为已有预测实例重建多数覆盖映射；保留欠分割实例覆盖的全部 GT 身份。
python scripts/relabel_detections.py --input pred.h5 --gt-h5 gt.h5 --out pred_v2.h5

# 用现有 nnU-Net 冻结 encoder 提取特征，无需重训；逐帧、逐块、可断点续提。
python scripts/nnunet/export_encoder_features.py --h5 pred_v2.h5 \
  --images /path/imagesTs_eval --model /path/trained_model --seq 01 \
  --out encoder01.npz --device cuda
```

侧车保存帧号、实例 label、预处理和模型指纹。相同实例数量不保证顺序一致，因此载入时
按 label 对齐；旧侧车必须通过物理质心的一一匹配。缺失特征不允许静默回退。

## 标定、建图、训练和推理

```bash
# 先跑完整上游，记录 µm、µm³、C、原始 Γ、行列质量及候选真实边召回。
python paperpipe/scripts/calibrate_params.py --h5 pred_v2.h5 --gt-h5 gt.h5 \
  --config paperpipe/configs/paper_e2e_ce.yaml \
  --set node.encoder_feat_path=encoder01.npz \
  --cache-dir data/interim/calibration01 --out experiments/new_calibration/distributions.json \
  --apply-out experiments/new_calibration/calibrated.yaml

python scripts/run_ideas_pipeline.py --h5 pred_v2.h5 --gt-h5 gt.h5 --seq 01 \
  --config experiments/new_calibration/calibrated.yaml --exp-id new_build01 \
  --dump-graphs data/interim/new_graphs01 --build-only --cache-dir data/interim/new_cache01

python paperpipe/scripts/train_paper_gnn.py --graphs data/interim/new_graphs01 \
  --config experiments/new_calibration/calibrated.yaml --out experiments/new_model/artifacts/model \
  --epochs 60 --device cuda

python scripts/run_ideas_pipeline.py --h5 pred_v2.h5 --gt-h5 gt.h5 --seq 01 \
  --config experiments/new_calibration/calibrated.yaml --exp-id new_eval01 \
  --ckpt experiments/new_model/artifacts/model/best.pt --device cuda --official \
  --official-tools /root/EvaluationSoftware/Linux --official-gt-dir /path/01_GT
```

seq02 使用同一组标定参数和 seq01 权重，只更换检测、GT、encoder 路径。权重会校验上游
配置、检测来源、特征 schema 和 encoder 指纹。默认按时间块分训练/验证，并排除跨边界的
重叠上下文。训练每轮保存优化器和随机状态；`--resume` 可续训。建图使用
`--resume-graphs` 检查并补齐已有图。缓存不能跨配置或输入文件复用。

`--official` 默认通过仓库的 `cloud_eval.py` 调用远端；在官方二进制所在机器上应同时设置
`--official-tools` 与 `--official-gt-dir`，避免再通过 SSH 上传。格式失败、程序失败或指标
缺失都返回失败。官方评测不允许 `--no-validate` 或部分帧。

## 消融

```bash
python scripts/run_ideas_pipeline.py --list-modules
python scripts/run_ideas_pipeline.py --config calibrated.yaml --list-modules

# 只生成计划，不启动算力任务。
python paperpipe/scripts/run_ablation_matrix.py --config calibrated.yaml \
  --h5-01 pred01.h5 --h5-02 pred02.h5 --gt-01 gt01.h5 --gt-02 gt02.h5 \
  --encoder01 encoder01.npz --encoder02 encoder02.npz \
  --out experiments/new_matrix --ablations fgw motion multiscale tracklet gnn \
  --seeds 20261008 20261009 --official
```

注册表在 `runtime/ablation.py`。默认开启项测关闭，默认关闭项测开启。每个变体独立建图、
每个种子独立训练；关闭整个 OT 时一起关闭依赖质量的消费者。仅关闭 GNN 的变体直接用
OT 重建，无可学习模块。`--execute` 顺序运行，`--resume` 按状态文件恢复。全序列任务必须
在云端后台运行，启动前、训练中和结束后检查 GPU。

绝对质量阈值随耦合机制变化。对会改变 OT 质量或代价的变体，使用 `--recalibrate`，
在 seq01 重新测量分布并选择同一分位规则，seq02 沿用这些参数。`--skip-baseline`
用于已有同配置双序列基准证据的后续矩阵。比较时同时报告机制开关与派生阈值，
不能把基准的绝对阈值直接套到质量尺度不同的变体上。

CTC 标签以标准 TIFF zlib 无损压缩逐帧落盘。压缩不改变像素值；仍须由官方程序
实际读取并评测。已完成实验不会因为存储优化重新写入。

## 工程适配与验证边界

论文的原始质量乘积与绝对 Γ 阈值是默认形式。条件概率乘积、相对质量阈值、top-k 保底、
自适应 ε、强度外观与残差 GNN 都明确作为可切换的工程对照。CTC `fill` 用平移的源实例
补画空洞，仅写背景；无可用体素时报错。`split` 是显式拆段对照。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
```

测试验证公式、梯度、接口、失败模式和恢复行为。本地真实数据小样本验证不用于性能结论。
新版本的效果须由双序列官方指标、多种子和当前设置的噪声地板决定。

使用重新标定协议时，基准也必须重新标定一次；仅比较同协议基准与变体。
若先跑过初始配置，再启动 `--skip-baseline --recalibrate`，须补一个同协议的
双种子、双序列基准，不能把重新标定本身的收益归到某个模块上。
