# paperpipe —— 严格按 `ideas.pdf` 重建的 OT + GNN 细胞追踪 pipeline

这个目录是**"完完全全按论文口径"的第二条实现**，与原仓库 `src/celltracker` 并行存在：

* **符合原文的模块直接 import 复用**（OT 求解器、FGW 结构项、运动先验估速、CTC I/O、
  本地诊断、图批处理、实例拆分、实验留痕），避免"一个模块两套代码"造成口径漂移；
* **与原文有出入的模块在本包内重写**（帧内图与式6 的 W、式8 的 σ_s、
  式17-19 的逐尺度 λ、§1.5 的滑动窗口 tracklet、式23/24 的原始 Γ、
  η_death/η_birth、分裂的体积守恒、式25 的死特征、式27 的边特征布局、
  式29/33/34 的二分类、§2.0.1 的跨帧桥接边）。

每一条的"论文怎么写 / 原仓库怎么实现 / 本包怎么处理"见 [`FORMULA_MAP.md`](FORMULA_MAP.md)。
每个模块的 docstring 都抄了对应公式与出处（R1）。

---

## 1. 结构

```
paperpipe/
  configs/paper_default.yaml     # 全参数（含 PAPER/CALIB/ENG 标记）
  src/papertrack/
    measure.py      §1.2 式(1)-(7)   测度、帧内 kNN 图与边权 W、C_feat
    coupling.py     §1.3 式(8)-(14)  相邻帧（FGW / 非平衡）熵正则 OT
    multiscale.py   §1.4 式(15)-(19) 时间展开图 + 多尺度时间正则（交替优化）
    motion.py       §1.5 式(20)-(22) 两遍式运动先验（复用估速实现）
    tracklet.py     §1.5末/§1.6      滑动窗口 tracklet + 第二层 OT
    reconstruct.py  §1.6 式(23)-(24) 轨迹重建（OT 规则 / GNN 边决策两条路径）
    graph.py        §2.0.1 式(25)-(29) 时间展开图（含跨帧桥接边）
    model.py        §2.0.1 式(30)-(33) 边–节点消息传递 + sigmoid 边分类头
    train.py        §2.0.1 式(34)-(35) BCE + OT 一致性正则
    pipeline.py     主流程 + CTC 导出（含空洞帧补画）
    tracks.py       轨迹规范化（CTC 合法性、空洞登记）
    validate.py     CTC 提交格式校验（E2）
  scripts/
    run_paper_pipeline.py   # 端到端入口（OT 规则 或 GNN 决策 + 导出 + 可选官方指标）
    train_paper_gnn.py      # 训练 §2.0.1 的 GNN
    calibrate_params.py     # 论文未给数值的参数按物理量纲标定（R3）
  tests/test_papertrack.py  # 15 项回归测试（合成数据，秒级）
```

## 2. 怎么跑

```bash
# ① 未训练模型时：走 §1.6 的 OT 规则（式23/24）+ 落盘图数据集
python paperpipe/scripts/run_paper_pipeline.py \
    --h5   data/interim/Fluo-N3DH-CE_01_pred_v2.h5 \
    --gt-h5 data/interim/Fluo-N3DH-CE_01.h5 \
    --seq 01 --exp-id P1_otrule_ce01 \
    --dump-graphs data/interim/pg_graphs_01

# ② 训练 §2.0.1 的 GNN（本地 CPU，分钟级）
python paperpipe/scripts/train_paper_gnn.py \
    --graphs data/interim/pg_graphs_01 --out paperpipe/runs/gnn_01 --epochs 60

# ③ 端到端（GNN 决策）+ 导出 CTC 提交（会自动跑格式校验）
python paperpipe/scripts/run_paper_pipeline.py \
    --h5 ... --gt-h5 ... --seq 01 --exp-id P2_gnn_ce01 \
    --ckpt paperpipe/runs/gnn_01/best.pt

# ④ 加 --official 即在云端跑官方 DET/SEG/TRA（复用 scripts/cloud_eval.py）
```

参数标定：

```bash
python paperpipe/scripts/calibrate_params.py \
    --h5 data/interim/Fluo-N3DH-CE_01_pred_v2.h5 \
    --gt-h5 data/interim/Fluo-N3DH-CE_01.h5 --out paperpipe/calib/ce01.json
```

测试：`python -m pytest paperpipe/tests -q`（已并入仓库 pytest 的 `testpaths`）。

## 3. 与端到端测试（云端 nnU-Net）的衔接

检测前端完全复用原仓库，不改：

```
云端 nnU-Net 推理 (*.nii.gz)
  → scripts/predict_to_h5.py（实例拆分 + 强度统计 + gt_label 映射）
  → paperpipe/scripts/run_paper_pipeline.py --h5 <pred.h5> --gt-h5 <GT.h5>
  → experiments/<exp>/artifacts/submission/<seq>_RES/ → scripts/cloud_eval.py
```

**R11（训练/测试检测来源同分布）**：建图脚本 `--dump-graphs` 与端到端推理必须用**同一个**
`--h5`（预测实例），`--gt-h5` 只提供血缘与评测真值。

## 4. 需要你拍板的两件事

1. **空洞帧补画（`hole_policy`）**：CTC 格式不允许轨迹空洞，而论文 §2.0.1 要求轨迹不被
   截断。默认 `fill`（平移复制源实例体素，无处可放时覆盖），备选 `split`（截断但绝对合法）。
2. **θ_Γ / η_death / η_birth 的标定口径**：默认按"占源质量的比"定义（论文只给符号不给数值），
   可用 `theta_gamma=<绝对値>` 回到原文的单一标量形式。

详见 `FORMULA_MAP.md` §3 的 E-2/E-3/E-4。

## 5. 已实测（合成 + 真实数据小范围冒烟，非结论）

* 15/15 回归测试通过；仓库原有 89 项测试不回归；
* CE seq01 预测实例、帧 120–130（11 帧）：两条决策路径都能产出**格式校验合法**的 CTC 提交；
* 真实数据冒烟暴露并修掉 4 个问题：多尺度精度的 NaN/溢出与 16× 超时、tracklet 重编号后
  轨迹 id 为 0（CTC 非法）、桥接边过度触发（537→22 个空洞）、空洞无处补画时的回退。
* 这些数字只是"链路能跑通"的证据，**不是精度结论**（R4/R9：本地指标只判方向，
  结论要用双序列 + 官方指标 + 噪声地板）。
