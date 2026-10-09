# 测试流程详解：本地测试 与 CTC 官方测试

> 生成日期：2026-09-22 · 代码基线：`main` @ 44+ 次提交、82 项单测全绿
> 依据：`scripts/eval_pipeline.py`、`scripts/run_gnn.py`、`scripts/cloud_eval.py`、
> `src/celltracker/eval/{local_metrics,ctc_io}.py`、`experiments/INDEX.md`、
> `docs/reports/{phase1-2_CE_report,day2_summary,day3_summary}.md`
> 路径约定：本文所有路径均相对仓库根 `/home/henry/ot_idea/CellTracker`

本文回答一个问题：**模型跑完之后，到底怎么被测？** 分两条独立、互补的通道——
本地快速测试（方向判断）与 CTC 官方测试（权威定论）。两条通道的边界是本仓库方法论的核心。

---

## 0. 一句话总览

| | 本地测试 | CTC 官方测试 |
| --- | --- | --- |
| **在哪里跑** | 本机（WSL2，22 核 CPU，无 GPU） | 云服务器（Linux 内核 5.15，**官方二进制**） |
| **算什么** | 自写 SEG + 追踪诊断量 + 误差归属工具 | 官方 `SEGMeasure` / `DETMeasure` / `TRAMeasure` |
| **多快** | 秒级~分钟级，可逐帧迭代 | 分钟级（含打包上传、解压、评测、回传） |
| **干什么用** | **只看方向**（R4）：改这个方向对不对、错在哪一环 | **下结论**：写进论文/汇报的数字全以它为准 |
| **入口脚本** | `pytest` / `scripts/run_pipeline.py` / `scripts/eval_pipeline.py`（不加 `--official`） | `scripts/eval_pipeline.py --official`（内部调 `scripts/cloud_eval.py`） |
| **可信度依据** | E0.3 自检：本地 SEG 与官方**逐位一致（1e-6）** | E0.3 自检：官方二进制在云端复现文档基准值 |

> **铁律 R4**：调参与结论一律以官方 CTC 指标为准；本地指标只用于方向判断，不得用于定论。
> **铁律 R10**：显著性以**噪声地板 0.0007** 为标尺；ΔTRA < 0.001 只能说"与噪声不可区分"。

---

## 1. 被测对象：链路最终产出什么

无论哪条通道，被测的都是同一个东西——**一份合法的 CTC 提交目录**。
理解测试流程前，先要理解这个产物。

主控函数：`src/celltracker/pipeline/runner.py::run_pipeline`，按论文顺序串联七个阶段：

```
S0 检测来源（GT 标记 / nnU-Net 预测实例）
 → S1 帧内测度 + kNN 图（式1-7）
 → S2 相邻帧 OT（式8-14,22）
 → S3 运动先验、S4 多尺度时间正则（式17-21）
 → S5 时间展开图（式25-29）
 → S6 决策：GNN 边分类（式30-35）或 OT 规则
 → S7 第二层 tracklet OT（§1.6）
 → S8 轨迹重建 + CTC 格式校验
 → S9 评测
```

最终产物（由 `src/celltracker/eval/ctc_io.py::ResultWriter` 流式写出）：

```
<exp>/artifacts/submission/<seq>_RES/
  mask000.tif … maskNNN.tif     每帧标签体数据，3D 为多页 TIFF 栈，像素值 = 轨迹 id
  res_track.txt                 每行：L B E P（标签 / 起始帧 / 结束帧 / 父标签）
```

**这份产物同时编码了分割和追踪**：`mask*.tif` 里的像素值就是轨迹标签，
所以官方三个度量都从同一份目录里读。

### 1.1 检测来源：必须分开报告的三档

同一套追踪 pipeline，检测从哪里来，决定了报告里那一列的含义（AGENTS.md §8.3）：

| 档位 | 检测来源 | 用途 | 报告口径 |
| --- | --- | --- | --- |
| GT 标记上界 | `_GT/TRA` 标记点重着色 | 隔离追踪贡献，DET 恒 1.0 | SEG 无意义，只报 DET/TRA |
| 真实检测 | nnU-Net 掩码 → 实例拆分 → h5 | 端到端真实设定 | DET/SEG/TRA 三项全报 |
| 合成退化 | 对 GT 掩码人工加 FN/FP/过欠分割 | 鲁棒性 | 4 档退化 × TRA 曲线 |

对应两个独立入口：

- `--h5`：**检测/实例来源**（GT 标记，或预测实例）；
- `--gt-h5`：**评测真值**（提供 GT 标记用于诊断与 SEG）。

> **铁律 R11**：训练与测试的检测来源必须同分布。预测检测的 h5 必须带 `gt_label` 映射
> （每条检测对应哪个 GT 轨迹 id），才能在同一来源上训练与推理。

---

## 2. 本地测试

本地测试分三层：单测 → 链路运行 + 本地指标 → 误差归因。**全部只用于方向判断。**

### 2.1 第一层：单元测试（`pytest`）

```bash
pytest -q          # 当前 82 项，必须全绿（E6：修 bug 必须补回归测试）
```

覆盖分布（`tests/`，16 个文件）：

| 测试文件 | 守护的关键行为 |
| --- | --- |
| `test_ctc.py` | CTC 命名约定、3D 逐切片金标准、`L B E P` 解析 |
| `test_ot.py` | Sinkhorn 平衡/非平衡解，与线性规划对照 |
| `test_fgw.py` | FGW 结构项与梯度（有限差分校验）、目标单调下降 |
| `test_multiscale.py` / `test_multiscale_stage.py` | 式(19) 正则、`enabled=False` 恒等映射 |
| `test_motion.py` | 式(20)(21) 速度正确性、**死特征检查**、两遍式等价性 |
| `test_instances.py` | 物理间距 EDT、`seeds=` 路径、标签面积判据 |
| `test_tracking_consistency.py` | **幽灵轨迹回归**、父子关系、`finalize_tracks` |
| `test_pipeline_config.py` / `test_pipeline_runner.py` | 消融开关语义、七阶段串联 |
| `test_gnn_adapter.py` / `test_fusion.py` / `test_decision_and_ablation.py` | GNN 接入、冲突消解、融合分解 |
| `test_detection_ceiling.py` | 检测层天花板 U0–U3 口径 |

### 2.2 第二层：跑链路 + 本地指标

本地指标实现在 `src/celltracker/eval/local_metrics.py`，有两类。

**(a) 本地 SEG —— 与官方逐位一致**

`seg_measure(gt_dir, res_dir)`：对每个 GT 目标找重叠最大的结果目标；
若 `|R∩S| > 0.5·|R|` 记 Jaccard，否则记 0；取所有 GT 目标的平均。
支持 3D 逐切片金标准（`man_seg_T_Z.tif`）。

> E0.3 自检确认：seq01 = 0.232874、seq02 = 0.443686，**与官方文档基准值精确匹配（1e-6）**，
> 所以本地 SEG 可以直接当官方 SEG 用（但正式数字仍以云端为准）。

**(b) 追踪诊断量 —— 只用于方向判断**

`StreamingDiagnostics` 逐帧累积（3D 数据必须流式，见 E1）：

- FN / FP、detection_recall / precision
- ID switch（同一预测轨迹跨帧匹配到不同 GT）
- fragmentation（一条 GT 轨迹被多少条预测轨迹覆盖）
- division precision / recall（与 GT 血缘逐事件比对）

> **本地 `detection_precision` 已知缺陷**：当结果对象数少于 GT 对象数时会 > 1，
> 只作参考，不作结论（见 `docs/pipeline_code_map.md` §8）。

调用方式（`scripts/eval_pipeline.py`，不加 `--official`）：

```bash
python scripts/eval_pipeline.py \
    --h5 data/interim/Fluo-N3DH-CE_01.h5 \
    --dataset Fluo-N3DH-CE --seq 01 \
    --exp-id <本地区分用编号> \
    --ckpt experiments/B1_gnn_paper_ce01/artifacts/model/best.pt
```

它内部做的事（`eval_pipeline.py` 主流程）：

1. 建 `Experiment` 目录（八件套留痕，见 §2.4）；
2. 若给 `--ckpt`：算 md5 + 大小，写 `checkpoint_ref.json`（记录"这组数字由哪个模型产生"）；
3. 构造 GNN runner，跑 `run_pipeline`；
4. **逐帧流式**写 CTC 提交目录，同时 `diag.add_frame(t, gt, res)` 累积诊断；
5. `seg_measure` 算本地 SEG；组装 `metrics.json` 存盘。

### 2.3 第三层：误差归因工具（本项目最重要的诊断手段）

逐模块调参只能回答"这个改动有没有用"，回答不了"误差主要在哪一环丢的"。
两个工具把这个问题变成可量化的漏斗。

**(a) 检测层天花板 `scripts/detection_ceiling.py`（纯 h5 计算，不依赖 GNN）**

| 层级 | 含义 |
| --- | --- |
| U0 | GT 节点（marker）在检测集中出现的比例 |
| U1 | GT 移动边的**两端**都被检测到的比例 |
| U2 | 两端被检出且位移在候选半径 `R_max` 内 |
| U3 | 分裂边的父 + 两子都被检测到的比例 |

它回答"追踪开始之前就已经输掉了多少"。

**(b) 全链路误差漏斗 `scripts/pipeline_funnel.py`**

把每条**真实关联边**从 GT 追溯到最终轨迹，计算逐级留存率：

```
S0  GT 真实边（R_max 可达）
S1  ├─ 式(26) 候选集是否保留       ← 不可逆：丢掉就永远找不回
S2  ├─ 决策（OT / GNN）是否选中正确目标
S3  └─ 轨迹重建后是否仍属同一轨迹   ← 冲突消解、分裂判定
```

一眼看出瓶颈在候选生成、模型判定还是后处理。

> 三个隐性 bug（EDT 物理间距、分裂标签污染、标定配置≠部署配置）都是这两个工具逼出来的，
> 详见 `docs/reports/day3_summary.md`。

### 2.4 本地测试也强制留痕：实验八件套

本地每次运行都通过 `src/celltracker/experiment/runner.py::Experiment` 自动落盘：

```
experiments/<exp_id>/
  config.yaml      实验元信息（目的、参数、argv、seed、status）
  command.sh       复现命令（自动记录 sys.argv）
  env.txt          环境（平台、CPU、Python、GPU、pip freeze）
  git_commit.txt   HEAD commit + clean/dirty 标记
  metrics.json     本地指标
  logs/run.log     运行日志
  artifacts/       中间产物（pipeline/、submission/、gnn/、model/）
  figures/         图
  notes.md         人工结论文档
```

**结果只增不改**：若目录已有 `metrics.json`，`Experiment` 会直接抛
`FileExistsError`，要求换新编号（如 `-fix1`）。总表追加在 `experiments/INDEX.md`。

### 2.5 本地测试的边界（为什么不能当结论）

1. 本地指标与官方 TRA **不单调**（曾出现"阈值扫描本地变好、官方变差"）；
2. 本地噪点粒度粗，ΔTRA < 0.001 无法与噪声区分；
3. **单序列结果不得作为结论**（R9）：要写进论文的结论至少 seq01 + seq02，
   关键结论补 ≥2 个随机种子。

---

## 3. 送官方评测前的最后一道闸：格式校验

官方 `TRAMeasure` 会直接拒评非法轨迹（曾报 `track 19 is not consistent with the image data`
并中断整个评测）。所以本地必须先跑校验器
`src/celltracker/track/base.py::finalize_tracks`（E2）。

它强制规范化五件事：

1. **丢弃幽灵轨迹**：出现在 `tracks` 但从未在任何帧被赋值的 id；
2. **起止帧以实际赋值为准**，而不是靠推断；
3. **打断不连续段**：某 id 中间缺帧 → 拆成多条连续轨迹，后段用新 id；
4. **子轨迹起点 = 父轨迹终点 + 1**（分裂必须紧邻）；
5. **一个父轨迹最多 2 个子节点**，超出的子节点断开父关系。

> `finalize_tracks` 是**论文之外的工程补充**，论文里必须单独标注。
> 未跑本地格式校验就送官方评测，属于禁止事项。

---

## 4. CTC 官方测试（云端）

### 4.1 为什么必须在云端跑

官方 `EvaluationSoftware`（15.5 MB，含 Linux/Mac/Win 二进制 + 官方测试数据集 + PDF）
在本机 **WSL2（内核 6.18）直接段错误**——2020 年静态链接二进制与新内核不兼容，
`setarch -R`、gdb 诊断均无效，且本机无 docker。

改在**云服务器（内核 5.15）**运行全部正常。因此形成了固定分工：
**权威指标一律在云端算，本地只保留自写 SEG 与诊断量。**

### 4.2 调用入口

只要给链路脚本加 `--official`，它就会在跑完本地评测后自动调用云端评测：

| 入口脚本 | `--official` | 说明 |
| --- | --- | --- |
| `scripts/eval_pipeline.py` | 支持 | Phase B 主力：跑链路 + 本地指标 + 官方指标 |
| `scripts/run_baseline.py` | 支持 | 匈牙利/贪心基线 |
| `scripts/run_ot.py` | 支持 | 纯 OT 追踪 |
| `scripts/run_gnn.py infer` | 支持 | GNN 一条龙（build/train/infer） |
| `scripts/run_ablation_matrix.py` | 支持 | 消融矩阵：重建图 → 重训 → 官方评测 → 成表 |
| `scripts/run_sweep.py` | 支持 | 参数网格扫描 |

也可以单独对一份已有结果目录调用：

```bash
python scripts/cloud_eval.py \
    --res-dir experiments/<exp>/artifacts/submission/01_RES \
    --dataset Fluo-N3DH-CE --seq 01 \
    --out experiments/<exp>/metrics_official_01.json
```

### 4.3 云端流程逐步拆解（`scripts/cloud_eval.py`）

云端固定路径：

| 变量 | 路径 | 含义 |
| --- | --- | --- |
| `CLOUD_CTC` | `/root/autodl-tmp/ctc/raw` | 云端 CTC 原始数据根（GT 来源） |
| `CLOUD_TOOLS` | `/root/EvaluationSoftware/Linux` | 官方二进制目录 |
| `CLOUD_EVAL` | `/root/autodl-tmp/celltracker/eval` | 评测临时工作区 |

连接凭据从本地 `本地文档/ssh&key` 解析（`cloud_run.py::parse_cred`），
**每次交互式输入密码，不写入任何脚本或配置文件（E13）**。

执行步骤：

**Step 0 — 预检**

- 确认 `--res-dir` 存在；
- 生成唯一 tag：`<dataset>_<seq>_<YYYYmmdd-HHMMSS>`；
- 打印 dry-run JSON（host / tag / remote_dir / dataset / seq / res_dir）。

**Step 1 — 打包并上传结果**

1. `ssh mkdir -p <remote_dir>` 建云端临时目录；
2. 本地把结果目录 tar.gz 打包，**归档名为 `<seq>_RES`**（保证解压后目录名合法）；
3. `scp` 上传 `<remote_dir>/res.tar.gz`，打印体积（MB）。

**Step 2 — 云端组装目录并运行三个官方度量**

通过 `ssh bash -s` 执行一段脚本（`set -e` 失败即停）：

```bash
cd <remote_dir>
mkdir -p <remote_dir>/<dataset>
# 把官方 GT 软链进来，形成 <dataset>/<seq>_GT 与 <dataset>/<seq>_RES 的合法布局
ln -sfn /root/autodl-tmp/ctc/raw/<dataset>/<seq>_GT  <...>/<dataset>/<seq>_GT
rm -rf <...>/<dataset>/<seq>_RES
tar xzf res.tar.gz -C <...>/<dataset>
# 三个官方度量，参数：<dataset_dir> <seq> <num_digits>
for exe in SEGMeasure DETMeasure TRAMeasure; do
  echo "=== $exe ==="
  /root/EvaluationSoftware/Linux/$exe <...>/<dataset> <seq> 3
done
ls <...>/<dataset>/<seq>_RES/*.txt
```

> `num_digits` 默认 3，即 `mask000.tif` 的位数，必须与本地写出时一致。
> 评测把整份掩码上传解压，不清理会撑爆数据盘，所以默认评测后删除临时目录（E4）。

**Step 3 — 解析指标、回传日志、清理**

1. 用正则 `<name> measure:\s*([0-9.]+)` 从输出抓 `SEG` / `DET` / `TRA`；
2. 写 `metrics_official_<seq>.json`，字段含：dataset / seq / num_digits / res_dir /
   evaluator（`CTC official EvaluationSoftware (cloud)`）/ host / SEG / DET / TRA /
   raw_output / timestamp；
3. `scp` 回传三个官方原始日志到 `<exp>/official_logs/`：
   `SEG_log.txt`、`DET_log.txt`、`TRA_log.txt`——这是评测的**原始证据**；
4. 除非显式 `--keep-remote`，`rm -rf <remote_dir>` 清理云端临时目录。

### 4.4 官方指标含义

| 指标 | 衡量什么 | 备注 |
| --- | --- | --- |
| **DET** | 检测（每帧目标是否被检出） | GT 检测档恒为 1.0 |
| **SEG** | 分割（Jaccard，GT 目标被 >50% 覆盖才计分） | 本地 `seg_measure` 与其一致 |
| **TRA** | 追踪（基于 AOGM 的归一化分数） | **最终权威指标** |

### 4.5 AOGM 六项分解（误差归属）

官方 `TRA_log.txt` 记录了失分明细，本地用
`scripts/analyze_tra_log.py` 解析成 AOGM 六项（CTC 官方权重，Matula et al. 2015）：

```bash
python scripts/analyze_tra_log.py experiments/*/official_logs/TRA_log.txt
```

| 项 | 含义 | 权重 |
| --- | --- | --- |
| NS | Splitting Operations（多余分裂操作） | 5 |
| FN | False Negative Vertices（漏检节点） | 10 |
| FP | False Positive Vertices（假阳性节点） | 1 |
| ED | Redundant Edges To Be Deleted（多余边） | 1 |
| EA | Edges To Be Added（缺失边） | 1.5 |
| EC | Edges with Wrong Semantics（语义错误边） | 1 |

> AOGM 越低越好；TRA = 1 − AOGM/AOGM₀（AOGM₀ 为全空结果的相关值）。
> 例：真实检测档 A（top-k=3）AOGM = 25810（NS 2199 / FN 87 / FP 2456 / ED 29 / EA 7602 / EC 57），
> TRA = 0.905706。

### 4.6 云端长任务规程（E10–E12）

官方评测是分钟级，但 nnU-Net 推理/重训是小时级（>10 分钟一律云端后台，R13）：

```bash
# 启动前确认卡空闲
bash scripts/cloud_run.sh "nvidia-smi"
# 后台启动并记录 PID + 日志路径
bash scripts/cloud_run.sh "cd /root/CellTracker && nohup <cmd> > logs/<name>.log 2>&1 & echo \$!"
# 非阻塞轮询
bash scripts/cloud_run.sh "tail -n 20 logs/<name>.log"
```

三条硬要求：启动后确认进程真占显存（防"静默退回 CPU"）；结束前确认 checkpoint 大小非零；
任务被打断后先看 `git log` + `git status` + 实验目录 + 日志尾部，再决定续跑/重跑。
云端 kill 禁止自匹配的 `pkill -f`（会杀掉自己），必须 `pgrep -f 'pytho[n] ...'` 取精确 PID。

---

## 5. 命令速查

```bash
# ───── 本地：单测 ─────
pytest -q

# ───── 本地：主链路（GT 检测上界），只出本地指标 ─────
python scripts/run_pipeline.py --h5 data/interim/Fluo-N3DH-CE_01.h5 \
    --exp-id <id> --dump-graphs data/interim/graphs_<id>

# ───── 本地：GNN 训练（CPU，约 90 秒 / 60 epoch） ─────
python scripts/run_gnn.py train --graphs data/interim/graphs_<id> \
    --exp-id <id> --epochs 60

# ───── 本地 + 官方：跑链路并评测（论文口径） ─────
python scripts/eval_pipeline.py --h5 <h5> --gt-h5 <gt_h5> \
    --dataset Fluo-N3DH-CE --seq 01 --exp-id <id> \
    --ckpt experiments/<train>/artifacts/model/best.pt \
    --set tracklet.enabled=true --official

# ───── 消融矩阵（自动"重建图 → 重训 GNN → 官方评测"） ─────
python scripts/run_ablation_matrix.py --h5 <h5> --dataset Fluo-N3DH-CE --seq 01 \
    --ablations ot_cand,cand_topk --official

# ───── 单独对已有结果目录跑官方评测 ─────
python scripts/cloud_eval.py --res-dir experiments/<id>/artifacts/submission/01_RES \
    --dataset Fluo-N3DH-CE --seq 01 --out experiments/<id>/metrics_official_01.json

# ───── AOGM 六项分解 ─────
python scripts/analyze_tra_log.py experiments/*/official_logs/TRA_log.txt

# ───── 误差归属 ─────
python scripts/detection_ceiling.py --pred-h5 <pred.h5> --gt-h5 <gt.h5>
python scripts/pipeline_funnel.py --h5 <h5> --graphs <graphs> --ckpt <ckpt>

# ───── 本地 SEG 自检（对比官方文档基准） ─────
PYTHONPATH=src python -c "
from celltracker.eval.local_metrics import seg_measure
assert abs(seg_measure('tools/testing_dataset/01_GT/SEG','tools/testing_dataset/01_RES') - 0.232874) < 1e-6
print('local SEG self-check OK')"
```

---

## 6. 两个完整示例

### 6.1 GT 检测上界档（隔离追踪贡献）

```bash
# 1) 构图 + 训练（本地 CPU）
python scripts/run_pipeline.py --h5 data/interim/Fluo-N3DH-CE_01.h5 \
    --exp-id B1_graphs --dump-graphs data/interim/graphs_ce01
python scripts/run_gnn.py train --graphs data/interim/graphs_ce01 \
    --exp-id B1_gnn_paper_ce01 --epochs 60

# 2) 本地诊断 + 云端官方指标（一条命令）
python scripts/eval_pipeline.py --h5 data/interim/Fluo-N3DH-CE_01.h5 \
    --dataset Fluo-N3DH-CE --seq 01 --exp-id B1_eval_ce01_gnn \
    --ckpt experiments/B1_gnn_paper_ce01/artifacts/model/best.pt \
    --official
```

结果（官方）：DET 1.000000 / TRA 0.996513 / AOGM 954.5。

### 6.2 真实检测档（端到端）

```bash
# 1) nnU-Net 掩码 → 实例 → 预测检测 h5（带 gt_label）
python scripts/predict_to_h5.py --pred-dir <nnunet_masks> \
    --gt-h5 data/interim/Fluo-N3DH-CE_01.h5 \
    --out data/interim/Fluo-N3DH-CE_01_pred_v2.h5

# 2) 用预测检测重建图 + 同源重训 GNN（R11）
python scripts/run_pipeline.py --h5 data/interim/Fluo-N3DH-CE_01_pred_v2.h5 \
    --gt-h5 data/interim/Fluo-N3DH-CE_01.h5 \
    --exp-id C5_graphs --dump-graphs data/interim/graphs_01_pred_v2b
python scripts/run_gnn.py train --graphs data/interim/graphs_01_pred_v2b \
    --exp-id C5.0b_pred_v2_train --epochs 60

# 3) 全链路官方评测：--h5 是检测来源，--gt-h5 是评测真值
python scripts/eval_pipeline.py \
    --h5 data/interim/Fluo-N3DH-CE_01_pred_v2.h5 \
    --gt-h5 data/interim/Fluo-N3DH-CE_01.h5 \
    --dataset Fluo-N3DH-CE --seq 01 --exp-id C4A_official_gnn_topk3 \
    --ckpt experiments/C5.0b_pred_v2_train/artifacts/model/best.pt \
    --official
```

结果（官方）：DET 0.939833 / SEG 0.673196 / TRA 0.905706 / AOGM 25810。

### 6.3 三档检测来源的官方结果对照（seq01，195 帧，同一套 pipeline）

| 档位 | DET | SEG | TRA | AOGM | 说明 |
| --- | --- | --- | --- | --- | --- |
| GT 标记上界（B1） | 1.000000 | — | **0.996513** | 954.5 | 检测=真值，隔离追踪贡献 |
| Oracle 实例（C4O） | 0.991581 | 0.688720 | **0.984784** | 4165 | 掩码=nnU-Net，实例归属=完美（作弊上界） |
| 真实检测 A（C4A-o，top-k=3） | 0.939833 | 0.673196 | **0.905706** | 25810 | 论文口径，端到端 |
| 真实检测 B（C4B-o，top-k=5） | 0.939833 | 0.673196 | **0.906542** | 25581 | Δ 仅 1.2× 噪声地板，不改默认 |

---

## 7. 一张图看懂两条通道

```
                     ┌──────────────── 本地（方向判断） ────────────────┐
  数据 h5 / 预测 h5 → │ run_pipeline / eval_pipeline（不加 --official）  │
                     │   → 逐帧写 <seq>_RES（mask*.tif + res_track.txt）│
                     │   → finalize_tracks 格式校验（E2）               │
                     │   → 本地 SEG（与官方逐位一致）+ 追踪诊断量        │
                     │   → detection_ceiling / pipeline_funnel 误差归因 │
                     │   → Experiment 八件套 + INDEX.md（R6/E9）        │
                     └──────────────────────────┬──────────────────────┘
                                                │ 需要定论时加 --official
                                                ▼
                     ┌──────────────── 云端（权威定论） ────────────────┐
                     │ cloud_eval.py:                                  │
                     │  [1/3] tar <seq>_RES → scp 上传                 │
                     │  [2/3] 软链 <seq>_GT；跑官方 SEGMeasure/         │
                     │        DETMeasure/TRAMeasure <dataset> <seq> 3  │
                     │  [3/3] 解析 SEG/DET/TRA → metrics_official.json │
                     │        回传 *_log.txt；rm -rf 云端临时目录        │
                     │  → analyze_tra_log.py → AOGM 六项分解            │
                     └─────────────────────────────────────────────────┘
```

---

## 8. 口径红线（写论文/汇报时必须遵守）

1. **权威指标**：CTC 官方二进制，云端运行；本地指标只判方向（R4）。
2. **噪声地板 = 0.0007**：ΔTRA < 0.001 一律表述为"与噪声不可区分"，标注倍数于噪声地板（R10）。
3. **三档检测来源必须分别报告**，不得混用（§1.1）。
4. **双序列**：写进论文的结论至少 seq01 + seq02；关键结论补 ≥2 个随机种子（R9）。
5. **送评前必跑本地格式校验**，否则禁止送官方评测（E2）。
6. **训练/测试检测来源同分布**：预测 h5 必须带 `gt_label` 映射（R11）。
7. **论文之外的工程补充必须单独标注**：top-k 保底、`ε=0.1×median(C)` 自适应、
   `finalize_tracks` 校验器。
8. **评测真值 vs 检测来源分离**：报告里必须写清 `--h5`（检测来源）与 `--gt-h5`（评测真值）各是哪一档。

---

## 9. 相关文件索引

| 文件 | 作用 |
| --- | --- |
| `scripts/eval_pipeline.py` | Phase B 主力：跑链路 + 本地指标 + 可选官方指标 |
| `scripts/run_gnn.py` | GNN build / train / infer（infer 支持 `--official`） |
| `scripts/cloud_eval.py` | 云端官方 DET/SEG/TRA 评测封装（上传→评测→回传→清理） |
| `scripts/cloud_run.py` / `cloud_run.sh` | 云端执行命令 / 文件传输（密码登录） |
| `scripts/analyze_tra_log.py` | 官方 TRA 日志 → AOGM 六项分解 |
| `scripts/detection_ceiling.py` | 检测层天花板 U0–U3 |
| `scripts/pipeline_funnel.py` | 全链路误差漏斗 S0–S3 |
| `src/celltracker/eval/local_metrics.py` | 本地 SEG + 追踪诊断量 |
| `src/celltracker/eval/ctc_io.py` | CTC 提交读写（`ResultWriter` 流式） |
| `src/celltracker/track/base.py::finalize_tracks` | 送评前格式校验 |
| `src/celltracker/experiment/runner.py` | 实验八件套留痕 |
| `experiments/E0.3_eval_selfcheck/` | 评测自检（本地 SEG 与官方逐位匹配） |
| `experiments/C4A_official_gnn_topk3/` | 真实检测档官方评测范例（含官方日志） |
| `docs/pipeline_code_map.md` | 模块 ↔ 论文公式 ↔ 代码位置总图 |
