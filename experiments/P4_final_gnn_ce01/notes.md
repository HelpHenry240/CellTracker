# P4 端到端（严格论文口径 paperpipe + 云端 nnU-Net）：CE 双序列官方指标

日期：2026-09-23 ~ 09-24 · 云端：备用2（RTX 4090，`connect.westb.seetacloud.com:15927`）

## 一句话

**完全按 `ideas.pdf` 重建的 pipeline（`paperpipe/`）跑通端到端**：云端 nnU-Net
推理 → 实例拆分 → paperpipe（式1–35）→ CTC 提交 → 官方 DET/SEG/TRA。
**双序列 TRA：seq01 0.898249 / seq02 0.899119**（seq02 是**留出序列**：GNN 只在 seq01 上训练）。

## 官方指标（CTC 官方二进制，云端）

| 序列 | SEG | DET | **TRA** | 轨迹数 | 空洞帧（补画） | 格式校验 |
| --- | --- | --- | --- | --- | --- | --- |
| 01 | 0.673196 | 0.938833 | **0.898249** | 6408 | 249 | ok |
| 02（留出） | 0.690867 | 0.938995 | **0.899119** | 7796 | 310 | ok |

对照（同一份检测 h5、同一评测协议）：

| 配置 | seq01 TRA | seq02 TRA | 备注 |
| --- | --- | --- | --- |
| §1.6 **OT 规则**路径（无 GNN，式23/24） | 0.871725 | — | 10715 轨迹；ε 调软后 argmax 失去区分度 |
| §2.0.1 **GNN**（τ=0.5，修 tracklet 前） | 0.897222 | — | 5156 轨迹、3487 假空洞 |
| §2.0.1 GNN（τ=0.20） | 0.898192 | — | 6091 轨迹 |
| §2.0.1 GNN（**τ=0.35**，最终） | **0.898249** | **0.899119** | 3 个阈值差 ≈0.001（1.5× 噪声地板，R10：与噪声不可区分） |
| 原项目（真实检测档，in-sequence 训练，含工程补充） | 0.905706 | 0.909340 | 见 `experiments/C4A*`；+再切 k=1.6 为 0.9269/0.9263 |
| GT 标记上界 | 0.996513 | 0.996369 | 隔离追踪贡献 |

## 关键发现（本轮跑出来的实证）

1. **云端 nnU-Net 推理可复现**：新实例重跑 385 帧（29 min，RTX 4090，TTA 开，
   `checkpoint_best.pth` md5 `fe98e73ed9a6f8b22c22562eae44f3c5`），实例拆分报告与 09-20 那次
   **逐位一致**（seq01 mean_instances 122.93333333333334 / recall 0.9962325684660498；
   seq02 129.1421052631579 / 0.9900232250054288）。
2. **ε 的两种失败模式互相牵制**（候选覆盖率扫描 `paperpipe/calib/cand_coverage_ce01*.json`）：
   ε=0.1×median(C)（µm² 量纲）→ 计划退化成硬分配，**真实分裂"双子都在候选集"只有 0.308**
   （GNN 无法挽回，R5 的不可逆损失）；ε=1.0 → 该比例升到 0.808（η=0.3+topk 时 0.857），
   但**式(23) 的 argmax 失去区分度** → OT 规则路径碎片化（10715 轨迹 / TRA 0.8717）。
   ⇒ GNN 路径（论文主路径）用 ε=1.0；OT 规则路径不是当前 ε 下的可用配置。
3. **FGW 结构项压低分裂候选覆盖**：η=0.3 时"双子都在候选集"0.929 → 0.643；
   top-k 保底可回到 0.857。原文允许 η=0（纯特征 W-type OT），本轮按论文口径保留 η=0.3。
4. **跨空洞串联必须限定"中间帧节点缺失"**：二层 tracklet 原本允许 gap≤3 任意串联，
   真实数据上造出 **3487 个假空洞**；按 §2.0.1 前置条件限定后降到 249，
   FP 5938 → 2704，官方 TRA +0.001。
5. **多尺度精炼局部化**：原实现每次线搜索对全部 194 对重算（O(T²)），194 对 40+ min；
   改为只算被更新帧对及其所属时间窗（数学等价）后 50 s（~40×）。
6. **GNN 跨序列几乎不退化**：seq01 训练 → seq02 评测，边分类 P 0.937 / R 0.826 /
   **F1 0.878**（seq01 自身留出块 F1 0.893），与双序列 TRA 一致（0.8982 vs 0.8991）。
   这是 R9 / 决策 0003 要求的跨序列证据。
7. **η_death/η_birth 在本实现里只作诊断**（`death_veto=false`）：标定不准的 η 会不可逆地
   删掉正确边（R5）；论文的字面读法（阈值否决已接受关联）保留为开关。

## 配置口径（每个数都有标定来源）

`paperpipe/configs/paper_e2e_ce.yaml`：`mass_mode=volume`（式1；预测实例体积真实可变）、
`r_max=4.3 µm`（相邻帧最近邻位移 p99.9）、`eps_rel=1.0`（候选覆盖率标定）、
`eta=0.3`（论文 FGW）、`tau_a=tau_b=1.0`（式14 非平衡）、`theta_gamma_frac=0.01`
（真实后继边 Γ/a_i 的 p5）、`div_ratio=0.05`、`vol_tol=1.35`（真实分裂体积守恒偏差 p90）、
`cand_topk=0`（原文口径）、`bridge_scope=missing_only`、`tau_edge=0.35`（官方 TRA 选出）、
`hole_policy=fill`。标定脚本：`calibrate_params.py`、`check_candidate_coverage.py`。

## 复现（按顺序）

1. 云端推理（脚本已固化）：`bash scripts/cloud_run.sh "cd /root/autodl-tmp/nnunet && bash cloud_infer_eval.sh > logs/infer_eval.log 2>&1"`（后台 nohup 启动，见 `paperpipe/scripts/cloud_nnunet_infer.sh`）。
2. 取回预测：`python scripts/cloud_run.py --download /root/autodl-tmp/nnunet/preds_eval.tgz /tmp/preds_eval.tgz` → 解包到 `data/interim/preds_nnunet_e2e/`。
3. 实例拆分：`python scripts/predict_to_h5.py --pred-dir data/interim/preds_nnunet_e2e --img-root data/raw/Fluo-N3DH-CE --seq 01 --out data/interim/Fluo-N3DH-CE_01_e2e.h5 --gt-root data/raw/Fluo-N3DH-CE --min-distance 3 --min-volume 300 --h-frac 0.1 --gaussian-sigma 0.0`。
4. 建图 + 训练：`run_paper_pipeline.py ... --dump-graphs data/interim/pg_e2e_01b` → `train_paper_gnn.py --graphs data/interim/pg_e2e_01b --out paperpipe/runs/pg_gnn_01 --epochs 60`。
5. 端到端：`run_paper_pipeline.py --config paperpipe/configs/paper_e2e_ce.yaml --set reconstruct.tau_edge=0.35 --ckpt paperpipe/runs/pg_gnn_01/best.pt`。
6. 官方指标：`CT_CLOUD_CONN=15927 python scripts/cloud_eval.py --res-dir <...>_RES --dataset Fluo-N3DH-CE --seq 01 --out experiments/P4_final_gnn_ce01/metrics_official_01.json`。

## 下一步（按价值排序）

1. **边召回是唯一瓶颈**（R 0.83 → 每条长轨迹平均断约 5 次）：类别加权 BCE / 更长训练 /
   `lambda_ot=0` 三个方向，用官方 TRA 判据逐个试（R4）。
2. **`f_i` 换成 nnU-Net encoder 特征**（论文口径，接口已留 `node.f_source=encoder_npz`）。
3. **`cand_topk=3` 消融**：候选覆盖 +0.21（0.643→0.857），但属工程补充，须与原文口径并列。
4. 前端"超大实例再切"（原项目 k=1.6，官方 +0.0212）：论文之外的工程补充，与主表并列报告。
