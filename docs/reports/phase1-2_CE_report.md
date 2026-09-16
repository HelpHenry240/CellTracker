# Phase 1–2 报告：CTC 3D（Fluo-N3DH-CE seq 01）基线与最优传输

日期：2026-09-15/16 · 数据：`Fluo-N3DH-CE` seq 01（195 个有标注帧，平均 122 个目标/帧，
GT 720 条轨迹 / 358 次二分裂）· 评测：CTC 官方二进制（云端，权威）

## 1. 口径说明（本阶段最重要的两条经验）

1. **CTC 的追踪金标准（`_GT/TRA`）是"标记点"而不是细胞分割**。
   实测：CHO 某帧 marker 5171 体素 vs 银标准全分割 117035 体素，marker 100% 被 ST 覆盖。
   因此"把 marker 重新着色当结果"能拿到 **DET = 1.0**，但 **SEG 无意义（≈0）**——
   本阶段所有实验都只报告 DET/TRA；SEG 留到 Phase 5 用真实分割报告。
2. **CE 的 marker 是等体积小块**：实测 358 次分裂中，两个子 marker 体积之和 / 父 marker
   体积 = **恰好 2.00（p10–p90 全为 2.00）**。所以 ideas.pdf §1.6 里
   "s_j1 + s_j2 ≈ s_i" 的体积守恒判据在 CTC marker 上**不成立**，必须改用
   **行内质量分配**判据（一行中出现两个显著传输目标）。
   这条经验直接修正了原计划的 P2 分裂判定实现。

补充：Fluo-N3DH-CHO 的 TRA 标记**全部位于 z=0**（该数据集标注实质是 2D），
不适合作为 3D 方法验证集；CE 才是真正的 3D 难点集。

## 2. 结果对比（官方指标）

| 方法 | 预测轨迹 | ID switch | 碎片化 | DET | TRA |
| --- | --- | --- | --- | --- | --- |
| 匈牙利（`max_dist=30`，含分裂判定） | 371 | 459 | 107 | **1.000000** | **0.995594** |
| 纯 OT（`ε=1.0, θ_Γ=0.2, div=0.25`） | 757 | 345 | 377 | 1.000000 | 0.994361 |
| OT（`θ_Γ=0.05, div=0.3`） | 693 | 367 | 335 | 1.000000 | 0.994604 |
| OT（`ε=0.3, div=0.35, sum≥0.9`） | 570 | 432 | 278 | 1.000000 | 0.994617 |

图：`figures/CE01_comparison.png`（TRA/DET、ID switch、碎片化三栏对比）
表：`experiments/CE01_summary.md`

## 3. 关键结论

1. 检测层面（用 GT 检测）三种方法 **FN=FP=0、DET=1.0**，说明差异全部来自**关联策略**，
   这正是"分割无关上界实验"要达到的效果。
2. **匈牙利基线在 CE 上仍领先 OT（0.9956 vs 0.9946）**，差距约 0.001（≈300 AOGM 单位）。
   原因在误差结构：
   - 匈牙利：轨迹少（371 vs GT 720）→ 把分裂前后的细胞"一路连下去"，
     漏掉分裂边（每条 EA 罚 1.5），但**身份连贯**、碎片少（107）。
   - 我们的 OT：分裂判定更积极 → 轨迹数接近 GT（693–757），
     但**碎片化严重（335–377）**，多出来的断裂边（EA）抵消了分裂边的收益。
3. 参数敏感性（本地快速指标，`experiments/E2.5_CE01_theta_div_sweep/`）：
   `θ_Γ ∈ [0.05, 0.2]` 几乎无影响；`div_ratio` 与 `ε` 是主要杠杆，但只改变
   "碎片 vs ID switch"的**权衡方向**，TRA 只在 0.9944–0.9946 间移动。
4. 因此**单纯调 OT 超参不足以超越基线**，必须引入计划中的两类机制：
   长程/多尺度一致性（P3，抑制碎片化）与动态图 GNN（P4，学习"何时该连、何时该分"）。

## 4. 已修复的工程缺陷（都留了回归测试）

| 问题 | 现象 | 修复 |
| --- | --- | --- |
| 匈牙利分裂分支 id 分配错误 | 产生"幽灵轨迹"（记录存在但掩码中不存在）→ 官方 TRAMeasure 直接报错 `track 19 is not consistent with the image data` | `track/base.py` 统一用 `child = next_id`；新增 `tests/test_tracking_consistency.py` |
| 3D 全量读入内存 | 195 帧卷一次性载入（≈4.9GB）+ int64 拷贝 → OOM，进程被系统杀掉 | 改为**逐帧流式**（读→画→写→累积诊断），峰值内存 340MB；新增 `ResultWriter`、`StreamingDiagnostics` |
| 诊断函数整卷 int64 拷贝 | 12.7M 体素 ×8 字节 ×2 | 先做布尔裁剪再转类型 |

## 5. 下一步（按优先级）

1. **P3 长程一致性**：多尺度时间正则 + tracklet 二层 OT，目标是**把碎片化从 ~335 降到 <150**。
2. **P4 动态图 GNN**：以 OT 候选边为输入、学习边级 0/1，直接针对"误连/漏分裂"两类错误。
3. **运动先验**（`α'`）在 CE 上的消融（已实现，尚未在 CE 上测）。
4. Phase 5：nnU-Net 3D 分割前端 + §2.0.1 的鲁棒性论断（含 `_ERR_SEG` 官方压力测试）。

## 6. 复现

```bash
# 数据（本地）：下载并转内部表示
bash scripts/download_data.sh Fluo-N3DH-CE
PYTHONPATH=src python -m celltracker.data.build_dataset \
    --dataset data/raw/Fluo-N3DH-CE --seq 01 --out data/interim/Fluo-N3DH-CE_01.h5

# 基线 / OT（含云端官方指标）
python scripts/run_baseline.py --h5 data/interim/Fluo-N3DH-CE_01.h5 \
    --dataset Fluo-N3DH-CE --seq 01 --exp-id E1.3_CE01_hungarian --official
python scripts/run_ot.py --h5 data/interim/Fluo-N3DH-CE_01.h5 \
    --dataset Fluo-N3DH-CE --seq 01 --exp-id E2.5c_CE01_ot_eps0.3 \
    --eps 0.3 --theta-gamma 0.05 --div-ratio 0.35 --div-sum-min 0.9 --official

# 汇总对比
python scripts/make_comparison.py --dataset Fluo-N3DH-CE --seq 01 \
    --out-md experiments/CE01_summary.md --out-fig figures/CE01_comparison
```
