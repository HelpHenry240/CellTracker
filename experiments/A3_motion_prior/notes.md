# A3 运动先验（式 20-22）接入

> 按用户指示：本阶段**只做"能跑通 + 接口到位"的验证**，性能消融统一留到 Phase B
> （GNN 训练完成后一起做，避免在链路定型前反复测评）。

## 做了什么

1. **两遍式实现**（`pipeline/motion.py`）——严格按原文口径：
   - 原文："假设已经通过上一轮追踪在帧 t−1 与 t 之间建立了硬关联 p(i)"
     → 速度必须来自**上一轮追踪**，不是凭空产生；
   - 第 1 遍：α′=0 跑追踪，得到硬关联；
   - 式(20)(21)：由关联估速 `v = x_t − x_{t−1}` 并外推 `x̂ = x + v`；
   - 第 2 遍：带上 α′ 重跑（式22 的第三项）。

2. **接口**：
   - `estimate_velocity(dets, result) → (velocity, valid)`：`valid` 区分"静止"与"无前驱"；
   - `attach_velocity(dets, velocity, valid)`：把速度写回检测表；
   - `velocity_feature_is_live(dets)`：验收断言（见下）；
   - `run_two_pass(dets, cfg, alpha_pred, runner=None)`：两遍式驱动，runner 可注入便于测试；
   - `OTTrackConfig` 新增 `eps_rel`；`run_tracking_ot(..., pred_from_detections=True)`
     支持"用预置速度"而非在线估速。

3. **上下游连通**：速度写回 `Detections.frames[t]["velocity"]` 后，
   `graph/build.py` 的边特征（第 6 列 `d_pred`，运动先验残差）自动变为非零——
   这条路径此前一直是**死特征**（恒为 0），现在打通。

## 验收（功能）

| 检查项 | 结果 |
| --- | --- |
| 两遍式能跑通（CE seq01 全量） | ✅ `passes=2, pass1_tracks=700, pass2_tracks=678` |
| 速度特征非零（AGENTS.md 的验收点） | ✅ `velocity_live=True`，23,102 个检测有前驱 |
| 单测 | ✅ 6 项（速度=真值、valid 掩码、死特征检查、两遍式元信息、α′=0 剪枝等价单遍、运动先验偏向预测位置） |
| 全量测试 | ✅ 53 项通过 |

## 待 Phase B 回答的问题（不做结论）

本地诊断量显示 α′=1 相比 α′=0：ID switch 363→421、碎片化 338→368（轨迹数 700→678）。
按 R4（本地指标不得用于定论）这只作为**待验证假设**：

- 假设 H-α′：两遍式的速度来自**第 1 遍的关联**，第 1 遍本身的 ID 跳变会污染速度估计，
  从而放大误差；α′ 过强时"惯性"反而对抗真实的分裂/转向。
- Phase B 的验证方式：在 GNN 概率空间里做 α′ ∈ {0, 0.3, 1, 3} 的官方指标扫描，
  并用 AOGM 六项分解看误差归属。

## 复现

```bash
python scripts/run_ot.py --h5 data/interim/Fluo-N3DH-CE_01.h5 \
    --dataset Fluo-N3DH-CE --seq 01 --exp-id A3_CE01_ap1 \
    --eps 1.0 --no-eps-rel --eta 0 --div-ratio 0.3 --alpha-pred 1.0
```
