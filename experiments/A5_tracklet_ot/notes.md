# A5 第二层 tracklet OT（§1.6）接入

> 按用户指示：本阶段只验证"能跑通 + 接口到位"，性能调优留到 Phase B。

## 做了什么

1. **阶段实现** `pipeline/tracklet_stage.py::link_tracklets`，严格按 §1.6 末段：
   - **超级节点**：tracklet 汇总特征（出现帧、首末位置、平均速度、时长）；
   - **测度**：tracklet 质量 ∝ 时长（越长越可信，在 OT 里话语权更大）；
   - **代价**：`‖末(A)−首(B)‖ + αv·‖匀速外推(A)−首(B)‖`，门限 `R_max×gap`
     （式20-22 的思想用在 tracklet 级）；
   - **第二层 OT**：对"各 tracklet 的末"与"各 tracklet 的首"求熵正则 **非平衡** OT；
   - **合并**：行内 argmax + 质量阈值 → 链式合并 → `finalize_tracks` 规范化。
2. **统一实现**：删除 `track/tracklets.py` 里旧的匈牙利版 `merge_tracklets`
   （被论文口径的第二层 OT 取代），该文件只保留 `tracklet_stats` 基元；
   `run_ot.py --merge-tracklets` 已改为调用新阶段。

## 接口验证（关键验收点）

```
第一层 OT  →  610 条轨迹（CE seq01 帧 120-195，75 帧稠密段）
第二层 OT  →  连接 5，链 5，轨迹 610 → 605，拆断 0，CTC 格式校验通过
```

## 过程中发现的两个设计问题

### 问题一：第二层必须是**非平衡** OT（否则不可能的边会被强行匹配）

最初把"不可行的配对"设为一个大常数 `1e9`。结果：**平衡 OT 的硬边缘约束
为了凑够质量，把这些不可能边也算进解里** —— 无后继的 tracklet 行 argmax
指向了无关轨迹（实测：行 4/5/6 的 argmax 全指向轨迹 1）。

修复：不可行配对置 **`np.inf`**；第二层改用**非平衡 OT**（式14 的 KL 松弛），
语义上也更对——tracklet 找不到后继就应当让质量"流失"（对应轨迹终止）。

### 问题二：CTC 格式约束使"跨空洞合并"自动失效（重要）

不加限制时，第二层找到 **19 个连接却只有 3 个真正生效**：
`finalize_tracks` 把 16 条合并后的轨迹又拆了回去，因为**CTC 要求轨迹在
起止帧之间每帧都出现**，而跨帧空洞（gap>1）的合并会留下空帧。

修复：默认**只接受 gap==1 的合并**（前后紧邻，天然合法），
并新增 `TrackletConfig.allow_gap_filling` 开关。限制后：连接 5、拆断 **0**。

**这意味着**：论文 §1.6 里"tracklet 串联以处理遮挡"的收益，
必须先**在空洞帧里补出检测/掩码**才能兑现——这属于 Phase C
（有了 nnU-Net 的真实掩码才能插值），在 GT 标记点设定下无法凭空补。

## 验收（功能）

| 检查项 | 结果 |
| --- | --- |
| 阶段能跑通（稠密段 75 帧） | ✅ 610 → 605 条轨迹 |
| 精度：CTC 格式合法 | ✅ 拆断 0、不连续轨迹 0、父>2 子 0 |
| `enabled=False` 恒等映射 | ✅ 单测覆盖 |
| 分裂子轨迹默认不被并回 | ✅ 单测覆盖 |
| gap>1 默认拒绝（格式约束） | ✅ 单测覆盖（含"开启后仍会被拆断"的说明） |
| 全量测试 | ✅ 64 项通过 |

## 复现

```bash
pytest tests/test_tracklet_stage.py -q     # 6 项
python scripts/run_ot.py --h5 data/interim/Fluo-N3DH-CE_01.h5 \
    --dataset Fluo-N3DH-CE --seq 01 --exp-id A5_tracklet \
    --eps 1.0 --no-eps-rel --div-ratio 0.3 --merge-tracklets
```
