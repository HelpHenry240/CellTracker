# A6–A8 统一 pipeline runner 与接口收口

> 按用户指示：本阶段只验证"能跑通 + 接口到位"；性能调优留到 Phase B。

## 做了什么

**统一 runner**（`pipeline/runner.py::run_pipeline`）按论文顺序串起七个阶段：

```
§1.2 测度/帧内图 → §1.3 相邻帧 OT(η/τ/ε) → §1.5 运动先验(两遍式)
  → §1.4 多尺度时间正则 → §2.0.1 时间展开图(式26 筛候选)
  → 决策(GNN 或 §1.6 OT 规则) → §1.6 第二层 tracklet OT → 评测
```

配套接口：

| 接口 | 说明 |
| --- | --- |
| `run_pipeline(h5, cfg, frames, artifacts_dir, gnn=None)` | 一次运行返回 `PipelineRun`（含各阶段中间产物） |
| `gnn=None` | 走 §1.6 OT 规则重建（消融对照）；传入可调用对象则走 GNN 路径 |
| `artifacts_dir` | 落盘 `couplings.npz` 与 `run_info.json`（各阶段元信息） |
| `--ablate fgw,multiscale,...` | 八类消融开关，统一入口 |
| `--set ot.eta=0.3 tracklet.enabled=true` | 点号路径覆盖任意配置字段（带字段校验） |
| `configs/pipeline_default.yaml` | 论文口径的默认配置，随实验记录入库 |
| `scripts/run_pipeline.py` | 薄 CLI，自动创建实验目录与留痕 |

## 端到端验证

```
python scripts/run_pipeline.py --h5 data/interim/Fluo-N3DH-CE_01.h5 \
    --exp-id A6_e2e_smoke8 --frames 150:158 \
    --config configs/pipeline_default.yaml \
    --set multiscale.enabled=true tracklet.enabled=true
```

输出（`run_info.json`）：

```json
{"n_frames": 9, "ablated": [], "n_couplings": 8,
 "multiscale": {"enabled": true, "ks": [2,3,5], "lambda_temp": 0.1, "n_pairs": 8},
 "decision": "ot_rule", "tracks_after_decision": 328,
 "tracklet": {"enabled": true, "n_tracklets": 328, "n_links": 0},
 "tracks_final": 328}
```

各阶段全部按预期执行并记录信息 ✅；中间产物已落盘 ✅。

## 过程中发现的两个问题

### 问题一（接口语义）：OT 规则重建复用了错误的阈值字段

`ot_rule_reconstruct` 最初用 `tracklet.theta_link` 作为分裂判据的质量阈值。
这不仅语义错位（tracklet 阶段的阈值是"第二层 OT 接受关联"的阈值），
而且与 A2 的结论冲突——**分裂判据与 ε 强耦合**，必须独立可调。

修复：在 `ReconstructConfig` 增加 `div_ratio` 字段（默认 0.2），
OT 规则路径专用；GNN 路径不读该字段（分裂由 GNN 概率决定）。

### 问题二（性能）：多尺度的回溯线搜索让耗时放大 5–10 倍

9 帧的端到端运行耗时 **74 秒**（约 8 s/帧）。主要开销在 §1.4：
每个 Gauss-Seidel 轮次最多尝试 5 次步长，每次尝试都要遍历所有帧对重解 Sinkhorn；
再加上跳帧直接耦合（k∈{2,3,5}）。按此推算，195 帧全量需 **25 分钟以上**。

这不影响本轮验收（接口正确），但 Phase B 必须处理，候选方案：

1. **缓存跳帧耦合**：`D_{t,t+k}` 只与帧对有关，跨轮次/跨 k 可复用（当前每轮重算）；
2. **减少线搜索尝试次数**：先记录"A4 实测线搜索几乎从不成功"，
   改为首步 + 单次半砍（2 次而非 5 次）；
3. **局部化**：多尺度项只在 k 步邻域内计算，避免全矩阵乘积。

## 验收（功能）

| 检查项 | 结果 |
| --- | --- |
| 七阶段按序执行 | ✅ `run_info.json` 逐阶段记录 |
| GNN 关闭时走 OT 规则 | ✅ `decision: "ot_rule"` |
| 消融开关生效 | ✅ 单测覆盖（fgw/unbalanced/motion 同时消融） |
| 中间产物落盘 | ✅ `couplings.npz`、`run_info.json` |
| 配置文件可加载/覆盖 | ✅ 单测覆盖（yaml 往返 + `--set` 点号覆盖） |
| 全量测试 | ✅ 63 项通过 |

## Phase A 完成情况

| 步骤 | 模块 | 状态 |
| --- | --- | --- |
| A1 | 配置层与消融开关 | ✅ |
| A2 | 相邻帧 OT（η/τ/ε/α′） | ✅ |
| A3 | 运动先验（两遍式） | ✅ |
| A4 | 多尺度时间正则 | ✅ |
| A5 | 第二层 tracklet OT | ✅ |
| A6–A8 | 统一 runner + 图/GNN/重建接口 | ✅（GNN 具体模型由 Phase B 接入） |

**Phase A 到此完成**：论文的七个模块全部接入主链路、各有独立配置与消融开关、
中间产物可落盘。当前唯一未接入的是"训练好的 GNN 权重"
（`run_pipeline(..., gnn=callable)` 已预留接口）。
