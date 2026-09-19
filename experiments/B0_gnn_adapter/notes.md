# B0 GNN 接入主链路（适配层）

## 目的

Phase A 搭好了论文口径的 pipeline 与接口，但 GNN 一直走的是**旁路**
（`build_dataset` 落盘图数据集 → 独立脚本训练/推理）。B0 把 GNN 接回主链路：

```
上游 OT 阶段产出的耦合（内存） → 构图（内存） → GNN 边分类 → 轨迹重建
```

## 做了什么

1. **`Detections` 增加 `meta`**：`from_h5` 自动填入 `shape` 等序列级信息，
   这样下游构图不必再持有 h5 句柄。
2. **`gnn/infer.py::predict_from_couplings`**：直接在内存里用耦合构图并推理
   （复用 `build_pair_graph(coupling=...)` 与 `gnn.data.collate`），
   返回结构与原 `predict_pairs` 一致，可直接喂给 `reconstruct_tracks`。
3. **`gnn/adapter.py::make_gnn_runner`**：生成签名 `(dets, couplings, cfg) ->
   (TrackResult, graph_dir)` 的可调用对象，直接传给 `run_pipeline(gnn=...)`。
   - 图构建配置由 `graph_cfg_from_pipeline(cfg)` 从 pipeline 配置段映射而来
     （**单一配置源**，避免两套参数各调一处）；
   - 决策阈值（`tau_move/tau_div/max_children`）取自 `cfg.reconstruct`。
4. `load_model()` 抽出为公共函数（原 `predict_pairs` 内联的加载逻辑）。

## 端到端验证（真实数据，CE seq01 帧 150-165）

| 路径 | decision | 轨迹数 |
| --- | --- | --- |
| GNN（`run_pipeline(gnn=runner)`） | `"gnn"` | **510** |
| 消融 `--ablate gnn`（§1.6 OT 规则） | `"ot_rule"` | 634 |

两条路径都能跑通，且消融开关确实切换了决策阶段。
（16 帧的绝对数值只用于验证链路，不作性能结论——按 R4 留到 Phase B 用官方指标裁决。）

## 验收

| 检查项 | 结果 |
| --- | --- |
| GNN 可作为决策阶段接入 runner | ✅ `decision == "gnn"` |
| 关闭 GNN 时回退到 OT 规则 | ✅ `decision == "ot_rule"` |
| 图配置来自 pipeline（单一配置源） | ✅ 单测覆盖 |
| 预测/中间产物落盘 | ✅ `gnn_preds.npz` |
| CTC 格式合法 | ✅ 单测断言轨迹连续 |
| 全量测试 | ✅ 66 项通过 |

## 复现

```python
from celltracker.pipeline import run_pipeline
from celltracker.gnn.adapter import make_gnn_runner
runner = make_gnn_runner("experiments/<...>/best.pt", InferConfig())
run = run_pipeline("data/interim/Fluo-N3DH-CE_01.h5", cfg, gnn=runner)
```

## 下一步（B1）

用**论文口径**重新训练 GNN（式34 边级 CE + 式35 `L_OT-reg = Σ ŷ_e·C_e`，
不含任何自创融合），训练数据用按式(26) 筛的候选集（top-3 保底）。
训练完成后即可用 Phase A 的 runner 跑完整消融矩阵（B2）。
