# 架构与源码职责

`src/` 是唯一源码根目录；两个命名空间共同安装，保留 `papertrack` 名称以兼容已有模型。共享求解器只保留一份。

```mermaid
flowchart LR
    A[原图及冻结NN前景] --> B[实例H5]
    B --> C[冻结encoder侧车]
    C --> D[测度与帧内结构]
    D --> E[粗追踪运动及相邻OT/FGW]
    E --> F[多尺度精炼]
    F --> G[多帧图]
    G --> H[GNN边判定]
    H --> I[轨迹重建及tracklet]
    I --> J[流式CTC导出与校验]
    J --> K[官方DET/SEG/TRA]
```

| 代码位置 | 职责 |
| --- | --- |
| `src/papertrack/representation` | 物理测度、代价、帧内结构 |
| `src/papertrack/coupling` | 相邻/跨帧耦合编排 |
| `src/papertrack/temporal` | 多尺度完整目标及回溯 |
| `src/papertrack/longrange` | 运动与二层tracklet |
| `src/papertrack/graph` | 命名节点/边特征、候选、监督与上下文 |
| `src/papertrack/gnn` | 论文二分类 BCE/OT 损失、训练、推理 |
| `src/papertrack/reconstruction` | 生死、分裂、冲突、空洞与导出 |
| `src/papertrack/runtime` | 契约、标定、消融、主流程、官方评测 |
| `src/celltracker/{data,detect,cost,ot,track,eval,experiment}` | 共享数据/实例化、求解器、容器、IO、留痕 |
| `src/celltracker/gnn/data.py`、`pipeline/motion.py` | 共享批处理与运动估计 |
| `src/celltracker/{pipeline,gnn,graph}`中的历史控制实现、`track/ot_tracker.py` | 早期对照与回归用途；当前 CLI 不调用旧编排器 |

`celltracker.gnn` 与 `celltracker.pipeline` 的顶层历史接口按需加载，避免导入公共数据模块时拉入整条旧链路。所有共享计算模块保持原实现；没有 vendor 路径覆盖。

H5记录检测来源、spacing、实例label及可选GT重叠映射；encoder侧车记录逐帧身份和模型指纹。图、缓存、GNN权重均校验上游契约。`--h5` 是检测来源，`--gt-h5` 只提供监督、标定与评测真值；GT辅助Oracle必须单独标注。

论文逐式对应与工程补充见 [公式映射](formula_map.md)。
