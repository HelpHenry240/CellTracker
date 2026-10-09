# vendor/ —— 从原仓库 `src/celltracker` 复制的、paperpipe 实际用到的模块

为什么复制：**让 `paperpipe/` 自包含**。这些模块已被核对为"与 `ideas.pdf` 原文口径一致"
（见 `paperpipe/FORMULA_MAP.md` §2 "直接复用的模块"），因此按用户要求把副本放进本目录，
`paperpipe` 不再依赖外层仓库的 `src/`。

复制方式：`cp src/celltracker/<路径>` → `paperpipe/src/vendor/celltracker/<路径>`，
共用修复同步到两处，并通过回归测试核对；包初始化裁剪未使用的导入。
上游改动不会自动同步；如需更新，重新执行复制并重跑 `paperpipe/tests`。

## 清单（共用模块与包初始化）

| vendor 路径 | 对应的论文口径 | 被 paperpipe 哪些模块使用 |
| --- | --- | --- |
| `celltracker/ot/sinkhorn.py` | 式(10)–(14) 熵正则（非）平衡 OT 求解器 | `coupling/pairwise.py`、`temporal/multiscale.py`、`longrange/tracklet.py` |
| `celltracker/ot/fgw.py` | 式(9) FGW 结构项与条件梯度求解 | `coupling/pairwise.py` |
| `celltracker/cost/features.py` | 式(1) 质量归一化、式(7) 距离原语 | `representation/measure.py` |
| `celltracker/track/base.py` | 数据容器 + CTC 规范化（`finalize_tracks`，split 口径） | 全流程（`Detections`/`TrackResult`/`paint_result`） |
| `celltracker/track/tracklets.py` | 历史片段工具 | 保留兼容；重建后的 tracklet 不调用该工具 |
| `celltracker/pipeline/motion.py` | 式(20) 用"上一轮硬关联"估速 | `longrange/motion.py` |
| `celltracker/data/ctc.py` | CTC 目录/`res_track.txt` 读写、实例表提取 | 多处 |
| `celltracker/eval/ctc_io.py` | CTC 结果写出（mask + res_track） | `reconstruction/exporter.py` |
| `celltracker/eval/local_metrics.py` | 本地 SEG（与官方逐位一致）+ 流式诊断 | `reconstruction/exporter.py` |
| `celltracker/gnn/data.py` | 图批处理 `collate`（格式中立） | `gnn/train.py` |
| `celltracker/experiment/runner.py` | 实验八件套留痕（E9） | `scripts/run_paper_pipeline.py` |

## 包初始化与同步约定

* `celltracker/gnn/__init__.py`：上游会 `from .model import EdgeGNN`（那是**三分类 + 残差**的
  对照实现，paperpipe 刻意不用）。这里裁剪成只导出 `data`，避免把被替换的实现带进来。
* `celltracker/pipeline/__init__.py`：上游会导入整条旧 pipeline（config/ot_stage/runner…），
  paperpipe 只复用其中的 `motion` 子模块，因此裁剪成空包 + 说明。

两者的裁剪都只是**去掉未被 paperpipe 使用的导入**，不改任何被复用模块的行为。

`detect/labels.py` 保存完整 GT 多数覆盖映射。FGW 完整目标、H5 字段读取和
诊断精确率的修复在外层与 vendor 同步；外层旧代价/高斯图属于历史基线，不作为新方法实现。
