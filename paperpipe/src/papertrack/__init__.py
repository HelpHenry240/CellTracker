"""`papertrack`：**严格按 ideas.pdf 重建**的显微细胞追踪 pipeline。

与原仓库 `celltracker` 的关系（重要）
------------------------------------
本包是"论文口径"的干净重建，不是 `celltracker` 的第二份拷贝：

* **符合原文公式/参数的模块 → 直接 import 复用**（避免两套代码漂移）：
  - 式(10)(11)(12)(13)(14) 的熵正则（非）平衡 OT 求解器
    → `celltracker.ot.sinkhorn.sinkhorn_log`
  - 式(9) 的 FGW 结构项与条件梯度求解
    → `celltracker.ot.fgw.{structural_term, structural_grad, fused_gw}`
  - 数据容器与 CTC 合法性规范化
    → `celltracker.track.base.{Detections, TrackResult, finalize_tracks, paint_result}`
  - CTC 结果写出 / 读回 → `celltracker.eval.ctc_io`
  - 本地诊断指标 → `celltracker.eval.local_metrics`
  - 图批处理（与图 npz 格式无关的通用拼接）→ `celltracker.gnn.data.{PairDataset, collate}`
  - 实例拆分前端（nnU-Net 掩码 → 实例）→ `celltracker.detect.instances`
  - 实验八件套留痕 → `celltracker.experiment.Experiment`

* **与原文不符的部分 → 本包重新实现**，每处都在模块 docstring 里注明
  "原实现差在哪、原文怎么写"。清单见 `FORMULA_MAP.md`。

本包**只依赖 numpy/scipy/torch/tifffile/h5py**，不新增第三方依赖。
"""

from ._paths import ensure_repo_src_on_path  # noqa: F401  (import 时即注入路径)

__all__ = ["__version__"]
__version__ = "0.1.0"
