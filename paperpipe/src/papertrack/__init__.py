"""`papertrack`：**严格按 ideas.pdf 重建**的显微细胞追踪 pipeline。

与原仓库 `celltracker` 的关系（重要）
------------------------------------
本包是"论文口径"的干净重建，不是 `celltracker` 的第二份拷贝：

* **符合原文公式/参数的模块 → 直接 import 复用**（避免两套代码漂移）：
  这些模块的**副本已放进 `paperpipe/src/vendor/celltracker/`**（见 `vendor/README.md`），
  由 `_paths.ensure_vendor_on_path()` 插到 `sys.path` 最前 ⇒ **paperpipe 自包含，
  不再依赖外层仓库的 `src/`**。
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

包内按**论文模块**分文件夹：

    representation/  §1.2 测度与帧内图（式1–7）
    coupling/        §1.3 相邻帧 OT（式8–14）
    temporal/        §1.4 时间展开图 + 多尺度正则（式15–19）
    longrange/       §1.5 运动先验（式20–22）+ 二层 tracklet OT
    reconstruction/  §1.6/§2.0.1 轨迹重建（式23–24、式33）+ 轨迹规范化 + CTC 导出
    graph/           §2.0.1 时间展开图（式25–29，含跨帧桥接边）
    gnn/             §2.0.1 消息传递与训练（式30–35）
    runtime/         编排（run_pipeline）与提交格式校验
    config.py        全参数（论文符号 + PAPER/CALIB/ENG 标记）

本包**只依赖 numpy/scipy/torch/tifffile/h5py**，不新增第三方依赖。
"""

from ._paths import ensure_vendor_on_path  # noqa: F401  (import 时即注入 vendor 路径)

__all__ = ["__version__", "run_pipeline", "export_ctc", "PipelineConfig", "load_config"]
__version__ = "0.1.0"


def __getattr__(name: str):
    """懒加载顶层快捷入口（避免 `import papertrack` 时把整条链路都拉进来）。"""
    if name in ("run_pipeline", "PipelineRun", "load_detections", "load_gt_parent"):
        from . import runtime

        return getattr(runtime, name)
    if name == "export_ctc":
        from .reconstruction import exporter

        return exporter.export_ctc
    if name in ("PipelineConfig", "load_config", "save_config", "override"):
        from . import config

        return getattr(config, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
