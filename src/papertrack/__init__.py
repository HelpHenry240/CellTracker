"""按 ideas.pdf 组织的追踪实现。

representation / coupling / temporal / longrange 对应论文 §1.2–1.5；
graph / gnn / reconstruction 对应 §1.6 与 §2.0.1；runtime 负责编排。
数据容器、OT 求解器和评测 IO 复用同一 src 下的 celltracker 公共模块。
公式与实现映射见 docs/architecture/formula_map.md。无需修改全局导入路径。
"""

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
