"""历史 pipeline 对照接口；当前主链路位于 papertrack.runtime。

公共运动估计仍由本包提供。历史接口按需加载，防止导入公共模块时
同时初始化旧模型与旧编排器。原有导出名称继续供回归测试使用。
"""
from importlib import import_module

_EXPORTS = {
    "ABLATIONS": "config", "PipelineConfig": "config", "apply_ablation": "config",
    "load_config": "config", "save_config": "config",
    "compute_pairwise_plan": "ot_stage", "CouplingArtifacts": "ot_stage",
    "MultiscaleResult": "multiscale_stage", "refine_couplings": "multiscale_stage",
    "TrackletResult": "tracklet_stage", "link_tracklets": "tracklet_stage",
    "PipelineRun": "runner", "run_pipeline": "runner", "ot_rule_reconstruct": "runner",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    globals()[name] = value
    return value
