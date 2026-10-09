"""共享图数据接口与历史 GNN 对照接口；按需加载，避免循环导入。"""
from importlib import import_module

__all__ = ["EdgeGNN", "ModelConfig", "make_gnn_runner", "graph_cfg_from_pipeline"]


def __getattr__(name):
    modules = {"EdgeGNN": "model", "ModelConfig": "model",
               "make_gnn_runner": "adapter", "graph_cfg_from_pipeline": "adapter"}
    if name not in modules:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{modules[name]}"), name)
    globals()[name] = value
    return value
