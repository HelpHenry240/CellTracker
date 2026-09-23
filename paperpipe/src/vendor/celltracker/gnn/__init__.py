"""裁剪版 `celltracker.gnn`（vendor）：只保留 paperpipe 复用的 `data` 子模块。

上游的 `celltracker/gnn/__init__.py` 会导入 `model.py`（三分类 + 残差连接的
边分类 GNN）与 `adapter.py`。那**不是** paperpipe 的口径：paperpipe 按原文式(29)(33)(34)
用二分类 sigmoid + 纯映射，实现在 `papertrack/gnn/model.py`。
为避免"被替换的实现"被意外导入，这里裁剪掉。
"""

__all__: list[str] = []
