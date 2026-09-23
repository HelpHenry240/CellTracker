"""让 `paperpipe` 自包含：把 vendored `celltracker` 放到 `sys.path` 最前。

背景
----
`papertrack` 复用了原仓库若干"已核对与 `ideas.pdf` 口径一致"的模块
（OT 求解器、FGW、运动先验估速、CTC I/O、本地诊断、图批处理、实验留痕）。
为了让 `paperpipe/` **不依赖外层仓库**即可运行，这些模块的副本放在
`paperpipe/src/vendor/celltracker/`（清单见 `vendor/README.md`），
并在本模块被导入时插到 `sys.path` 最前面 —— 于是 `import celltracker` 解析到**副本**。

外层仓库的 `src/` **不会**被加入 `sys.path`（这是自包含的关键）。
若将来要在同一进程里对照两套实现，请显式 `sys.path.insert` 并注意顺序。
"""

from __future__ import annotations

import sys
from pathlib import Path


def paperpipe_src() -> Path:
    """`paperpipe/src`。"""
    return Path(__file__).resolve().parents[1]


def vendor_root() -> Path:
    """`paperpipe/src/vendor`（内含 `celltracker/` 副本）。"""
    return paperpipe_src() / "vendor"


def repo_root() -> Path:
    """外层仓库根目录（`.../CellTracker`）；仅用于定位数据/实验目录，不作为代码依赖。"""
    return Path(__file__).resolve().parents[3]


def ensure_vendor_on_path() -> Path:
    """把 vendor 目录插到 `sys.path` 最前，返回该目录。"""
    v = vendor_root()
    s = str(v)
    if s in sys.path:
        sys.path.remove(s)
    sys.path.insert(0, s)
    return v


def celltracker_source() -> str:
    """当前 `celltracker` 实际来自哪里（自检/报告用）。"""
    try:
        import celltracker

        return str(getattr(celltracker, "__file__", "?"))
    except Exception as exc:  # noqa: BLE001
        return f"(import failed: {exc})"


ensure_vendor_on_path()
