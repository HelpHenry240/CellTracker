"""把原仓库的 `src/` 放到 `sys.path`，以便复用符合论文口径的既有实现。

本包故意不复制 `celltracker` 的代码：能直接调用的就 import，
只有与原文不符的模块才在本包内重写（见 `FORMULA_MAP.md`）。
"""

from __future__ import annotations

import sys
from pathlib import Path


def repo_root() -> Path:
    """仓库根目录（`.../CellTracker`）。"""
    return Path(__file__).resolve().parents[3]


def ensure_repo_src_on_path() -> Path:
    """确保 `import celltracker` 可用；返回仓库根。"""
    root = repo_root()
    src = root / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return root


ensure_repo_src_on_path()
