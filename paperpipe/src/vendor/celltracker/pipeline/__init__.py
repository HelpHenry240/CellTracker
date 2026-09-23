"""裁剪版 `celltracker.pipeline`（vendor）：只保留 paperpipe 复用的 `motion` 子模块。

上游这个包会导入整条旧 pipeline（`config` / `ot_stage` / `multiscale_stage` /
`tracklet_stage` / `runner`）。paperpipe 只需要其中的 `motion.py`（式20 的估速），
因此裁剪成空包，避免把旧实现整条拉进来。
"""

__all__: list[str] = []
