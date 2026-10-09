"""端到端编排与提交校验。

* `pipeline.run_pipeline` 按论文顺序串联 §1.2→§2.0.1，返回 `PipelineRun`
* `pipeline.load_detections` 读检测 h5（含补读 `intensity_std` 的修补）
* `validate.validate_ctc_dir` 送官方评测前的 CTC 格式校验（E2）
"""

from .pipeline import (PipelineRun, load_detections, load_gt_parent, run_pipeline)

__all__ = ["PipelineRun", "run_pipeline", "load_detections", "load_gt_parent"]
