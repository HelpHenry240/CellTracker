"""§1.2 帧内表示：经验测度与邻域图（`ideas.pdf` 式 1–7）。

* `measure.masses`           式(1)(3) 质量 `a_i = s_i/Σs_k`（复用 vendor 的归一化）
* `measure.point_features`   式(2) 点级特征 `f = [x, s]`
* `measure.knn_structure`    式(4)(5)(6)(7) 帧内 kNN 图、边权 `W`、距离 `D`
* `measure.build_cost`       式(8)(22) 特征代价 `C_feat`（含 `R_max` 门限）
* `measure.resolve_spacing`  R3：体素 → µm 的物理间距解析
"""

from .measure import (build_cost, knn_structure, masses, pairwise_distance,
                      point_features, resolve_spacing, sigma_from, sigma_s_from)

__all__ = ["masses", "point_features", "knn_structure", "build_cost",
           "resolve_spacing", "sigma_from", "sigma_s_from", "pairwise_distance"]
