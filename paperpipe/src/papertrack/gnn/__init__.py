"""§2.0.1 时间展开图上的边分类 GNN（`ideas.pdf` 式 30–35）。

* `model.EdgeGNN`   式(30)(31)(32) 边–节点交替消息传递 + 式(33) 的 sigmoid 边分类头
* `train.train`     式(34) 边级 BCE + 式(35) OT 一致性正则；`GraphDataset` 按时间块切分
* `infer`           载入权重 → 产出 `{t: 决策}`，供 `reconstruction.rules` 重建轨迹
"""

from .model import EdgeGNN, ModelConfig

__all__ = ["EdgeGNN", "ModelConfig"]
