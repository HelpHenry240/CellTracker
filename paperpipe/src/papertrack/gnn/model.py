"""时间图的节点–边交替消息传递（ideas.pdf 式30–33）。

式(30)：e_next=φ_e(e,h_source,h_target)；式(31)：m_v=Σ_{u→v}e_next。
式(32)：h_next=φ_h(h,m)；式(33)：ŷ=σ(ψ(e_final))。
默认使用纯映射和二分类边概率。残差连接是可选工程对照；分裂由轨迹重建
阶段的多个显著后继和体积守恒判据识别。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

__all__ = ["ModelConfig", "EdgeGNN"]


@dataclass
class ModelConfig:
    node_dim: int = 7          # 式(25)：x(3) + log1p(s)(1) + f(2) + t̃(1)
    edge_dim: int = 10         # 式(27)/(28) 的统一布局，见 graph.py docstring
    hidden: int = 64
    layers: int = 3            # 原文的 L 层
    dropout: float = 0.1
    residual: bool = False     # False = 严格按式(30)(32) 的纯映射


def _mlp(dims: list[int], dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(dims[i + 1]))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class EdgeGNN(nn.Module):
    """式(30)-(33) 的边–节点交替消息传递 + 边分类头。

    * `node_lift` / `edge_lift`：把式(25)(27) 的初始特征线性投影到隐藏维
      （原文 h^(0)/e^(0) 就是初始特征，线性提升是维度对齐所必需）；
    * 每层：`e ← φ_e([e, h_u, h_v])`、`h ← φ_h([h, Σ_{u→v} e])`；
    * 输出：每条边的 logit（式33 的 ψ，再用 sigmoid 得 ŷ_e）。
    """

    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        h = self.cfg.hidden
        self.node_lift = nn.Linear(self.cfg.node_dim, h)
        self.edge_lift = nn.Linear(self.cfg.edge_dim, h)
        self.edge_updates = nn.ModuleList(
            [_mlp([3 * h, h, h], self.cfg.dropout) for _ in range(self.cfg.layers)])
        self.node_updates = nn.ModuleList(
            [_mlp([2 * h, h, h], self.cfg.dropout) for _ in range(self.cfg.layers)])
        self.head = _mlp([h, h, 1], self.cfg.dropout)      # 式(33) 的 ψ

    def forward(self, node_feat: torch.Tensor, edge_index: torch.Tensor,
                edge_feat: torch.Tensor) -> torch.Tensor:
        """返回每条边的 logit，形状 (E,)。`edge_index` 为 (2, E)。"""
        h = self.node_lift(node_feat)
        e = self.edge_lift(edge_feat)
        u, v = edge_index[0], edge_index[1]
        for edge_mlp, node_mlp in zip(self.edge_updates, self.node_updates):
            msg = torch.cat([e, h[u], h[v]], dim=-1)
            upd = edge_mlp(msg)
            e = e + upd if self.cfg.residual else upd          # 式(30)
            agg = torch.zeros_like(h)
            agg.index_add_(0, v, e)                            # 式(31)
            nupd = node_mlp(torch.cat([h, agg], dim=-1))
            h = h + nupd if self.cfg.residual else nupd        # 式(32)
        return self.head(e).squeeze(-1)
