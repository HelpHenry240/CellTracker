"""时间展开图上的边分类 GNN（ideas.pdf 式 30–35）。

每层做"边更新 → 节点更新"的交替消息传递：

    e'_{u→v} = φ_e( e_{u→v}, h_u, h_v )
    m_v      = Σ_{u→v} e'_{u→v}
    h'_v     = φ_h( h_v, m_v )

最后在**候选跨帧边**上做 3 分类（0=无关联 / 1=同一细胞继续 / 2=分裂），
对应式 (33) 的边分类头，并把 OT 成本的一致性正则（式 35）用于训练。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

__all__ = ["ModelConfig", "EdgeGNN", "N_CLASSES"]

N_CLASSES = 3  # none / move / division


@dataclass
class ModelConfig:
    node_dim: int = 7
    edge_dim: int = 11        # 跨帧边 10 维 + intra 标志
    hidden: int = 64
    layers: int = 3
    dropout: float = 0.1


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
    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        h = self.cfg.hidden
        self.node_enc = _mlp([self.cfg.node_dim, h, h])
        self.edge_enc = _mlp([self.cfg.edge_dim, h, h])
        self.edge_updates = nn.ModuleList(
            [_mlp([3 * h, h, h], self.cfg.dropout) for _ in range(self.cfg.layers)])
        self.node_updates = nn.ModuleList(
            [_mlp([2 * h, h, h], self.cfg.dropout) for _ in range(self.cfg.layers)])
        self.edge_head = _mlp([h, h, N_CLASSES])

    def forward(self, node_feat: torch.Tensor, edge_index: torch.Tensor,
                edge_feat: torch.Tensor) -> torch.Tensor:
        """返回所有边的 3 分类 logits (E, 3)。`edge_index` 为 (2, E)。"""
        h = self.node_enc(node_feat)
        e = self.edge_enc(edge_feat)
        u, v = edge_index[0], edge_index[1]
        for edge_mlp, node_mlp in zip(self.edge_updates, self.node_updates):
            msg = torch.cat([e, h[u], h[v]], dim=-1)
            e = e + edge_mlp(msg)
            agg = torch.zeros_like(h)
            agg.index_add_(0, v, e)
            h = h + node_mlp(torch.cat([h, agg], dim=-1))
        return self.edge_head(e)
