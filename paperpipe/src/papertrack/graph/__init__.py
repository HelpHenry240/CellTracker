"""§2.0.1 时间展开动态图（`ideas.pdf` 式 25–29，含跨帧桥接边）。

* `build.build_graph`           单帧对的时间展开图（节点/帧内边/候选边/标签）
* `build.build_dataset`         把整条序列写成 `pair_XXXX.npz` 供训练
* `build.load_encoder_features` 式(25) 的 `f_i`（论文口径：分割网络 encoder 特征的侧车文件）
* `build.EDGE_COLS`/`NODE_COLS` 特征布局的命名常量（E3：禁止裸写下标）
"""

from .build import (EDGE_COLS, EDGE_DIM, NODE_COLS, PairGraph, build_dataset,
                    build_graph, load_encoder_features, node_dim_for)

__all__ = ["build_graph", "build_dataset", "load_encoder_features", "node_dim_for",
           "PairGraph", "NODE_COLS", "EDGE_COLS", "EDGE_DIM"]
