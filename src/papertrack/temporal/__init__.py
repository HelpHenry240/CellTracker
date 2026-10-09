"""§1.4 时间展开图 + §1.5 多尺度时间耦合（`ideas.pdf` 式 15–19）。

* `multiscale.time_expanded_edges` 式(15)(16) 节点并集与 `E_inter` 的查询接口
* `multiscale.refine_couplings`    式(17)–(19) 的交替优化 + 回溯线搜索
* `multiscale.entropy`/`kl_divergence`/`pair_objective`  式(11)(13)(17) 的分量
"""

from .multiscale import (MultiscaleResult, entropy, kl_divergence, pair_objective,
                         refine_couplings, time_expanded_edges)

__all__ = ["MultiscaleResult", "refine_couplings", "pair_objective", "entropy",
           "kl_divergence", "time_expanded_edges"]
