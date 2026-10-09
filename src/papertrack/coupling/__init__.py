"""§1.3 相邻帧最优传输耦合（`ideas.pdf` 式 8–14）。

* `pairwise.solve_coupling`       式(9)(12)(14)：FGW / 平衡 / 非平衡熵正则 OT
* `pairwise.solve_jump_coupling`  式(18)(19) 的 `Γ^{t,t+k}_direct`（跳帧直接耦合）
* `pairwise.eps_from`             式(12) ε 的口径（显式值 / `eps_rel×median(C)`）
* `pairwise.PairCoupling`         Γ、C、a、b 与逐行阈值（θ_Γ/θ_C）的载体
"""

from .pairwise import (PairCoupling, eps_from, load_coupling, save_coupling,
                       solve_coupling, solve_jump_coupling)

__all__ = ["PairCoupling", "solve_coupling", "solve_jump_coupling", "eps_from",
           "save_coupling", "load_coupling"]
