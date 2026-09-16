# 决策记录 0001：质量向量 `a_i` 采用等质量而非"与尺寸成正比"

日期：2026-09-16 · 状态：**已采纳**（用户确认：按建议执行但保留记录与说明）

## 背景

`ideas.pdf` 式 (1) 定义每帧经验测度的质量：

```
a_i^t = s_i^t / Σ_k s_k^t ,   μ^t = Σ_i a_i^t δ_{f_i^t}
```

即"质量与细胞尺寸成正比"。实现里对应 `CostConfig.mass_mode ∈ {"volume", "uniform"}`，
默认取 **uniform**（`a_i = 1/n_t`）。

## 证据：在 CTC 数据上"尺寸"与"质量"被解耦

用 `Fluo-N3DH-CE` seq01 的全部 **358 次真实分裂**做验证
（脚本见 `experiments/E2.6_CE01_division_diag/notes.md` 相关分析）：

| 统计量 | 值 |
| --- | --- |
| 子细胞 marker 体积之和 / 父细胞 marker 体积 | **恒为 2.00**（p10–p90 全为 2.00） |
| 可分析的分裂事件 | 358 / 358 |

即 CTC 的 `_GT/TRA` 标注是**等体积的标记点（marker）**，每个细胞无论大小都占相同的
体素数。因此：

- "体积"不携带细胞质量信息，`s_i` 在 marker 上近似常数；
- 若强行按式 (1) 取 `a_i ∝ s_i`，等价于均匀质量加上标注噪声；
- 更关键的是，**式 (1) 的物理直觉（质量守恒）在 marker 上不成立**：
  分裂时子代 marker 体积之和是父代的 2 倍（各占一份等体积 marker），
  而不是 1 倍。

补充证据（`Fluo-N3DH-CHO`）：marker 与银标准全分割的体素比为
5171 : 117035（约 1:23），即 marker 只占细胞的极小一部分。

## 决策

1. **默认使用等质量** `a_i = 1/n_t`（`mass_mode="uniform"`）。
   这在 OT 里等价于"每个细胞携带相同质量"，与 marker 标注的语义一致。
2. `mass_mode="volume"` **保留为可选项**，用于将来接入真实分割
   （nnU-Net 输出的细胞掩码面积/体积）时的对照实验。
3. 因此**不使用式 (1) 的体积成比例质量**，但保留式 (1) 的测度论框架
   （`μ^t = Σ a_i δ_{f_i}`）与全部下游公式不变。
4. 与之一致地，式 (14) 的 KL 边缘惩罚、式 (23)(24) 的重建阈值都按等质量口径实现。

## 影响

- **不会**改变方法的理论结构：OT/FGW/多尺度正则/GNN 各式的形式都不依赖 `a_i` 的具体取法。
- **会影响**两点，需在论文中明确：
  1. 分裂判定不能使用"体积守恒"（§1.6 原文的 `s_j1 + s_j2 ≈ s_i`），
     必须改用**行内质量分配**判据（一行对多个目标显著传输）；
  2. 与使用真实分割质量的方案不可直接比较质量尺度，需在实验设置中说明。

## 复现

```bash
# 验证 marker 等体积性（父子 marker 体积比恒为 2.00）
PYTHONPATH=src python -c "
import numpy as np, h5py
from collections import defaultdict
f = h5py.File('data/interim/Fluo-N3DH-CE_01.h5','r')
tracks = np.asarray(f['tracks'])
vol = {int(k): dict(zip(np.asarray(f['frames'][k]['label']).tolist(),
                        np.asarray(f['frames'][k]['volume']).tolist())) for k in f['frames']}
by = {int(l): (int(b), int(p)) for l, b, p in zip(tracks['label'], tracks['begin'], tracks['parent'])}
children = defaultdict(list)
for l, (b, p) in by.items():
    if p: children[p].append(l)
ratios = [ (vol[by[c1][0]].get(c1,0)+vol[by[c1][0]].get(c2,0)) / vol[by[c1][0]-1][p]
           for p,(c1,c2) in ((p,c) for p,c in children.items() if len(c)==2)
           if by[c1][0]-1 in vol and p in vol[by[c1][0]-1] and c1 in vol[by[c1][0]] ]
print('mean ratio', np.mean(ratios), 'n =', len(ratios))
"
```
