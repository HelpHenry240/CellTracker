# ideas.pdf 与代码对应（重建版）

主实现位于 `paperpipe/src/papertrack/`，共用求解器、数据容器和 CTC I/O 位于
`paperpipe/src/vendor/celltracker/`。公式原文写在对应模块的注释中。

| 论文 | 实现 | 开关或参数 | 校验重点 |
| --- | --- | --- | --- |
| 式1–3：体积质量、点特征 | `representation/measure.py` | `measure.mass_mode` | 质量归一；尺寸 µm³ |
| 式4–7：kNN、W、D | 同上 | `structure_mode`、`knn_k`、`graph.use_intra_similarity` | 高斯无额外 2；FGW 使用 D；同帧边使用 W |
| 式8：特征代价与物理门限 | `representation/measure.py::build_cost` | `alpha`、`beta`、`sigma_s`、`r_max` | 位置 µm；尺寸归一；原始 C 不被代理梯度覆盖 |
| 式9–14：FGW、熵与 KL | `coupling/pairwise.py`、`vendor/celltracker/ot/` | `coupling.enabled`、`eta`、`eps`、`tau_a/b` | 线搜索评价完整目标，含熵和广义 KL |
| 式15–19：时间展开、多尺度 | `temporal/multiscale.py` | `enabled`、`ks`、`lambda_temp`、`line_search` | 原始 Γ 乘积；所有输入解析梯度；完整固定目标下降 |
| 式20–22：运动先验 | `longrange/motion.py`、`runtime/pipeline.py` | `motion.enabled`、`alpha_pred` | 第一次追踪硬关联估速；物理速度报告 |
| §1.5：局部 tracklet 和二层 OT | `longrange/tracklet.py` | `tracklet.enabled`、`window`、`local_confidence` | 滑窗高置信链；平均速度/尺寸趋势；父终末片段；稀疏分量求解 |
| 式23–24：硬关联 | `reconstruction/rules.py` | `theta_gamma`、`theta_c` | argmax 原始 Γ；双阈值 |
| §1.6：生死与体积守恒分裂 | 同上 | `birth_death_enabled`、`division_enabled`、`volume_conservation` | 行/列绝对质量；两个显著子目标及尺寸守恒 |
| 式25：节点特征 | `graph/build.py` | `node.f_source` | 冻结 encoder；按实例身份对齐；模型指纹 |
| 式26：候选边 | 同上 | `use_ot_candidates`、`cand_topk` | Γ/C 双阈值；top-k=0 完全关闭；候选损失报告 |
| 式27–28：时间/帧内边 | 同上 | `use_ot_features`、`intra_enabled`、`intra_knn` | 命名列；空间和外观差；历史时间图连通 |
| 式29：真实前驱二分类 | 同上、`detect/labels.py` | 完整 `gt_ids` | 完整 GT 标记作覆盖分母；欠分割保留全部身份 |
| 式30–33：节点–边 GNN | `gnn/model.py` | `gnn.enabled`、`layers`、`hidden` | 边更新、入边求和、节点更新、标量 sigmoid |
| 式34：BCE | `gnn/train.py` | `class_weighted_ce` | 默认无类别加权；每轮训练样本仅打乱一次 |
| 式35：OT 一致性 | 同上 | `lambda_ot` | 使用原始非负 C；训练脚本读取统一配置 |
| §2.0.1：漏检桥接 | `graph/build.py`、`reconstruction/rules.py` | `bridge`、`bridge_gap`、`bridge_scope` | 真缺检测的跨帧边进入监督与决策 |
| §2.0.1：FP/过分割/欠分割 | `reconstruction/rules.py` | `filter_isolated`、`mark_uncertainty`、`max_children` | 孤立节点、冲突选择、不确定区域、有限父子；不重画分割 |
| 格式与官方指标 | `reconstruction/exporter.py`、`runtime/validate.py`、`runtime/official.py` | `hole_policy` | 补画不覆盖；官方前强制校验；失败不可标成功 |

工程细节必须与方法定义区别报告：自适应 ε、节点位置归一与体积 log、统一十维边布局、
两项学习损失同时取均值、冲突排序、滑窗重叠链合并、粗层具体权重、CTC 补画/拆段、
时间块验证与重叠排除、身份侧车与来源契约、断点缓存。论文未指定这些离散实现细节。

条件概率乘积 `multiscale.norm=cond`、相对 Γ 阈值、生死相对质量、强度外观、候选 top-k、
残差更新和类别加权是工程对照；主模板使用 raw、绝对阈值、冻结 encoder、纯映射 BCE。
模板数值不能替代新设置的标定和双序列官方验证。

2026-09 的旧实验属于历史实现，数值保留在 experiments。2026-10 重建的回归和性能证据
分别记录，不能把旧实现的 TRA 当作新实现已达到的结果。
