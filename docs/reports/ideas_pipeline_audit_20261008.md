# ideas.pdf → 代码 pipeline 审核（2026-10-08）

**结论：工程链路已打通，严格按 ideas 构建完成的验收不通过。** `paperpipe` 已有完整阶段入口、二分类 GNN、桥接和官方提交产物，但目标函数、代价传递、历史消息通路、tracklet 单位和谱系、监督标签存在可复现问题；式(25) 的 encoder 特征和 §2.0.1 的部分分割错误处理也未落成。外层 `src/celltracker` 是另一套带工程扩展的实现，不能替代 `paperpipe` 的论文符合性验收。

本报告审核源码快照 `ee33622c278c5c392c0c7ecc1dc87584b9d5a813`。依据为本仓库 [ideas.pdf](/home/henry/ot_idea/CellTracker/ideas.pdf) 全部 16 页，SHA256 `975d7f55182d979c4e83b905a05d737770400aadc7f7f272bdd608f8b5c47045`。覆盖两个运行入口、配置、检测转换、复用求解器、图构建、训练/推理、轨迹重建、tracklet、导出/校验、公式映射和已有实验记录。未修改算法、数据、权重或历史结果；未使用云 GPU，未重跑官方评测。

源码判断与 15 项小型 CPU 探针见 [复现脚本](/home/henry/ot_idea/CellTracker/experiments/E0.4_ideas_audit_20261008/artifacts/probe_pipeline.py) 和 [数值证据](/home/henry/ot_idea/CellTracker/experiments/E0.4_ideas_audit_20261008/metrics.json)。既有测试分开运行：外层 89 项、paperpipe 16 项均通过。它们验证了已有行为，但没有覆盖下面的论文目标/语义错误。

## 1. 已经接起来的链路

`nnU-Net 掩码 → 实例化/H5 → 测度和空间图 → 两遍运动估计 → 相邻 OT/FGW → 多尺度精炼 → 直接/桥接候选图 → GNN 或 OT 规则 → 二层 tracklet → 连续轨迹/空洞补画 → CTC 格式校验 → 官方评测`。

这条链在代码入口和历史端到端产物中均存在。历史 P4 官方 TRA 为 seq01 **0.898249**、seq02 **0.899119**；这些是既有实现的测量结果，不能据此证明每个公式正确。P5 的 286 次桥接和 85 次分裂说明相关分支确实执行过，也不能替代机制验收。上述数字只引用历史产物，本次未生成新的性能结论。

![当前链路的审核状态](/home/henry/ot_idea/CellTracker/experiments/E0.4_ideas_audit_20261008/figures/pipeline_audit.svg)

## 2. 逐公式/机制对照

“通过”指对应定义或代码关系可确认；“偏离”区分明确实现错误、已标注工程选择和未实现项。任选扩展无需强制开启，选择合理的算法也无需逐字相同，但所求目标和关键数据语义必须一致。

| 原文（PDF 页） | paperpipe 实际状态 | 外层 src/celltracker 状态 | 判定 |
| --- | --- | --- | --- |
| 式(1)–(3)，质量/最小特征（2） | 体积归一质量与 `[x,s]` 有实现；真实实例默认 volume | 支持 volume，但 GT marker 默认 uniform，有决策记录 | 基础定义具备；marker 特例需独立说明 |
| 式(4)–(7)，kNN、W、D（3） | kNN 对称化与 D/W 实现；式(6) 多了分母因子 2；D 默认邻接截断 | 也有分母因子 2；D 在物理 spacing 路径仍按体素算 | 核宽不逐式一致；稀疏近似须说明；外层单位不统一 |
| 式(8)，位置/尺寸代价与 Rmax（3–4） | 初始 C 符合定义，坐标按 spacing；精炼后 C 被替换 | 尺寸项的尺度为 `sigma_s × mean(s_dst)`，参数语义不同；默认 beta=0 | 初始 paperpipe 通过，下游 C 不通过 |
| 式(9)，FGW 结构项（4） | 使用 D；结构项展开及梯度正确；求解接受准则不含全部正则 | 同族 FGW 求解器，同问题 | 结构定义通过，完整优化目标不通过 |
| 式(10)，平衡边际（4） | tau_a/tau_b=None 走平衡 Sinkhorn | 同求解器，默认平衡 | 接口/实现具备；不可行支撑仍需边际检查 |
| 式(11)，熵（4） | 多尺度的 `entropy` 把 `−ΣΓ` 写成 `+ΣΓ` | 基础 Sinkhorn 的熵形式正确 | paperpipe 全局目标不通过 |
| 式(12)–(14)，熵正则/非平衡 OT（5） | eta=0 的 log-Sinkhorn 具备；eta>0 外层线搜索只看 L | 同族问题 | 纯 OT 具备；FGW 完整目标不通过 |
| 式(15)–(16)，时间展开网络（6） | 支撑边函数存在；GNN 图只接当前直接/桥接边，历史帧断开 | 外层图会接窗口内时间边 | paperpipe 历史上下文未生效 |
| 式(17)，全局联合目标（7） | 有局部回溯入口；熵符号、C 漂移、结构梯度缺失使实际优化不等于式(17) | 回溯只检查时间正则，没有联合基础目标 | 两版都不通过 |
| 式(18)–(19)，多尺度乘积（7–8） | 有逐尺度 lambda 与直接跳帧；默认比较条件概率；raw 梯度仍除行和 | 只在条件概率空间做，共用 lambda；跳帧丢 spacing/体积 | 属目标改写加实现缺陷，须先定口径 |
| 式(20)–(22)，运动先验（8–9） | 先无速度求解再估 v，第二遍 C 加预测位置残差，按 spacing | 两遍实现具备 | 主关联路径基本通过；速度诊断标成 µm 却用体素范数 |
| §1.5 末段，滑窗/二层 tracklet（9–10） | 完成轨迹后按节点数切不重叠块；没有独立更严格阈值；只用末步速度/端点 | 整条轨迹当超节点，非短滑窗 | 均是替代策略；paperpipe 另有单位/亲子映射 bug |
| 式(23)–(24)及 §1.6 出生/死亡/分裂（10） | 单目标 argmax+Γ/C 门槛具备；生死只统计；分裂有体积判据 | 行内占比阈值；缺论文所述生死和规则体积判据 | 规则路径部分完成；不能称非平衡生死机制已生效 |
| 式(25)，节点 `[x,s,encoder f,t/T]`（13） | 默认强度均值/标准差；encoder 仅消费者接口，侧车匹配有 bug | 强度统计；加载器缺 std，部分路径该维恒 0 | encoder 链路未完成，当前为已标注工程近似 |
| 式(26)，Γ和C双阈值（13） | 初始原始 Γ 双阈值与 topk=0 开关正确；默认阈值为 frac×a，且消费漂移的 C | 行归一 Γ；topk=0 仍强制 top1；C 门槛仅显式配置时启用 | paperpipe 条件式接口具备；默认/实际输入有偏离 |
| 式(27)–(28)，跨帧/帧内特征（13） | 六项都在，尺寸 log 预处理；f_dist 实际重复位置距离；intra_knn 开关不生效 | 布局是另一套工程扩展；帧内边除标志外为零 | 不能把列名齐全当成特征语义齐全 |
| 式(29)，真实前驱标签（13） | 二分类定义具备；检测→GT 分母错误、空亲子字典被转 None | 三分类已标注扩展；检测→GT 分母为共用前处理问题 | 监督语义验收不通过 |
| 式(30)–(33)，消息传递/sigmoid（14） | edge MLP → incoming sum → node MLP → 单 logit，结构对应 | 带残差、三类 softmax，属已注明扩展 | paperpipe 架构通过；输入图/外观仍不完整 |
| 式(34)–(35)，BCE/可选 OT 正则（14） | BCE 和 prob×C 都具备；两者均取 mean 可视为共同缩放；C 已被上游污染 | 三分类加权 CE 与相关概率 OT 正则 | paperpipe 损失形式基本具备，输入 C 不通过；可选正则无需强制打开 |
| §2.0.1 漏检/假阳性/过分割/欠分割（15–16） | 桥接具备；孤立节点仍输出；未实现不可靠区域标记及重新评分/过分割代表节点选择 | 缺相应完整处理链 | 只完成部分鲁棒性机制 |
| CTC 导出/格式（工程） | 流式掩码写出与校验具备，空洞通过移位补画；CLI 可跳过校验并送官方 | 格式修复器具备 | 工程可用；补画是论文外补充，不能替代漏检恢复证据 |

**纠正旧对齐文档的一点：式(9) 原文用的是 D，不是 W。** 求解器内部出现 `D*D` 是平方差展开的正常代数，并非“应该改成 W”或“把距离错算成平方距离”。§1.7 允许稀疏/截断近似，不能把未做完整密集距离矩阵直接定性为缺模块；但将非邻接 D 置零并不等于在 FGW 四重和中忽略所有相关项，应说明近似定义。

## 3. 阻断严格验收的问题与复现

下面的数值是公式/数据语义探针，不是 TRA 改善量，不适用“以本地代理指标作性能结论”。尚未量化各 bug 对已有官方数字的独立影响。

### A1. FGW 接受准则没有优化式(12)/(14) 的完整目标（共享核心问题）

[FGW 求解器](/home/henry/ot_idea/CellTracker/paperpipe/src/vendor/celltracker/ot/fgw.py:39) 的目标只算式(9) 的 L，线搜索与停止条件都使用它，漏掉 εEnt 与 KL；初始化还用 C，而非 `(1−eta)C`。这不是仅有日志名称问题，会拒绝完整目标的下降方向。外层 [同族求解器](/home/henry/ot_idea/CellTracker/src/celltracker/ot/fgw.py:39) 也有此问题。

**解析复现**：一个源/目标，a=b=1，D=D′=0，C=3，eta=0.5，eps=tau_a=tau_b=1。式(14) 退化为 `1.5p + p(log p−1) + 2(p log p−p+1)`，解析最优 `p=exp(−0.5)=0.606531`。代码返回 **0.367879**，外层接受 0 次更新。当前 e2e 配置 eta=0.3、非平衡开启，实际走此求解器。不能据旧消融负收益证明论文 FGW/非平衡本身无效，需先验证正确目标后重新比较。

### A2. 多尺度全局目标错误，且原始 C 被优化代理 G 覆盖

[multiscale.py](/home/henry/ot_idea/CellTracker/paperpipe/src/papertrack/temporal/multiscale.py:50) 有四个独立问题：

1. 式(11) 写成 `ΣΓ logΓ + ΣΓ`，与原文相差 `2ΣΓ`。探针总质量 0.5 时原文 **−1.512663**，代码 **−0.512663**。非平衡质量可变，此差不是可忽略常数。
2. `resolve_with_cost` 返回 `cost=G`，然后 `base[i]=cand`。后续精炼继续拿 G 当基础 C，下游式(26)/(27)/(35) 也消费 G。[覆盖点](/home/henry/ot_idea/CellTracker/paperpipe/src/papertrack/temporal/multiscale.py:348)。探针同一帧对的 C 最大变化 **6.517946**。验收应固定原始 C，仅更新计划与显式优化状态；当前 history 的目标值基于移动的 C，不能证明固定式(17) 单调下降。
3. 实际重解 G=`C+时间梯度扰动`，没有文档声称已线性化进去的 eta 结构梯度，也没有相应 `(1−eta)` 项。[更新点](/home/henry/ot_idea/CellTracker/paperpipe/src/papertrack/temporal/multiscale.py:298)。Ds 参数传入后未参与重解；这使全局 FGW 的保真性继续丢失。
4. `reg_grad` 无论 raw/cond 都除行和。[梯度点](/home/henry/ot_idea/CellTracker/paperpipe/src/papertrack/temporal/multiscale.py:275)。raw 模式不应除；cond 模式若 Q=Γ/r，正确链式导数应含 `(g−Σ_j g_j Q_j)/r`，现有 `g/r` 漏掉行和导数。默认 cond 的概率乘积也不是式(18)/(19) 字面 Γ 乘积。

应先由用户确认保留字面质量乘积还是将条件概率复合明确写成方法改写，再修实现。原文在式(18)/(19) 后介绍 Markov 行归一化，不能直接据此把公式中的所有 Γ 解释为 Q。这里存在论文定义的尺度疑问，审核揭示问题，不擅自选新机制。

### A3. GNN 历史帧节点存在，但无法传递历史信息

[build_graph](/home/henry/ot_idea/CellTracker/paperpipe/src/papertrack/graph/build.py:170) 收入 t−ctx_window 的节点，然而跨帧边只添加 t→t+1 和 t→t+gap，[拼接点](/home/henry/ot_idea/CellTracker/paperpipe/src/papertrack/graph/build.py:305) 为 intra+target，没有窗口内上下文时间边。

探针当前 t=1 时，图含帧 0/1/2/3；边的帧对只有 `(0,0),(1,1),(1,2),(2,2),(3,3)`。把帧 0 全部节点特征加 1000，当前预测的最大变化 **0**。增加历史窗口不会让历史进入当前消息传递。外层旧图反而存在窗口内时间边；不能在重构后宣称该能力已保持。

### A4. 二层 tracklet 混用单位，并在切段后丢亲子关系

[tracklet 距离](/home/henry/ot_idea/CellTracker/paperpipe/src/papertrack/longrange/tracklet.py:147) 用体素范数，与已经是 µm 的 r_max 比较；同一函数检查中间帧时却乘了 spacing。运动预测项 d2 也未换算。探针横向移动 10 体素、spacing=0.09 µm，为 **0.9 µm**，应该在 3 µm 门限内，代码候选数却为 **0**。这是明确的同族单位 bug；外层 tracklet 与旧跳帧求解也要同步查验。

[亲子重编号](/home/henry/ot_idea/CellTracker/paperpipe/src/papertrack/longrange/tracklet.py:223) 把原父 ID 映射到**第一段**，而分裂应连到**终末段**。父帧 0–5、窗口 3、两个孩子起帧 6，禁止片段合并的探针中，孩子指向结束于帧 2 的父首段，随后 normalize 清掉错误父关系：输入 **2** 个孩子有父，输出 **0** 个；`parent_fixed=2`。CTC 格式通过可能是清除谱系后的结果。已有 P4 run_info 记录 `parent_fixed=72`，该计数是排查线索，不能认定这 72 次全由本 bug 造成。

此外，当前按轨迹检测节点数切块，不是原文的滑动 3–5 帧高置信局部窗口；没有独立更严格 OT 阈值；聚合表示缺平均轨迹/平均速度/尺寸变化，速度只用末一步。属于需明确说明的工程替代，不能以“window=5”作为原机制完成的证据。

### A5. 检测→GT 映射的分母不正确，空血缘被误当作无标签

[predict_to_h5.py](/home/henry/ot_idea/CellTracker/scripts/predict_to_h5.py:142) 先筛 GT 与预测同时前景，再从交集计算 g_areas；它表示“GT 被任意预测覆盖的部分”，不等于完整 GT 标记体积。原注释的“重叠 >50%”据此失效。运行真实 CLI 的最小例子：完整 GT 10 体素、预测只覆盖 2 体素，期望映射 0，实际 **gt_label=1**。该前处理是两版训练共同入口，可能污染正负监督；当前不能量化受影响真实实例数。

[runtime dump_graphs](/home/henry/ot_idea/CellTracker/paperpipe/src/papertrack/runtime/pipeline.py:247) 把合法空血缘 `{}` 变成 None，而 [标签段](/home/henry/ot_idea/CellTracker/paperpipe/src/papertrack/graph/build.py:280) 遇 None 连同同 ID 的移动边一起跳过。无分裂序列仍应有移动正样本：探针 `{}` 有 **2** 个正标签，None 有 **0** 个。需要分别表示“有效但无分裂”和“缺少监督”，后者训练应明确报错。

### A6. 非平衡的出生/死亡没有成为决策机制

[规则重建](/home/henry/ot_idea/CellTracker/paperpipe/src/papertrack/reconstruction/rules.py:155) 将死亡计数加一后继续单目标关联；出生仅为已经未分配的目标统计，随后无论列质量如何都新建轨迹。`death_veto` 在配置存在，在实现中没有读取。

探针把两条 OT 行流量设在死亡阈值之下，true/false 都得到 **n_death=2、同 ID 继续到下一帧**。e2e 配置注释已经说明只作诊断，这是有意识的工程偏离；但 true 开关无作用是额外缺陷。GNN 路径只按分数/冲突/体积赋 ID，不消费这些生死阈值。应把“保守保留关联”与“按论文非平衡生死判据终止/新建”分开验收。

### A7. 式(25) 的 encoder 外观链路没完成；已有消费者也可能错配

当前 `f_source=intensity` 已明确标注工程近似，可作为简化设置，但不能称式(25) encoder 已接入。没有 encoder 前向/聚合导出脚本；[缺路径时](/home/henry/ot_idea/CellTracker/paperpipe/src/papertrack/runtime/pipeline.py:279) 会静默走强度路径。

[load_encoder_features](/home/henry/ot_idea/CellTracker/paperpipe/src/papertrack/graph/build.py:86) 在检测数量相同时忽略侧车质心顺序直接采用特征，只有数目不同时才匹配。倒序质心探针期望 `[10,20]`，实际 `[20,10]`。不同数目时最近邻也没传 spacing。接 encoder 前必须保证实例身份、特征维度、帧覆盖、物理匹配和训练/推理一致。

## 4. 其他需要明确记录的问题

| 项目 | 证据与影响 | 性质 |
| --- | --- | --- |
| 式(6) 分母因子 2 | [measure.py:151](/home/henry/ot_idea/CellTracker/paperpipe/src/papertrack/representation/measure.py:151)，双项距离/尺度=1 时原文 exp(−2)=0.135335，代码 exp(−1)=0.367879；可通过重定义 sigma 等价，但目前公式/标定语义未对应 | 明确公式偏差，非必然性能下降 |
| 新 GNN 每批重新置换 | [train.py:174](/home/henry/ot_idea/CellTracker/paperpipe/src/papertrack/gnn/train.py:174)，10 图每批 2，固定种子例子一轮 10 次抽取只见 6 图；不是完整 epoch 遍历。外层训练置换在 epoch 外层，**没有此 bug** | 训练流程缺陷 |
| f_dist 与 intra_knn | [build.py:203](/home/henry/ot_idea/CellTracker/paperpipe/src/papertrack/graph/build.py:203) 实际使用 measure.knn_k；graph.intra_knn 不参与。f_dist 的输入 vol=None，仅坐标，不能称 encoder/外观差异 | 特征语义及开关不实 |
| 孤立 FP 没被排除 | [rules.py:334](/home/henry/ot_idea/CellTracker/paperpipe/src/papertrack/reconstruction/rules.py:334)，所有未继承节点都建新轨迹；4 个无接受边节点输出 4 条轨迹。已有实验也证实直接删孤立短轨迹会误伤真实细胞 | 原文机制缺口；修法需先讨论，不能直接删轨迹 |
| 欠分割/过分割专门处理 | 没有不可靠区域标记→重评分，也没有选择过分割代表节点的流程；入/出度可被消息聚合隐式反映，但仅限制父子数不能替代这些流程 | 未实现；原文仅文字，落成新方案前需用户决定 |
| 训练/推理配置与来源契约 | train CLI 的 TrainConfig 与 pipeline 的 gnn 配置独立；checkpoint/图没有完整检测/图配置指纹校验。R11 靠操作记录保证，程序不能阻止错权重/错源 | 工程保障缺口 |
| 验证块重叠 | 按图文件前后块切分，图包含额外上下文/未来帧；边界附近共享节点，无 purge。不能直接称为完全独立时段 | 实验协议限制 |
| seq02 “留出”的范围 | P4 是 GNN 跨序列迁移；nnU-Net fold0 仍含 seq02 的 150 帧，依据 [决策 0003](/home/henry/ot_idea/CellTracker/docs/decisions/0003-seq02-eval-protocol.md) 及原 splits | 不能宣称全链路未见序列泛化 |
| 格式校验可绕过 | [run_paper_pipeline.py:90](/home/henry/ot_idea/CellTracker/paperpipe/scripts/run_paper_pipeline.py:90) 的 `--no-validate --official` 可绕过 E2；官方失败仍进入 finish | 入口/失败状态缺陷 |
| “自包含”的范围 | 算法包 vendor 化成立；[official 入口](/home/henry/ot_idea/CellTracker/paperpipe/scripts/run_paper_pipeline.py:104) 仍依赖外层 scripts/cloud_eval.py；检测和 encoder 导出也不在包内 | 包装声明需限定范围 |
| 外层消融与诊断 | [build.py:196](/home/henry/ot_idea/CellTracker/src/celltracker/graph/build.py:196) topk=0 强制 top1；[train.py:173](/home/henry/ot_idea/CellTracker/src/celltracker/gnn/train.py:173) 二分类存在性指标覆盖而未累加各 batch；[跳帧](/home/henry/ot_idea/CellTracker/src/celltracker/ot/multiscale.py:43) 未传 spacing/体积 | 原版本同族待办，影响消融/本地诊断可信度 |

不把以下事情列为“没按论文搭好”：eta=0（原文明示允许）、不开任选 OT 损失、没做可选空间分块、用原分辨率求跳帧 OT、数值用 LayerNorm/梯度裁剪、BCE 和 OT 损失共同除以边数。后者改变批权重，但相同批内只是整体正比例缩放，不能简单称公式错误。

单节点 kNN 曾被怀疑越界，实际探针运行正常，**排除该疑点**。本报告只保留已验证行为或明确标注的静态缺口。

## 5. 后续验收顺序（建议，尚未执行）

1. 先确定“严格版”的唯一口径：式(18)/(19) raw 还是明确改成条件概率；Γ 绝对门槛还是按 a 缩放；生死判据是否执行；tracklet 是否按原文滑窗。涉及方法改写的选项须用户确认，不由此次审核代选。
2. 修基础正确性：FGW 完整目标/解析例；熵符号；固定原 C 和正确多尺度梯度；标签映射/空血缘；历史时间边；tracklet 物理单位及终末父段；每项补回归测试，并检查两份同族代码。
3. 补原文已明确要求但仍空缺的 encoder 特征与局部 tracklet 数据流；针对 §2.0.1 无具体公式的错误处理先形成可审阅方案，避免擅自自创。
4. 在小样本上验证全部开关实际改变对应机制，检查原 C 不变、固定全目标下降、历史节点能影响输出、谱系保持、正负标签符合真值。然后按同一检测来源重建图/重训，生成双序列官方 DET/SEG/TRA 与关键多种子复验。
5. 重测新设置的噪声地板，追加公式表/实验 INDEX 更正，不改历史结果。不能仅凭修复本地探针直接宣称 TRA 会改善，也不应继续拿有目标函数偏差的旧消融归因论文模块。

当前无需先烧 GPU或调阈值：应先消除会破坏论文定义与监督的数据流问题。完整性能结论要在修复后按 R4/R9/R10/R12 的官方双序列、独立重训和噪声口径重新验收。
