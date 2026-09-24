> **目录结构**：下表里的"代码位置"用 `模块.函数` 记法，
> 对应的**文件路径**如下（`paperpipe/src/papertrack/` 下，按论文模块分了文件夹）：
>
> | 模块名 | 文件路径 |
> | --- | --- |
> | `measure` | `representation/measure.py` |
> | `coupling` | `coupling/pairwise.py` |
> | `multiscale` | `temporal/multiscale.py` |
> | `motion` | `longrange/motion.py` |
> | `tracklet` | `longrange/tracklet.py` |
> | `reconstruct` | `reconstruction/rules.py` |
> | `tracks` | `reconstruction/tracks.py` |
> | `export_ctc` | `reconstruction/exporter.py` |
> | `graph` | `graph/build.py` |
> | `model` | `gnn/model.py` |
> | `train` | `gnn/train.py` |
> | `infer` | `gnn/infer.py` |
> | `pipeline` | `runtime/pipeline.py` |
> | `validate` | `runtime/validate.py` |
>
> 复用到的原仓库模块副本在 `paperpipe/src/vendor/celltracker/`（清单见 `vendor/README.md`），
> 由 `_paths.py` 插到 `sys.path` 最前 ⇒ **paperpipe 自包含**。

---

## 1. 逐式对照

| 论文 | 公式要点 | 处理 | 代码位置 | 与原仓库的差别 |
| --- | --- | --- | --- | --- |
| 式(1)(3) | 质量 `a_i=s_i/Σs_k` | **复用** | `measure.masses` → `celltracker.cost.features.masses` | 默认 `mass_mode=volume`（原默认 uniform，依据 decisions/0001；预测实例体积真实可变，故按原文） |
| 式(2) | 点级特征 `f=[x,s]` | **复用+薄封装** | `measure.point_features` | 无（`x` 按 µm 表示，供式(6) 用） |
| 式(4)–(7) | 帧内 kNN 图 `G^t`、边权 `W`、距离 `D` | **重写** | `measure.knn_structure` | ① 原实现把 `W` 算出来又全部丢弃（死代码）→ 本实现**真正使用 W**（帧内边特征第 7 列）；② 距离按 `spacing` 换算成 µm（R3） |
| 式(8) | `C_feat=α‖Δx‖²+β((Δs)/σ_s)²`，`‖Δx‖>R_max⇒+∞` | **重写** | `measure.build_cost` | ① 原实现分母写成 `σ_s·mean(s_dst)`，与原文 `(s_i−s_j)/σ_s` 不同 → 恢复原文；② 原文要求 `α,β>0`，原默认 `β=0` → 本默认 `β=1`；③ `R_max` 单位 µm |
| 式(9) | FGW 结构项 `L(Γ)` | **复用** | `coupling.solve_coupling` → `celltracker.ot.fgw` | 无（`structural_term` 是四重和的等价矩阵形式；外层轮数上限 10 为 `ENG`） |
| 式(10)–(12) | 平衡熵正则 OT | **复用** | `coupling.solve_coupling` → `celltracker.ot.sinkhorn.sinkhorn_log` | ε 默认 `0.1×median(C)`（`ENG`，见 §3 的 E-1） |
| 式(13)(14) | 非平衡 OT（KL 松弛） | **复用** | 同上（`tau_a/tau_b`） | 默认取**有限 τ**（原文 §1.3 的主线就是非平衡；且 η_death/η_birth 只在非平衡下才有意义） |
| 式(15)(16) | 时间展开图 `V`、`E_inter` | **新增** | `multiscale.time_expanded_edges` | 原仓库没有把这两个定义落成显式对象 |
| 式(17)–(19) | 全局目标 + 多尺度时间一致性 `R^(k)` | **重写** | `multiscale.refine_couplings` | ① 原文是**逐尺度** `λ^(k)_temp`，原实现所有 k 共用一个 λ → 恢复逐尺度；② 暴露 `norm ∈ {cond, raw}`（原文同时给了条件概率定义与字面乘积两种口径）；③ 重解用 Sinkhorn 并加了回溯线搜索与有界扰动（`ENG`） |
| 式(20)–(22) | 运动先验（两遍式） | **复用** | `motion.py` → `celltracker.pipeline.motion.{estimate_velocity,attach_velocity}`；两遍驱动在 `pipeline.run_pipeline` | 无（原实现与原文一致；本包补了速度验收报告） |
| §1.5 末段 | 短时窗 tracklet → 超级节点 → 第二层 OT | **重写** | `tracklet.link_tracklets` | ① 原实现直接用整条第一层轨迹当超级节点，缺"滑动窗口 3–5 帧"切分 → 本实现按 `window` 切分；② 代价用式(20)(21)(22) 的超级节点版本；③ 第二层用式(14) 非平衡 OT；④ 切分后**父子关系随重编号一起搬**（原实现无此步） |
| 式(23)(24) | `j*=argmax Γ`，接受条件 `Γ≥θ_Γ, C≤θ_C` | **重写** | `reconstruct.reconstruct_from_ot` | θ_Γ 用**原始 Γ**（原文），原仓库用行归一化质量；θ_Γ 数值按"源质量占比"标定（`CALIB`，见 E-3） |
| §1.6 死亡/出生 | `m^out_i<η_death` / `m^in_j<η_birth` | **新增** | `reconstruct.reconstruct_from_ot`（统计 + 显式终止/起点） | 原仓库完全没有这两个机制 |
| §1.6 分裂 | 质量占比 ≥ 阈值 **且** `s_j1+s_j2≈s_i` | **新增体积守恒项** | `reconstruct.volume_consistent` | 原实现只用质量占比；本实现补上原文的守恒判据（`vol_tol`，`CALIB`） |
| 式(25) | 节点特征 `[x,s,f,t̃]` | **重写** | `graph.build_graph` | ① 原 `Detections.from_h5` 没读 `intensity_std` → 该维恒 0（死特征）；本包 `pipeline.load_detections` 修补；② `f_i` 支持 `encoder_npz`（论文口径）与 `intensity`（工程近似，默认）两条路 |
| 式(26) | 候选边 `Γ≥θ_Γ, C≤θ_C` | **重写** | `graph._candidates` | 用原始 Γ；`cand_topk=0`（原仓库默认 3，是 `ENG`，本包默认关闭＝原文口径） |
| 式(27)(28) | 边特征（跨帧 6 维 / 帧内差异） | **重写为统一布局** | `graph._edge_features`（10 维，见模块 docstring） | 原实现 11 维且含义不同（位移向量 / log 尺寸比 / `is_argmax` / 边类型标志）；本实现按式(27) 6 维 + 式(28) 的 `f` 距离 + 式(6) 的 `W` + 2 个边类型 one-hot |
| 式(29) | 边标签 `y_e∈{0,1}` | **重写** | `graph.build_graph`（标签段） | 原文是**二分类**（是否为真实前驱）；原实现是三分类（move/div）`ENG`。分裂的语义由 §1.6 的判据决定 —— 这是原文的分工 |
| 式(30)–(32) | 边–节点交替消息传递 | **重写** | `model.EdgeGNN` | 原文是纯映射 `e'=φ_e(...)`；原实现带残差连接。本默认 `residual=False`（可切） |
| 式(33) | `ŷ_e=σ(ψ(e^L))` | **重写** | `model.EdgeGNN.head` | 原文 sigmoid 单标量；原实现 3 类 softmax |
| 式(34) | 边级 BCE | **重写** | `train.train` | 原文二分类 BCE；原实现是加权三分类 CE。本实现按原文用 BCE（可选类频加权，默认关） |
| 式(35) | `L_OT-reg=Σ ŷ_e C_e` | **重写** | `train.train` | 原文是求和；批训练按候选边数取均值（`ENG`，见 E-8），且不再对 C 做均值归一 |
| §2.0.1 | 跨帧桥接边 `(i,t−1)→(k,t+1)` | **新增** | `graph.build_graph`（`bridge`）+ `reconstruct`（两条路径）+ `pipeline._stamp_hole` | 原仓库完全没有：`window>0` 的上下文边不参与损失、不参与决策。本实现把桥接边当**目标边**（Δt=2）参与训练与决策，并在导出阶段补画空洞帧 |
| §2.0.1 结构约束 | 每节点 ≤1 父、有限个子节点 | **复用口径 + 重写实现** | `tracks.normalize_tracks`、`reconstruct` | 原仓库的父/子约束一致；但原 `finalize_tracks` 对"不连续"一律**拆断**，与论文"轨迹不被截断"冲突 → 本包分 `fill`/`split` 两种口径 |

---

## 2. 直接复用的模块

| 复用对象 | 对应原文 | 理由 |
| --- | --- | --- |
| `celltracker.ot.sinkhorn.sinkhorn_log` | 式(10)–(14) | log-domain 非平衡 Sinkhorn，缩放因子 `λ=τ/(τ+ε)` 正是式(14) 的不动点形式 |
| `celltracker.ot.fgw.{structural_term,structural_grad,fused_gw}` | 式(9) | 四重和的等价矩阵形式（非近似），条件梯度 + Sinkhorn 线性 oracle；梯度有有限差分测试 |
| `celltracker.pipeline.motion.{estimate_velocity,attach_velocity}` | 式(20) | 用"上一轮追踪的硬关联"估速，与原文一字不差 |
| `celltracker.cost.features.masses` / `pairwise_distance` | 式(1)(3)(7) | 归一化与距离原语可直接用（本包补 spacing 支持） |
| `celltracker.track.base.{Detections,TrackResult,paint_result}`、`celltracker.data.ctc.*` | 数据容器 | 与论文无关的基础设施 |
| `celltracker.eval.ctc_io.ResultWriter`、`celltracker.eval.local_metrics.*` | 评测 I/O | 与论文无关（本地诊断只用于方向判断，R4） |
| `celltracker.gnn.data.{PairDataset,collate}` | 训练基础设施 | 键名中立；本包图 npz 与它对齐 |
| `celltracker.detect.instances.split_instances` | 前端实例化 | 论文 §2.0.1 的前置（nnU-Net 掩码 → 实例），原文未规定算法 |
| `celltracker.experiment.Experiment` | 工程留痕 | 八件套（AGENTS.md E9） |

---

## 3. 论文之外的工程补充（必须单独标注，`ENG`）

| 编号 | 内容 | 位置 | 依据 |
| --- | --- | --- | --- |
| E-1 | `ε = eps_rel × median(C)`（原文只写 `ε>0`） | `coupling.eps_from` | C 是平方距离；固定 ε 会让软匹配退化成硬分配。**要换成原文的绝对 ε 只需 `coupling.eps=<值>`** |
| E-2 | **空洞帧补画**：CTC 官方格式要求轨迹在 `[begin,end]` 内每帧都出现，而论文 §2.0.1 要求"轨迹不被截断" | `pipeline.export_ctc` / `_stamp_hole` | 二者冲突。默认 `hole_policy=fill`：把源帧实例体素按插值质心平移复制；无处可放时回退**覆盖**并计数 `hole_frames_overwritten`。备选 `split`（CTC 合法但截断轨迹）。**这是最需要你拍板的一条** |
| E-3 | θ_Γ 默认按"源质量的占比"标定：`θ_Γ(i)=frac×a_i^t` | `reconstruct._threshold`、`graph._candidates` | 原文式(24)(26) 的 θ_Γ 是绝对标量，但 Γ 的元素量级随 `Σa=1` 与帧内细胞数变化；给 `theta_gamma=<绝对値>` 即回到原文口径 |
| E-4 | `η_death/η_birth` 定义为行/列和占 `a_i/a_j` 的比例 | `config.ReconstructConfig` | 原文只说"低于阈值"，未给数值 |
| E-5 | 多尺度精炼的重解用 Sinkhorn（结构项在当前解处线性化），扰动幅度按 `median(C)` 有界、失败则回退 | `multiscale.resolve_with_cost` | §1.7 只要求"在全局目标下做**少量迭代式微调**"；重复求解 FGW 会让每对帧开销上一个量级（实测 11 帧 105s → 6.3s） |
| E-6 | 3–5 帧窗口 tracklet 的具体切分实现（长轨迹按 `window` 切片，长程关联交给第二层） | `tracklet.cut_pieces` | 原文只说"滑动时间窗口内形成局部 tracklet" |
| E-7 | `f_i` 默认用实例内强度统计（论文口径是 nnU-Net encoder 特征） | `config.NodeFeatureConfig.f_source` | 接口已留 `encoder_npz`（按帧存特征、按质心最近邻对齐）；encoder 前向 hook 的导出脚本待补（需要云端 nnU-Net） |
| E-8 | 式(34)(35) 的求和改为按候选边数取**均值** | `train.train` | 批训练稳定性；`lambda_ot` 因此是"每条边的权重"，与 C 的量纲（µm²）耦合 |
| E-9 | FGW 外层条件梯度轮数上限 `fgw_outer=10` | `config.CouplingConfig` | §1.7 未给迭代次数 |
| E-10 | 桥接边只在"帧 t+1 该节点缺失（R_max 内无候选）"时生成 | `graph.build_graph`（`bridge_scope`） | 原文前置条件就是"该帧真实细胞未被分割出来"。**实测**：不限定时真实数据上 10 帧出现 252 条隔帧轨迹/537 个空洞；限定后降到 20 条/22 个空洞 |
| E-11 | ε 用 `1.0 × median(C)`（不是原仓库的 0.1） | `coupling.eps_from` + `configs/paper_e2e_ce.yaml` | **端到端实测**：ε=0.1×median(C) 在 µm² 量纲下让传输计划退化成硬分配，真实分裂"双子都在候选集"只 0.308；ε=1.0 升到 0.808。副作用：ε 大 ⇒ 式(23) 的 argmax 失去区分度，**OT 规则路径**碎片化（官方 TRA 0.8717 @ 10715 轨迹）。GNN 路径不受此影响（主路径） |
| E-12 | 二层 tracklet 的 gap>1 串联只在"中间帧节点确实缺失"时允许 | `tracklet.link_tracklets` | 与 §2.0.1 桥接边同一前置条件。**实测**：不限定时 CE-01 造出 3487 个假空洞（中间帧其实有检测）、FP 5938；限定后 249 个空洞、FP 2704、官方 TRA +0.001 |
| E-13 | 多尺度精炼的目标函数**只对当前更新的帧对**做局部评估 | `multiscale.refine_couplings` | 纯性能（数学等价：其它帧对的基础目标不变，只有包含该帧对的时间窗正则项会变）。**实测**：194 对帧从 40+ min 降到 50 s（≈40×） |
| E-14 | η_death/η_birth 默认只作诊断（`death_veto=false`） | `config.ReconstructConfig.death_veto` | 原文把它们写成"标记终止点/起点"的判据，字面上可否决式(24) 已接受的关联；标定不准的 η 会不可逆删掉正确边（R5）→ 默认不否决，开关保留 |

---

## 4. 标定参数（论文未给数值 → R3 量纲标定）

用 `scripts/calibrate_params.py` 复算（打印经验分布 + 建议值）：

| 参数 | 字段 | 默认 | 标定方式 |
| --- | --- | --- | --- |
| `σ_x` | `measure.sigma_x` | `null` | 帧内**正距离中位数** |
| `σ_f` | `measure.sigma_f` | `null` | 特征差矩阵的正值中位数 |
| `σ_s` | `coupling.sigma_s` | `null` | 两帧**体积中位数** |
| `R_max` | `coupling.r_max` | 3.0 µm | 相邻帧最近邻位移的 **p99.9**（脚本给出；CE 实测 ≈2.7 µm） |
| `ε` | `coupling.eps_rel` | 0.1 | `0.1 × median(C)` |
| `τ_t,τ_{t+1}` | `coupling.tau_a/tau_b` | 1.0 | 有限值＝非平衡；论文只要求"τ 小⇒更多出生/死亡" |
| `θ_Γ` | `graph/reconstruct.theta_gamma_frac` | 0.05 | 真实边质量占比的 5% 分位（脚本给出） |
| `η_death/η_birth` | `reconstruct` | 0.5 | 无真实后继的源的行和占比分布（脚本给出） |
| `div_ratio` / `vol_tol` | `reconstruct` | 0.5 / 0.5 | 真实分裂事件的显著质量比例 / 体积守恒偏差分布 |
| `λ^(k)_temp` | `multiscale.lambda_temp` | {2:0.1, 3:0.05, 5:0.02} | §1.7"跳帧项作为软正则而非主导"→ 随 k 递减 |
| `K` | `multiscale.ks` | {2,3,5} | **原文给出** |
| `window` | `tracklet.window` | 5 | **原文给出范围 3–5 帧** |
| `λ_ot` | `gnn.lambda_ot` | 0.2 | 与 `median(C)` 联调（见 E-8） |

---

## 5. 还未实现的

1. **`f_i` 的 nnU-Net encoder 特征**（E-7）：消费者接口已实现，导出脚本（云端 hook）未写；
2. **§1.7 的"空间分块"**（更大规模数据）：未实现（当前规模不需要）；
3. **§2.0.1 的欠分割显式机制**（入/出度特征、"标记不可靠区域并重打分"）：原文描述但未给公式，
   当前只有结构约束（每节点 ≤1 父、≤2 子）+ 桥接边，未加显式度特征；
4. **§1.5 的"速度先验用于遮挡后重新捕捉"**：靠两遍式 + α′ 生效，但没有针对遮挡的专门实验（对应 C6/C7）。
