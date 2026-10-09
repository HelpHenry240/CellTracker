# 三维细胞追踪 SOTA 调研：论文、评测指标、测试集，以及"我的模型该怎么比"

日期：2026-10-09
调研方式：直接抓取 CTC 官方榜单数据文件（`public.celltrackingchallenge.net/documents/*.xlsx`）+ 官方评测方法页 + 论文原文页；所有榜单数字**由我本机从官方 xlsx 重算**，附复核脚本见附录 A。
口径声明：**本文档中标注「一手核实」的数字来自官方发布文件；标注「待核」的来自二手检索结果。**

---

## 0. 执行摘要（先看这 16 条）

1. **三维细胞追踪没有第二个公认横评**。CTC（Cell Tracking Challenge）是唯一被 Nature Methods / TMI 级论文共同承认的基准；2023 起它把榜单拆成三个：**CTB（追踪）/ CSB（分割）/ CLB（连接）**。任何"我跟 SOTA 比了"的说法，最终都要落到 CTC 上。
2. **官方主排序指标是 OPCTB = 0.5·(SEG + TRA)**，不是 TRA 单值。只报 TRA 会被审稿人问"你的分割呢"。
3. **CTC 官方 AOGM 权重（一手核实，取自官方评测器日志表头）**：分裂操作 NS=5、漏检顶点 FN=10、多检顶点 FP=1、多余边 ED=1、缺失边 EA=1.5、语义错误边 EC=1。TRA 极值 `TRA = 1 − min(AOGM, AOGM₀)/AOGM₀`，其中 `AOGM₀ = 10·|V_GT| + 1.5·|E_GT|`。
4. **3D+Time 数据集共 9 个**（CE / CHO / DRO / TRIC / TRIF / A549 / H157 / MDA231 / N3DH-SIM+），其中 CE 是 0.09×0.09×**1.0** µm 的强各向异性体数据（z 比 xy 粗 11 倍）——这正是本项目踩过坑的地方。
5. **Fluo-N3DH-CE 的 TRA 已经饱和**：榜上前 6 名 TRA ≥ 0.967，第 1 名 AMOLF-NL = 0.9937。在这个数据集上，**TRA 差 0.05 已经不是"方法好坏"的量级，而是"前端能不能拆出实例"的量级**。
6. **CE 上真正没被解决的是 SEG**：人类 SEG = 0.8437，而最好的常规提交 SEG 只有 0.759。**实例级分割是 CE 的公开缺口**，也是本项目该打的靶子。
7. **本项目的定位（用我核实过的榜面对齐）**：GT-marker 档 TRA 0.996513（**注意：该档正确口径应是 LNK，见第 12 条；这里只作诊断信号**）已高于榜单第 1 名 AMOLF-NL 的 0.9937；但真实预测检测档 TRA 0.898（最新重建 0.898249）在 CE 的 23 个有效提交里**排第 11**，历史最佳档 0.9269 排第 10。**结论：追踪核心已是世界级，缺的全部在前端实例化。** 这与仓库自己的诊断（前端值 +0.079 TRA）完全一致。
8. **必须分三档检测来源报告**（GT marker / 预测实例 / 合成退化），否则数字不可比。榜单方法全部是"自己检测 + 自己追踪"的端到端档，**不能拿你的 GT-marker 档和榜单硬比**。
9. **你现在所有数字都在 CTC 训练序列上自评；榜单是隐藏测试序列**。要拿到可引用的名次必须提交 CTC（`regular` 常规档；若只有一套参数跑所有数据集则走 `generalizable` 泛化档——目前全球只有 5 个泛化档提交，第 1 名 PURD-US 的全库 TRA 只有 0.8549，**这是"诚实泛化"的可比区间**）。**提交要求（一手核实）**：需注册、上传**命令行可执行程序**（官方要在隐藏测试集上重跑）、评测**每月滚动**、官方评测机 ≥128GB RAM + ≥1× A100 80GB。
10. **必须补报的、榜单没有的东西**：CT/TF/BC(i)/CCA（谱系/分裂/细胞周期）、AOGM 六项分解、跨序列与跨数据集泛化、多种子 + 噪声地板、运行时间/显存/吞吐。**只报 TRA/SEG 的 3D 追踪论文在 2026 年已不足以支撑"优于 SOTA"的声明。**
11. **CTC 指标已被正式批评为"饱和 + 奖励删分裂"**（一手核实摘要）：[CHOTA](https://arxiv.org/abs/2408.11571)（Kaiser、Ulman、Rosenhahn，ECCV 2024 BIC workshop，**CTC 官方参与**）指出当前指标"偏向局部正确、弱奖励全局连贯"，方法是把"轨迹"重定义为整条谱系，代码在 [py-ctcmetrics](https://github.com/CellTrackingChallenge/py-ctcmetrics)；Cell-TRACTR 独立报告 **"removing division links can increase the tracking score"**。[Cell-HOTA](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1013071) 则把分裂准确度 DivA 作为一等项。→ **你的方法有显式分裂判据，必须补报 CHOTA / BC(i) / Cell-HOTA(DivA)，否则是在用"奖励删分裂"的指标给自己打分。**
12. **GT-marker 档应该改报 LNK，而不是 TRA**。CTC 的 **CLB 就是"在准备好的检测上只评连接"**（把检测质量从等式中移除），正是你那一档的官方框架；用 TRA 报"上界档"会让审稿人以为你在跟端到端方法比 TRA。→ **上界档报 LNK + 端到端档报 TRA/DET/SEG/OPCTB，分工干净。**（可直接对标：CE 上官方 CLB 第 1 是 PAST-FR LNK **0.9822**，第 2 是 **Trackastra = EPFL-CH** LNK **0.9714**；详见 §3.6。）
13. **2026 年出现了一篇直接挑战"候选图 + GNN"路线的论文**：[HOCT](https://arxiv.org/abs/2607.11754)（arXiv:2607.11754，CZ Biohub，一手核实摘要）原文指出候选图中"**共享节点的边之间标签一致性接近随机，因此候选图拓扑对 GNN 聚合不携带有用信息**"，并用"边中心 transformer + 3D 几何先验"在 CTC 与细菌分裂基准上取得 SOTA，**且不使用预训练图像编码器**。→ 你的 GNN 增益必须明确归因到**边特征**（外观/运动/几何）而非**图拓扑**，并把 HOCT 作为**独立佐证**引用——你自己的"FGW/结构项/非平衡全为负收益"与它同向，这是两条独立证据。
14. **好消息（新颖性定位）**：公开的 3D SOTA 只有两条路线——**transformer 关联**（Trackastra / HOCT）与 **ILP / 多假设优化**（Ultrack / MAMHT）。**没有已成型的"可微 / 神经 OT 细胞追踪器"**，你的 OT+FGW+多尺度+GNN 是一条独立技术线，新颖性成立；代价是**没有直接对手可比**，所以对比必须落在"**同检测下的关联模块替换**"（§5.3 第 9 条）。
15. **除 CTC 外还有两个值得加入的 3D 基准（强烈建议）**：① **Biohub – Cell Tracking During Development**（Kaggle / CZ Biohub，斑马鱼光片，**指标显式含分裂项** `adjusted_edge_jaccard + 0.1×division_jaccard`、**按胚胎留出**、稀疏 GT、**约 4,020 支队伍的公开解可直接同分比较**）；② **BlastoSPIM 1.0/2.0**（小鼠胚胎，653 张全标注 3D 图 / 18,336 核，**9.6× 各向异性**，谱系边 F1 **0.996** 待超越，且有**已发表的跨系统迁移**）。→ 三者构成"**榜单合法性（CTC）+ 分裂感知指标（Biohub）+ 宽松许可与跨系统迁移（BlastoSPIM）**"的完整三角。详见 §2.8。
16. **⚠️ 许可风险（必须处理）**：CTC 数据条款写明"**任何与 CTC 无关的公开科学用途需官方明确许可**""**禁止克隆数据集及其标注**"。→ 如果你打算**公开 CTC 上训练的权重、或发布数据集派生版本**，**先向 CTC 组委会书面确认**；做"可发布"研究时，宽松许可的基座是 BlastoSPIM / Zebrahub / LTDB / NIST / Kaggle 数据。

---

## 1. 三维细胞追踪的"任务版图"

"3D 细胞追踪"不是一个任务，而是任务族。先确定自己在哪一族，才知道该跟谁比。

| 任务族 | 典型对象 | 代表数据集 | GT 形态 | 主要评测指标 | 这个族的现实 SOTA |
| --- | --- | --- | --- | --- | --- |
| **A. 胚胎/谱系核追踪**（最主流） | 线虫、果蝇、甲虫、斑马鱼、小鼠胚胎的细胞核 | **Fluo-N3DH-CE**、Fluo-N3DL-DRO、Fluo-N3DL-TRIC/TRIF、Fluo-N3DH-SIM+；Zebrahub、BioEmergences-wt3、BlastoSPIM | CTC 用小圆点 marker（稀疏）；部分数据集只评"部分细胞" | CTC DET/SEG/TRA + CT/BC/CCA | 榜上 TRA 0.95–0.99（CE/DRO），**泛化档只有 0.85** |
| **B. 培养细胞 3D 侵袭/迁移** | 肺癌 A549/H157、乳腺癌 MDA231、CHO 核 | Fluo-C3DH-A549、Fluo-C3DH-H157、Fluo-C3DL-MDA231、Fluo-N3DH-CHO | 全量分割掩码（SEG 是硬指标） | SEG 权重高于 TRA | TRA/SEG 均 0.85–1.0，**接近饱和** |
| **C. 光片/器官尺度谱系** | 斑马鱼全脑、果蝇全胚胎百万级细胞 | Zebrahub、Cell Tracking During Development（Kaggle/CZ Biohub）、FlyWire（连接组，非时序追踪） | 稀疏点轨迹图（`.geff`） | 自定义（边 F1、谱系完整度），无统一榜 | 无公认横评；Ultrack 是事实标准工具 |
| **D. 跨模态/粒子追踪** | 荧光颗粒、cryo-ET 粒子、细菌生物膜 | CTC 2D 集、mother machine、cryo-ET 数据集 | 点轨迹 | 自定义 + HOTA 类 | 无统一榜。⚠️ **cryo-ET 竞赛（CZII Kaggle）是静态 3D 目标识别，没有时间轴，不是追踪**，别当追踪基准引用 |

**关键判断**：本项目（Fluo-N3DH-CE、核 marker、OT+GNN 关联、nnU-Net 前端）属于 **A 族 + B 族前端**。因此对标面就是 **CTC CTB/CSB 的 3D 行**，外加 **Zebrahub / DNN-Tracking** 这类"CTC 之外的时序谱系数据"（仓库已有 `docs/reports/dataset_survey_nuclei_seg_tracking.md` 做过详表）。

---

## 2. CTC：官方基准、指标与 3D 成绩（一手核实）

### 2.1 三个榜与官方排序公式

| 榜 | 全称 | 官方排序指标 | 参与算法 |
| --- | --- | --- | --- |
| **CTB** | Cell Tracking Benchmark | **OPCTB = 0.5·(SEG + TRA)** | 分割+追踪算法 |
| **CSB** | Cell Segmentation Benchmark | **OPCSB = 0.5·(DET + SEG)** | 纯分割 + 分割追踪算法 |
| **CLB** | Cell Linking Benchmark | **OPCLB = 0.5·(LNK + BIO)** | 追踪算法（顶点集合对齐后只评连接） |

每个榜都分两种提交：**regular（常规档，允许每个数据集一套参数）** 与 **generalizable（泛化档，全部数据集共用一套参数）**。CTB 页面按"每个数据集常规提交的前 3 名"给出榜单。

### 2.2 指标定义

- **DET**：`1 − min(AOGM-D, AOGM-D₀)/AOGM-D₀`，只比节点集合（AOGM-D₀ = 10·|V_GT|）。
- **SEG**：`SEG = Σ_{c∈TP} J(R_c, S_c) / (|TP| + |FN|)`，逐实例匹配（匹配规则 `|R∩C| > 0.5·|R|`，允许一个预测匹配多个 GT、反之不行），**漏检计入分母**。**注意两点**：① 对 CE 这类数据集，SEG 真值是 **silver truth**（由多个高分方法融合、非完美，CE 是 0.09×0.09×1.0 µm 的强各向异性），所以 SEG 上限远低于 1；② **CTC 的 gold 分割真值平均只有 17.8% 的实例覆盖**（人工标注成本所致），silver truth 把覆盖提到 **99.1%**——所以"SEG 是在一个小子集上算的"，报数时必须说明（[CTC 10 年报告，Nat Methods 2023，PMC10333123](https://pmc.ncbi.nlm.nih.gov/articles/PMC10333123/)，一手核实原文）。另：**tracking gold truth 是全覆盖的，只有大型胚胎数据集（CE/DRO/TRIC/TRIF）只覆盖选定的部分细胞**。
- **TRA**：`1 − min(AOGM, AOGM₀)/AOGM₀`，节点 + 边 + 语义 + 分裂一起算。
- **LNK**（CLB 新增）：`1 − min(AOGM-A, AOGM-A₀)/AOGM-A₀`，先把顶点集合同步（同步不罚分），只评"连接对不对"。**这个指标专治"我检测差所以我 TRA 低"的争辩**——它把检测质量剥离出去。
- **BIO**：生物启发指标均值，含
  - **CT**（Complete Tracks）：金标准整条轨迹被完整重建的比例；
  - **TF**（Track Fractions）：平均一条真实轨迹被连续正确重建的最长比例；
  - **BC(i)**（Branching Correctness）：容忍 i 帧的分裂事件检出率；
  - **CCA**（Cell Cycle Accuracy）：细胞周期长度重建准确度。

### 2.3 AOGM 六项与官方权重（一手核实）

官方评测器 `TRA_log.txt` 表头直接打印了权重：

| 符号 | 含义 | 官方罚分 |
| --- | --- | --- |
| **NS** | Splitting Operations（多余分裂） | **5** |
| **FN** | False Negative Vertices（漏检顶点） | **10** |
| **FP** | False Positive Vertices（多检顶点） | **1** |
| **ED** | Redundant Edges To Be Deleted（多余边） | **1** |
| **EA** | Edges To Be Added（缺失边） | **1.5** |
| **EC** | Edges with Wrong Semantics（语义错误边） | **1** |

**这组权重直接决定了优化方向**：漏一个细胞（FN，10 分）等于多报 10 个细胞；而漏一条边（EA，1.5）远贵于多一条边（ED，1）。**在密集胚胎序列上"宁可多连不可漏检"是有严格数学依据的**——本项目 C5.0e 的"孤立短轨迹过滤"负结果（TRA −0.053）正是被这条权重惩罚的教科书案例。

**权重从哪来、该怎么解读（Matula et al. 2015 原文，一手核实摘要）**：

- AOGM 的六个操作是 **split / delete / add 一个顶点** 与 **delete / add / alter-the-semantics 一条边**——正好一一对应 NS / FN / FP / ED / EA / EC。这个命名不是约定俗成，而是**图的六个基本编辑操作**。
- 设计动机原文：此前的两类追踪指标（"完整重建轨迹的比例"与"正确时序关系比例"）**既不惩罚多报轨迹、也不处理分裂事件**——AOGM 就是为了同时惩罚这两件事而设计的。
- 原文明确：权重是**可以按需调的选择**（"may especially help developers and analysts to tune their algorithms according to their needs"），作者只证明了"**在小范围改动权重时排名仍然稳健**"（"robustness and stability … against small changes in the choice of weights"）。
- **推论（写论文时要用）**：① 权重的物理含义是"**人工标注成本**"，不是物理量，所以 **ΔTRA = 0.001 是加权计数差，不能做物理解释**；② 因此**必须同时报 NS/FN/FP/ED/EA/EC 原始条数**，否则读者无法判断你的 TRA 变化来自哪类错误；③ 由于权重可调，**任何人换一组权重都可能得到不同排名**——这也是"噪声地板 + 原始计数"必须一起报的原因。

### 2.4 3D+Time 数据集规格（一手核实，取自官网数据集页）

| 数据集 | 对象 / 显微镜 | 体素尺寸 (µm) | 时间间隔 | train / test 体积 | GT 类型 |
| --- | --- | --- | --- | --- | --- |
| **Fluo-N3DH-CE** | 线虫胚胎核 / Zeiss LSM510 63×1.4 oil | **0.09 × 0.09 × 1.0**（11× 各向异性） | 1 min (1.5) | 3.1 / 1.7 GB | silver 分割 + gold 追踪；**仅 1 份人工标注** |
| Fluo-N3DH-CHO | CHO GFP-PCNA 核 | 0.202 × 0.202 × 1.0 | 9.5 min | 98 / 105 MB | silver 分割 + gold 追踪 |
| Fluo-N3DL-DRO | 果蝇胚胎核 / SIMView 光片 16×0.8 | 0.406 × 0.406 × 2.03 | 30 s | 5.8 / 5.9 GB | **只评神经系统细胞**（首帧标出） |
| Fluo-N3DL-TRIC | 甲虫胚胎（制图投影） | **NA**（投影后非均匀） | 1.5 min | 20.6 / 19.9 GB | **只评胚盘边缘谱系** |
| Fluo-N3DL-TRIF | 甲虫胚胎（多视角融合） | 0.38³ 各向同性 | 1.5 min | **320 / 467 GB** | 只评胚盘边缘谱系；含配准伪影 |
| Fluo-C3DH-A549 | GFP-actin A549 肺癌（Matrigel） | 0.126 × 0.126 × 1.0 | 2 min | 244 / 294 MB | 全量分割 |
| Fluo-C3DH-H157 | GFP-H157 肺癌 | 0.126 × 0.126 × 0.5 | 2 (1) min | 7.0 / 7.1 GB | 全量分割 |
| Fluo-C3DL-MDA231 | MDA231 乳腺癌（胶原） | 1.242 × 1.242 × 6.0（极粗） | 80 min | 182 / 179 MB | 全量分割 |
| **Fluo-N3DH-SIM+** | 仿真 HL60 核 | 0.125 × 0.125 × 0.200 | 29 min | 3.1 / 5.9 GB | **完美掩码**（适合做"分割无关"上界） |

**三个对本项目直接有用的推论**：

1. CE 的 z 间距是 xy 的 11 倍 → 任何体素单位的距离/EBT/平滑核都是错的（这正是仓库 R3 与 C5.0 修复的来源）。
2. `Fluo-N3DL-*` 只评部分细胞：**多检会被 CTB 记为 FP**，与 CSB"额外检测自动过滤、不罚分"完全相反。做跨数据集对比时必须先读这句话。
3. `Fluo-N3DH-SIM+` 给完美掩码 → 是**唯一能干净分离"追踪侧贡献"的数据集**，建议加入对比矩阵（它的 SEG 不与检测耦合）。

### 2.5 官方 3D 成绩表（2026-08-08 累计结果，一手核实·本机重算）

**Fluo-N3DH-CE（23 个同时有 TRA 与 SEG 的有效提交）**

| 方法 | TRA | SEG | DET | OPCTB |
| --- | --- | --- | --- | --- |
| AMOLF-NL | **0.9937** | 0.5725 | 0.9952 | 0.7831 |
| MPI-GE (CBG) (3) | 0.9868 | 0.4654 | 0.9900 | 0.7261 |
| JAN-US（linajea） | 0.9788 | 0.5993 | 0.9814 | **0.7891** |
| IGFL-FR | 0.9747 | 0.6313 | 0.9795 | 0.8030 |
| THU-CN (3) | 0.9744 | 0.7252 | 0.9817 | 0.8498 |
| **AC (9)** | 0.9689 | 0.7321 | 0.9765 | 0.8505 |
| CZB-US | 0.9671 | 0.7218 | 0.9719 | 0.8444 |
| KTH-SE (1) | 0.9452 | 0.6617 | 0.9594 | 0.8035 |
| …（第 10 名区间）→ | **0.90–0.93** | 0.63–0.66 | 0.93–0.96 | 0.77–0.80 |
| KIT-GE (3) | 0.9012 | 0.6421 | 0.9346 | 0.7717 |
| RWTH-GE (2) | 0.8966 | 0.5570 | 0.9139 | 0.7268 |
| … | | | | |
| **MU-CZ (2\*)** | 0.7817 | **0.7590** | 0.8413 | 0.7703 |

（`*` = 从泛化档转入的提交。**可复现性提示**：第 1 名 `AMOLF-NL` 的参与者页无方法描述、描述 PDF 返回 404（本次实测），`AC (9)` 为匿名提交——**这两条无法核实方法细节**；`THU-CN (3)` 参与者页有描述链接，本次未展开核实。）

**其余 3D 数据集的前 2–3 名（一手核实）**

| 数据集 | #1 | #2 | #3 |
| --- | --- | --- | --- |
| Fluo-N3DH-CHO | KTH-SE (1) TRA 0.9532 / SEG 0.8986 | KIT-GE (2) 0.9480 / 0.8708 | CUNI-CZ 0.9353 / 0.7549 |
| Fluo-N3DL-DRO | CZB-US 0.8019 / 0.6132 | JAN-US 0.7854 / 0.3971 | MPI-GE (CBG)(3) 0.7854 / 0.3932 |
| Fluo-N3DL-TRIC | MPI-GE (CBG)(3) 0.9517 / 0.6795 | KTH-SE (2) 0.9417 / 0.7914 | CZB-US 0.8540 / 0.6542 |
| Fluo-N3DL-TRIF | MPI-GE (CBG)(3) 0.9545 / 0.6540 | CZB-US 0.9357 / 0.7455 | RWTH-GE (3) 0.8857 / 0.6841 |
| Fluo-C3DH-A549 | MU-CZ (2\*) 1.0000 / 0.8758 | PURD-US (\*) 1.0000 / 0.8628 | KIT-GE (3) 1.0000 / 0.8492 |
| Fluo-C3DH-H157 | KTH-SE (1) 0.9872 / 0.8880 | KTH-SE (1\*) 0.9824 / 0.8919 | KIT-GE (3) 0.9802 / 0.8778 |
| Fluo-C3DL-MDA231 | KIT-GE (3) 0.8845 / 0.7096 | KTH-SE (1) 0.8821 / 0.6322 | LEID-NL 0.8798 / 0.6416 |
| Fluo-N3DH-SIM+ | BGU-IL (5) 0.9740 / 0.8203 | KIT-GE (3) 0.9724 / 0.7585 | LEID-NL 0.9673 / 0.6286 |

### 2.6 人类上界 vs 算法（一手核实，官方 Extras 表）

| 数据集 | 人类 SEG | 最好算法 SEG | 人类 TRA | 最好算法 TRA |
| --- | --- | --- | --- | --- |
| **Fluo-N3DH-CE** | **0.8437** | **0.7590** | **1.0000** | 0.9937 |
| Fluo-N3DH-CHO | 0.9037 | 0.9171 | 0.9851 | 0.9532 |
| Fluo-C3DH-A549 | 0.8587 | 0.8910 | 1.0000 | 1.0000 |
| Fluo-C3DH-H157 | 0.9245 | 0.8919 | 0.9912 | 0.9872 |
| Fluo-C3DL-MDA231 | 0.7416 | 0.7096 | 0.9353 | 0.8845 |

**读法**：CE 上"人类 TRA = 1.0"，说明**轨迹标注本身是自洽的、没有标注噪声天花板**——算法 0.9937 与人类的差距是真实差距；而 SEG 上人类 0.8437 vs 算法 0.759，**相差 0.085，是 CE 上唯一还开着的大口子**。

> ⚠ 口径提醒：CTC 官方明确说明 **CE / DRO / TRIC / TRIF 只有 1 份人工追踪标注**，因此这些数据集没有 inter-annotator variability（官方表里为 NA）。**CE 的 "TRA Human = 1.0" 应理解为"参考图自身一致"，而不是独立的人类重复测量**；引用这两个人类数字时请同时给出这一限制。

### 2.7 泛化档：这才是"诚实对比"的战场（一手核实）

CTC 自 2023-08 开放泛化档提交，**全球累计只有 5 个**，官方按全部 78 个测试用例汇总：

> **协议细节（一手核实，来自我在官方 Generalizability 工作簿里逐表核对）**：泛化档 = **13 个真实数据集 × 6 种规定的训练数据配置（GT / ST / GT+ST / allGT / allST / allGT+allST）× 2 个视频 = 78 个结果集**，全程同一套方法。
> **其中 3D 只有 5 个：Fluo-N3DH-CE、Fluo-N3DH-CHO、Fluo-C3DH-A549、Fluo-C3DH-H157、Fluo-C3DL-MDA231。**
> ⚠️ **注意：Fluo-N3DH-SIM+、Fluo-N3DL-DRO / TRIC / TRIF 都不在泛化档里**——所以"一套参数跨 3D 数据集"的官方可比范围就是上面这 5 个。

| 排名 | 方法 | SEG(78例) | TRA(78例) | **OP(78例)** | CT | TF | BC(i) | CCA |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | PURD-US | 0.7277 | 0.8549 | **0.7913** | 0.1526 | 0.6791 | 0.000 | 0.000 |
| 2 | MU-CZ (2) | 0.6491 | 0.8148 | 0.7320 | 0.1358 | 0.5696 | 0.0648 | 0.0583 |
| 3 | IGFL-FR | 0.5061 | 0.7685 | 0.6373 | 0.0673 | 0.5517 | 0.000 | 0.000 |
| 4 | KTH-SE (1) | 0.5303 | 0.7326 | 0.6315 | 0.1296 | 0.5521 | 0.2267 | 0.4811 |
| 5 | DREX-US | 0.5261 | 0.6984 | 0.6122 | 0.1060 | 0.5518 | 0.2644 | 0.3469 |

**这张表是本调研最有价值的发现**：单数据集能刷到 TRA 0.99，一套参数跑全部 13 个数据集时**最好只有 0.855**。如果你的模型能在一套参数下跨 3–5 个 3D 数据集，**就是直接对标 PURD-US，而不是对标 CE 上的 0.9937**——后者是过拟合到数据集的产物。

---

### 2.8 除 CTC 外，**值得加进对比矩阵的两个 3D 基准**

这是本轮调研最重要的"新目标"发现：**两个有公开数据 + 公开评测代码的 3D 核追踪基准**，正好补上 CTC 的两个缺口（分裂感知指标、宽松许可）。

#### (a) ★ Biohub – Cell Tracking During Development（Kaggle，CZ Biohub / Royer lab）

**这是 2026 年 3D 细胞追踪最大的新变化，而且不在 CTC 里。**

| 项 | 内容 |
| --- | --- |
| 任务 | 斑马鱼胚胎光片 3D+time：① 检测细胞 ② 跨帧关联 ③ **识别分裂**；输出 = 图（节点 `(t,z,y,x)`、边 = 关联、分裂 = 一节点两出边） |
| 数据 | OME-Zarr，每样本 `(100, 64, 256, 256)` uint16；**体素 1.625 × 0.40625 × 0.40625 µm（4.0× 各向异性）**；GT 为 **GEFF / tracksdata** 图格式，**稀疏标注**；**按胚胎划分 train/test**（防止记住某个胚胎） |
| 指标（据官方 repo README/metrics.md） | `score = adjusted_edge_jaccard + 0.1 × division_jaccard`，micro 平均。节点匹配 = **最优二部匹配**，最大质心距离 **7 µm**；**过检测惩罚** `adjusted_jaccard = max(0, jaccard·(1 − 0.1·(T_pred − T_true)/T_true))` → **分数可以合法地超过 1.0**；分裂用"祖→父→子→孙"局部窗口、容许 ±1 帧 |
| 官方 baseline | `TemporalUNet3D`（3D U-Net + 时序注意力）出检测图与逐体素特征 → 池化节点特征 → `SimpleNodeTransformer` 对全部 `(t, t+1)` 配对打分，端到端训练（**明确未训到收敛**） |
| 状态 | **已结束**（约 4,020 支队伍，公开 writeup 很多）；**数据 + 指标 + baseline 全部可复用**，但不能再提交 |
| 为什么对你极其重要 | ① 指标**显式含分裂项**——正好解决"CTC 口径奖励删分裂"的问题，是你分裂卖点的**最佳主指标**；② **稀疏 GT + 显式过检测惩罚**，天然惩罚"多报细胞"，与 AOGM 的 FP=1 不同；③ **按胚胎留出**，泛化声明干净；④ 你可以用官方 scorer 与 4,020 个公开解**直接同分比较**；⑤ 它源自 Zebrahub/DaXi，与 Ultrack 生态互通 |

**依据级别**：任务/数据/指标/基线细节来自**官方 repo 的 README 与 metrics.md（子代理核实）**；我本环境无法独立复核（`raw.githubusercontent.com` 取不到，GitHub 页面为 JS 渲染）——**付印前请自己打开 repo 复核指标公式**。竞赛页与 writeup 链接见附录。

#### (b) BlastoSPIM 1.0 / 2.0（小鼠植入前胚胎，光片）

| 项 | 内容 |
| --- | --- |
| 规模 | **573 + 80 = 653 张全标注 3D 图、18,336 个核实例**，31 + 25 个胚胎；其中一个胚胎有 **89 个连续时间点**标注 |
| 体素 | xy 0.208 µm / z 2.0 µm → **≈9.6× 各向异性**（介于 CE 的 11.1× 与 DRO 的 5× 之间） |
| GT | **全量 3D 实例掩码**（逐 z 切片人工勾轮廓）——与 CTC 的稀疏 marker 不同 |
| 已报指标 | 分割：**IoU≥0.5 的 F1 与 panoptic quality**；追踪：**谱系边 F1 = 0.996**（对比 TrackMate 0.971 已配准 / 0.646 未配准） |
| 为什么有用 | ① **许可最宽松**（BlastoSPIM 是 CC0 级开放；CTC 有"非 CTC 用途需官方许可"的限制，见 §6）；② 有**已发表的跨系统迁移结果**（肠道类器官、*Platynereis* 胚胎、转盘共聚焦），是现成可引用的域偏移实验模板；③ 9.6× 各向异性正好检验你的 µm 口径 |

**结论**：你的"对比矩阵"理想形态是 **CTC 3D（榜单合法性）+ BlastoSPIM（宽松许可 + 密掩码 + 跨系统迁移）+ Biohub Kaggle（分裂感知指标 + 4,020 个公开解）** 三角，而不是只挂 CTC。



> 标注「一手核实」= 本次直接读过官方页面/官方 xlsx/论文摘要；「据检索」= 二手检索，数值需付印前复核。

## 3. 论文级 SOTA 方法 + CTC 榜标签对照

### 3.1 CTC 榜标签 → 论文对照表（**本调研最实用的一张表**）

榜面上只有机构缩写，看不到论文名。逐个解析后（一手核实参与者页源码链接）：

| CTC 标签 | 方法 / 论文 | 出处 | 代码 |
| --- | --- | --- | --- |
| **CZB-US** | **Ultrack** | Nature Methods 2025, doi:10.1038/s41592-025-02778-0（一手核实摘要） | `royerlab/ultrack` |
| **EPFL-CH (\*)** | **Trackastra**（一手核实：参与者页作者 = Gallusser & Weigert，源码链接 = `weigertlab/trackastra`） | ECCV 2024, arXiv:2405.15700 | `weigertlab/trackastra`（有预训练权重） |
| **THU-CN (3)** | **CELLECT**（据检索：对比嵌入学习，Nature Methods 2025, doi:10.1038/s41592-025-02886-x） | Nat Methods 2025 | `zzz333za/CELLECT`（据检索） |
| **IGFL-FR** | **ELEPHANT**（据检索：3D 增量深度学习，eLife 2022;11:e69380） | eLife 2022 | `elephant-track/elephant-client` |
| **JAN-US** | **linajea** | Nature Biotechnology 2023（据检索） | `funkelab/linajea` |
| **KIT-GE (3)** | KIT 距离图 + watershed + 位移 + min-cost flow（Scherr/Löffler/Mikut） | CTC 提交 2021（据检索） | `git.scc.kit.edu/KIT-Sch-GE` |
| **KTH-SE** | Viterbi/网络流全局关联（Magnusson & Jaldén） | IEEE TMI 2015（据检索） | 经参与者页 |
| **PAST-FR (\*)** | 无监督光流 + Kalman 滤波 | CTC CLB 提交 | 经参与者页 |
| **AMOLF-NL / AC (9) / MPI-GE (CBG) (3)** | **无公开方法描述**（AMOLF-NL 描述 PDF 实测 404） | — | ❌ 不可复现 |
| **BGU-IL (5)** | 双任务 ConvLSTM-UNet（据检索），FLuo-N3DH-SIM+ TRA 第 1 | — | ❌ 软件"upon request" |

**用法**：论文里对标"有代码的方法"（Ultrack / Trackastra / CELLECT / ELEPHANT / linajea / KIT pipeline），把 AMOLF-NL / AC(9) / MPI-GE(CBG)(3) 只作为"榜上存在更高结果"的引用。

### 3.2 必须对打的 3D 方法（有能力 + 有代码）

| 方法 | 机制一句话 | 维度 | 范式 | 为什么必须对打 |
| --- | --- | --- | --- | --- |
| **Ultrack**（CZB-US） | 每帧建"分割假设层级"（超度量轮廓图），用 ILP 选互斥分割并跨时关联 | **3D+t** | 优化（无训练，模型无关） | DRO 第 1（TRA 0.8019）；事实标准工具；Zebrahub 轨迹由它生成 |
| **Trackastra**（EPFL-CH） | 检测 token 上的编解码 transformer，用 **parental softmax** 预测成对关联（允许一对多、禁止多对一） | 2D **与 3D** | 学习（需标注链接）+ 贪心/LAP/ILP | **CE 上 LNK 0.9714（CLB 第 2）**；卖点正是"免调参关联"，**与你的 OT+GNN 正面对撞** |
| **HOCT**（Higher-Order Cell Tracking Transformer） | **边中心** transformer：候选**链接**之间在 3D 线-线距离先验下互相注意力，parental softmax 扩展到 Δt>1 + 两遍 tracklet ILP | **3D** | 学习（**不用预训练图像编码器**，19 维手工边特征）+ ILP | **2026 最新的 3D 追踪 SOTA 主张**：CTC + 细菌分裂基准 SOTA；且它**专门论证了"图拓扑对 GNN 无用"**（见 §3.3） |
| **CELLECT**（THU-CN (3)） | 逐体素细胞中心嵌入的**对比学习** → 中心检测 + 同名/分裂/异名三分类 LAP | **3D** | 学习（**稀疏中心标注**） | CE 上 TRA 0.9744 / SEG 0.7252 / **CTB 第 2**；且是"稀疏标注"路线的代表 |
| **ELEPHANT**（IGFL-FR） | 两个 3D U-Net（核检测 + 光流关联），**增量深度学习**从少量标注训练 | **3D** | 学习（人在环） | CE TRA 0.9747 / SEG 0.6313；Parhyale 504 时间点、26 万核的实战规模 |
| **linajea**（JAN-US） | 稀疏点标注 → CNN 预测细胞指示峰 + **到前一帧的运动矢量** → 候选图 → ILP 谱系 | **3D** | 学习（稀疏）+ ILP | CE TRA 0.9788、DRO 0.7854 |
| **KIT pipeline**（KIT-GE (3)） | 距离图 CNN → watershed 实例化 → 位移估计 → min-cost flow 匹配 + 分割误差纠正 | **3D** | 学习（分割）+ 优化 | **它就是你 C5 计划"距离图 + 分水岭"路线的 CTC 验证版**；CE 0.9012 / SIM+ 0.9724 / MDA231 0.8845 |

### 3.3 ⚠️ 一条直接挑战你 GNN 模块的已发表结论（必读）

**HOCT 摘要原文（一手核实，arXiv:2607.11754, 2026-07-13, Bragantini / Theodoro / Royer, CZ Biohub）**：

> "…these and other existing methods overlook two structural obstacles in candidate tracking graphs: (i) cell divisions entangle distinct lineage paths in the node embedding space, and (ii) **edges sharing a node have near-random label agreement, so the candidate-graph topology carries no useful information for graph neural networks to aggregate.**"

即：**在细胞候选图上，共享节点的边之间标签一致性接近随机**（据检索的具体数字：调整同配性 ℋ_adj = 0.01 ± 0.04，仅 29% 的共点边共享标签）→ **GNN 的消息传递拿不到有用结构信号**。

**这对你的方法意味着什么（重要）**：

- **威胁**：你的 §2.0.1 核心模块是"GNN 边级分类"。审稿人（尤其读过 HOCT 的）会问："既然候选图拓扑对 GNN 无用，你的 GNN 增益从哪来？"
- **但你的数据其实在同一方向上**：你自己的 A2 测量显示**外观/几何先验 AUC 0.9931**（强判别力），而**结构项 FGW / 非平衡 / 运动先验在官方指标上全为负收益**——这正是"拓扑不携带标签信息"的独立证据。
- **应对（建议写进论文）**：① 明确论证你的 GNN 增益来自**边特征**（外观/运动/几何/尺寸），而不是图拓扑——这与 HOCT 的发现一致；② 把 HOCT 当作**独立佐证**引用（两条独立证据同向），而不是回避；③ 若再做结构类模块（FGW），先回答"它为什么违反了 HOCT 的发现"。
- **另一条可用的正面证据**：HOCT 报的细菌分裂基准显示，**通用视觉基础模型特征并不是免费午餐**（AOGM：HOCT 6.36 ± 1.35 最好；CoTracker3 20.81；Trackastra 原始贪心 29.0），所以"用轻量边特征 + 针对性设计"是站得住的技术路线。（数据出自 HOCT 正文表格，**据检索，我未读全文表，付印前需复核**。）

### 3.4 分割前端（你的 SEG 0.673 直接对标它们）

| 方法 | 出处 | 用途 |
| --- | --- | --- |
| **nnU-Net** | Isensee et al., Nature Methods 2021 | 你现在用的（Dice 0.9605）。它输出**语义**掩码、**不产出实例**——这是 50.4 vs 122.1 实例/帧缺口的根源。**没有任何 nnU-Net 变体本身就是追踪器**，必须配 linker（Trackastra / Ultrack / LAP / ILP） |
| **Cellpose-SAM（= Cellpose 4.0）** | bioRxiv 2025.04.28.651001（据检索） | SAM backbone + Cellpose 框架，2D 与 3D（经 Cellpose 生态），对通道置换/尺寸/噪声/模糊/**各向异性**鲁棒。**建议作为"实例前端"主对照** |
| **StarDist 3D** | 星凸多面体实例分割 | 3D 核实例化的另一条成熟路线 |
| **u-Segment3D** | Nature Methods 2025（据检索） | 把 2D 分割堆叠共识成 3D 实例，在密集/复杂细胞上可超过原生 3D |
| ⚠️ **更正** | **EmbedTrack 是 2D-only**（Löffler & Mikut, IEEE Access 2022），**不要把它当 3D 前端**；3D 能力在 KIT-GE (3) 那条 pipeline 里 | — |

### 3.5 端到端 / 2D 参照，以及**不要引用为追踪**的论文

- **值得提（2D，提供指标与范式）**：Cell-TRACTR（+ Cell-HOTA）、**ConstTrack**（MICCAI 2026，DETR + 邻域星座特征 + delayed-decision parent head，报 `TRA | HOTA | DetA | AssA | DivA`）、**MAMHT**（IEEE TMI 2025，mitosis-aware 多假设追踪）。
- **零样本基线（审稿人会要）**：**SAM4CellTracking**（arXiv 2509.09943，零样本 SAM2 + SAM-Med3D，2D/3D）、**Cellpose-SAM 掩码 + 你的 linker**。
- ❌ **不要当追踪引用**：CellOT / GENOT / stVCR / TracingFlow（单细胞基因组扰动与轨迹推断，不是显微谱系）、BigNeuron（3D 神经元重建）、FlyCellAtlas（转录组）、Ma et al. multimodality challenge（**分割**基准，非追踪）。

### 3.6 CLB 榜：你的"上界档"应该对标的官方数字（一手核实）

既然 §0 第 12 条建议上界档改报 LNK，这里给出可直接引用的官方 LNK/BIO 数字（`CellLinkingBenchmark.xlsx` 的 `Cell Linking Benchmark` 分表，per-dataset mean over 01+02）：

| 数据集 | #1 | #2 | #3 |
| --- | --- | --- | --- |
| **Fluo-N3DH-CE** | PAST-FR (\*) LNK **0.9822** / BIO 0.8619 | **EPFL-CH = Trackastra** LNK **0.9714** / BIO 0.7825 | RWTH-GE LNK 0.9620 / BIO 0.6949 |
| **Fluo-N3DH-SIM+** | SIAT-CN (\*) LNK 0.9997 / BIO 0.9936 | PAST-FR (\*) LNK 0.9993 / BIO 0.9841 | KTH-SE (\*) LNK 0.9963 / BIO 0.9309 |
| Fluo-N3DH-CHO | RWTH-GE LNK 0.9930 / BIO 0.8028 | EPFL-CH LNK 0.9928 / BIO 0.9548 | RWTH-GE (\*) LNK 0.9903 / BIO 0.9152 |
| Fluo-N2DL-HeLa | PAST-FR (\*) 0.9961 / 0.9510 | EPFL-CH 0.9953 / 0.9135 | SIAT-CN 0.9946 / 0.9051 |

**CLB 泛化档总榜（全 13 数据集）**：PAST-FR (\*) LNK **0.9840**（#1）/ BIO 0.8622（#1）/ OP 0.9231（#1）；**EPFL-CH = Trackastra** LNK **0.9774**（#2）/ BIO 0.7938（#2）/ OP 0.8856（#2）；RWTH-GE (\*) 0.9718（#3）；SIAT-CN (\*) 0.9655（#4）；KTH-SE (\*) 0.9576（#5）。

> ⚠️ **注意**：DRO / TRIC / TRIF **不在 CLB 榜上**（CLB 只在"准备好检测"的 13 个数据集上评）。所以"用 LNK 报上界档"目前**只有 CE / CHO / SIM+ 可用**，DRO 系仍只能用 TRA/SEG。

### 3.7 基准本身的论文（引用必用）

- Ulman et al., *An objective comparison of cell-tracking algorithms*, **Nature Methods 2017**（CTC 立论）
- Matula et al., *Cell tracking accuracy measurement based on comparison of acyclic oriented graphs*, **PLoS ONE 2015**（AOGM/DET/TRA 定义）
- Maška et al., *The Cell Tracking Challenge: 10 years of objective benchmarking*, **Nature Methods 2023**（含 CT/TF/BC/CCA、silver truth、泛化档设计）

---

## 4. 指标清单：CTC 之外还必须报什么

### 4.1 必报（CTC 官方，直接可得）
`DET / SEG / TRA / OPCTB` + **AOGM 六项分解（NS/FN/FP/ED/EA/EC）**。六项分解是把"我为什么输"讲清楚的唯一工具——本项目已用它定位出 EA 44% + NS 43% 的误差结构。

### 4.2 强烈建议补报（CTC 官方也有，但大家常忽略）
`CT / TF / BC(i) / CCA` + `LNK`。
- **CT/TF**：轨迹级完整度（比 TRA 更贴近生物学者关心的"谱系对不对"）；
- **BC(i)/CCA**：分裂与细胞周期——**你的方法有显式"体积守恒分裂判据"，不报分裂指标等于白做**；
- **LNK**：把检测误差剥离后的纯关联能力，**这是回应"你 TRA 低是因为前端差"的最有力证据**。

### 4.2b 指标演进：CHOTA 与 Cell-HOTA（2024–2025 新增，**建议必报**）

这是本次调研最重要的方法论更新，直接命中我们"TRA≈0.996 已经没有区分度"的处境。

| 指标 | 出处 | 做法 | 为什么必须报 |
| --- | --- | --- | --- |
| **CHOTA** | Kaiser, **Ulman（CTC 组织者）**, Rosenhahn, **ECCVW 2024 / BIC**, [arXiv:2408.11571](https://arxiv.org/abs/2408.11571)（一手核实摘要） | 把 HOTA 的"轨迹"**重定义为整条谱系（lineage clique）**，即一条轨迹连同其所有祖先/后代算作一个身份 | 论文明确指出当前指标"favor local correctness and weakly reward global coherence"；CHOTA 是**对全部错误类型（FP/FN/ID switch/漏匹配/漏分裂）都敏感**的指标。**代码在 CTC 官方的 [py-ctcmetrics](https://github.com/CellTrackingChallenge/py-ctcmetrics)** |
| **Cell-HOTA（含 DivA）** | O'Connor & Dunlop, PLoS Comput Biol 2025 | 标准 HOTA + **分裂准确度 DivA**（TPD 要求两个子代/父代同时匹配，允许 ±1 帧柔性），`AssDivA=√(AssA·DivA)` | 把分裂从"附带项"变成**一等项**；论文明确报告 **"removing division links can increase the tracking score"**，即 CTC 口径会**奖励删掉分裂** |
| **MOTA / IDF1 / MT-PT-ML** | CLEAR MOT / Ristani 2016 / Li 2009 | 经典 MOT 家族 | 跨领域审稿人会用；但 **MOTA 是检测主导且重罚 FP，与 AOGM（FN 罚 10 倍于 FP）取向相反**，报的时候要说明 |

> **对我们的直接含义**：我们方法的卖点之一就是"体积守恒分裂判据"。**只报 TRA 的话，删掉分裂可能反而涨分**（Cell-TRACTR 的实测结论），即指标方向与我们的机制相反。**必须补 CHOTA + BC(i)（或 Cell-HOTA DivA），并在论文里显式指出这个"奖励删分裂"的已知缺陷。**

### 4.2c HOTA 类指标的聚合陷阱（审稿人会抓）

- HOTA/AssA 跨序列聚合时用**按 TP 数加权的平均**，不是简单平均（`combine_sequences`，TrackEval）。**"per-sequence HOTA 的平均" ≠ "池化后的 HOTA"**，两者是不同的量。
- 因此必须写清：**聚合是 micro/pooled（先合并 TP/FN/FP 再算）还是 macro/mean（先各序列算再平均）**；主表给 pooled，附表给 per-sequence（我们的双序列表结构本来就是对的，只需补一句聚合口径）。


### 4.3 通用 MOT 指标（跨领域审稿人会问）
`MOTA`（及其 FP/FN/IDSW 分解）、`IDF1 / IDP / IDR`、`ID switches`、`MT/ML`（mostly tracked/lost）、`Fragmentations`、`HOTA`（含 DetA/AssA 分解）、以及 **Cell-HOTA**（加分裂维度）。
注意：**MOTA 在细胞追踪上会惩罚"多检"，与 AOGM 的 FP=1 保守取向相反**，报的时候要说明口径。

### 4.4 统计与成本（2026 年审稿的硬要求）
- **多种子**（≥3）均值 ± 波动；**噪声地板**（你的仓库已定为 0.0007）——所有 Δ 都要除以它标注倍数；
- **配对检验**：逐序列/逐视频的配对 Wilcoxon signed-rank，或对视频做 bootstrap CI（不要把同一视频的帧当独立样本！）；
- **运行时间 / 峰值显存 / 参数量 / 吞吐（cell·frame/s）**，并注明 CPU/GPU 型号；
- **失败模式可视化**：过分割/欠分割/ID switch 各给一例。

### 4.5 还有哪些"CTC 之外"的基准与工具值得知道

| 名称 | 是什么 | 对我们的用处 |
| --- | --- | --- |
| **CTMC**（CVPRW 2020，[IEEE 9150652](https://ieeexplore.ieee.org/document/9150652/)） | 唯一真正独立的细胞追踪榜：14 种细胞、~2900 轨迹、15.2 万帧，**框标注**，主要按 MOTA 排名 | **负面参考**：CHOTA 的注入实验显示 CTMC 类指标对"分裂错误"的相关性为 **0** → 谱系类论文不能拿它当主指标 |
| **DeepCell / DynamicNuclearNet Tracking** | 大规模**全量标注**的 2D 时序核数据 | 回应"CTC 的 SEG 真值只有 17.8% 覆盖"的最干净方案（reviewer 大概率会建议） |
| **WormID** | 分辨率鲁棒性基准（1×0.16×0.16 → 1.5×0.32×0.32 µm/voxel），Ultrack 用它验证 | 你要做"鲁棒性"章时的现成参照设计 |
| **Metrics Reloaded**（Nat Methods 2024） | 按"问题指纹"给指标选择的权威指南（实例分割推荐 PQ） | 论文指标选择段落的可引用依据 |
| **Pareto 式退化曲线** | Ultrack 用 Δt 扫描（Δt=10 TRA 0.443→0.623；Δt=1 0.927→0.929）报曲线而非单点 | 直接对应你的 C6 合成退化设计：**报曲线比报单点更有说服力** |
| **双通道稀疏标记建 GT** | Ultrack 的 Nature Methods 方法：荧光全标 + 另一波长稀疏随机标记，可人工校对出人工做不到的长时间 GT | 回应"你的 GT 是部分/有歧义"的最有力引用（如果有条件做） |

---

## 5. 你的模型该怎么比（核心章节）

### 5.1 先满足"可比性三要素"（不满足，数字一律不可比）

| 要素 | 你的现状 | 要做什么 |
| --- | --- | --- |
| ① **检测来源一致** | 有 GT-marker 上界档 / 预测实例档 | 报告里**每个数字必须标档**；与榜单比只能用"预测实例档" |
| ② **评测真值一致** | 用官方二进制的 DET/SEG/TRA | 保持。但你现在评的是 **CTC 训练序列**，榜单是**隐藏测试序列** → 想上表必须**提交 CTC** |
| ③ **参数选择协议一致** | seq01 标定、seq02 留出 | 必须写清"哪些序列参与了调参"；否则泛化声明无效（仓库 0003 决策已量化过：seq02 有 79% 帧被 nnU-Net 见过） |

### 5.2 十三个比较维度（建议直接做成论文表格的列）

| # | 维度 | 具体指标/做法 | 为什么必须有 |
| --- | --- | --- | --- |
| 1 | **检测来源三档** | GT marker / 预测实例 / 合成退化 | 隔离"追踪贡献"与"前端贡献" |
| 2 | **前端 vs 追踪贡献分解** | Oracle 实例上界（你已测：TRA 0.9848） | 一句话说清上限在哪 |
| 3 | **TRA + AOGM 六项** | NS/FN/FP/ED/EA/EC 条数 | 误差归属 |
| 4 | **SEG / DET / OPCTB** | 官方二进制 | 官方排序口径 |
| 5 | **DS 谱系指标** | CT/TF/BC(i)/CCA | 生物学意义 + 分裂卖点 |
| 6 | **LNK（上界档的正规口径）** | CLB 口径：在准备好的检测上只评连接 | **GT-marker 档应报 LNK 而不是 TRA**——把检测从等式中移除，回应"你 TRA 低是因为前端差" |
| 7 | **CHOTA + Cell-HOTA(DivA)** | 谱系级 HOTA / 分裂准确度 | 反制"TRA 饱和 + 奖励删分裂"（§4.2b） |
| 8 | **同源可复现基线** | 匈牙利 / 最近邻 / 纯 OT（无 GNN）/ Trackastra / Ultrack | **必须自己跑**，不能抄数字 |
| 9 | **跨序列泛化** | seq01 训练 → seq02 零调参（你已有：0.8991） | 最基础的泛化证据 |
| 10 | **跨数据集泛化** | 一套参数跑 **CE + CHO + A549 + H157 + MDA231**（= CTC 泛化档的 5 个 3D 成员） | 对标 CTC 泛化档（PURD-US 全库 TRA 0.8549）；⚠️ **SIM+ / DRO / TRIC / TRIF 不在泛化档里**，别拿它们当"官方泛化"证据 |
| 11 | **鲁棒性** | 合成退化：FN/FP/过分割/欠分割 4 档 × TRA 曲线 | 审稿人必问；也证明"不依赖完美检测" |
| 12 | **统计显著性** | ≥3 种子 + 双序列 + 噪声地板倍数 + 配对检验 + 序列级 bootstrap CI | 你的仓库已有这套纪律，务必写进论文 |
| 13 | **计算成本** | 秒/帧、体素/秒、峰值显存、GPU 型号、是否含 I/O | 3D 方法没有成本表会被质疑可用性 |

### 5.3 必须跑的对照（按性价比排序）

1. **匈牙利/最近邻 + 你的前端**（最弱基线，证明 OT+GNN 的必要性）；
2. **纯 OT（无 GNN）**——你已测 0.8717（旧口径），必须在新口径下重跑；
3. **Trackastra**（= 榜上 EPFL-CH，有预训练权重）：**最关键的一条**，它的卖点正是"免调参 SOTA 关联"，且它在 **CE 的官方 LNK 0.9714**——**用 LNK 和你自己的上界档直接对撞**，这是最干净的关联能力比较；
4. **HOCT**（2026，边中心 transformer，有代码）：新的 3D SOTA 主张，且它明确论证了"候选图拓扑对 GNN 无用"——**必须对打，也必须在论文里正面回应**；
5. **CELLECT**（THU-CN (3)）与 **ELEPHANT**（IGFL-FR）：CE 上 SEG 表现好的两个（0.7252 / 0.6313），是"实例质量"维度的对手；
6. **KIT pipeline / Ultrack**：KIT 的"距离图 + watershed"是你 C5 路线的验证版；Ultrack 代表大规模光片方向（DRO 第 1）；
7. **你的方法 − 各模块**（消融矩阵，你已有 25 个开关）；
8. **你的方法 + Cellpose-SAM 前端**替代 nnU-Net：专治"实例归属"瓶颈，预计收益最大；
9. **同检测下的关联替换实验**：把 Trackastra / HOCT / 匈牙利 / 纯 OT 接到**完全相同的前端检测**上，只在关联步骤替换——这是唯一能把"我的关联更好"讲干净的对照，也是回应 §3.3 质疑的必需实验；
10. **零样本基线**：SAM4CellTracking（或 Cellpose-SAM 掩码 + 简单 linker）——2025/26 审稿人会要一个"不训练"的参照。

> **工具链建议（可直接落地，不用自己实现指标）**：`pip install py-ctcmetrics` 可一次拿到 **CTC DET/SEG/TRA + AOGM 六项 + MOTChallenge + HOTA + CHOTA + MT/ML + `ctc_noise` 噪声注入**（[CellTrackingChallenge/py-ctcmetrics](https://github.com/CellTrackingChallenge/py-ctcmetrics)）；[traccuracy](https://traccuracy.readthedocs.io/en/stable/metrics/ctc.html) 提供 CT/TF/Division F1/CHOTA 并明确警告"GT 非稠密时慎用 FP 类指标"；3D 实例质量用 [panoptica](https://github.com/BrainLesion/panoptica)（PQ/SQ/RQ，支持从语义图连通域近似实例——正好对上你的 nnU-Net 语义→实例瓶颈）。

### 5.4 建议的论文主表模板（双序列 × 三档 × 三指标）

| 检测来源 | 方法 | CE seq01 DET | SEG | **TRA** | OPCTB | CT | BC(i) | 运行时长 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GT marker（上界档，不可与榜单比） | 匈牙利 | 1.000 | — | 0.9956 | — | — | — | — |
| | 纯 OT | 1.000 | — | ~0.995 | — | — | — | — |
| | **本方法** | 1.000 | — | **0.9965** | — | — | — | — |
| 预测实例（端到端档，与榜单可比） | 匈牙利 | — | — | — | — | — | — | — |
| | Trackastra | — | — | — | — | — | — | — |
| | **本方法** | 0.9388 | 0.6732 | **0.8982** | 0.7857 | — | — | — |
| | **本方法 + 实例前端改进** | 0.9558 | 0.6636 | **0.9269** | 0.7952 | — | — | — |
| 合成退化（鲁棒性） | 4 档退化 × 本方法 | — | — | 曲线 | — | — | — | — |

### 5.5 把你自己放进榜单的正确姿势（**不要错过这个**）

你现在 CE seq01 的数字已经足够投榜，而且有两个非常强的故事：

- **GT-marker 上界档 TRA 0.9965 > 榜面第 1 名 0.9937** → 作为"关联算法本身处于第一梯队"的**诊断信号**（但正式报告应用 LNK 口径，与 §3.6 的 Trackastra 0.9714 / PAST-FR 0.9822 对比，避免跨档比 TRA）；
- **端到端档 TRA 0.9269 → 按 TRA 排 CE 第 10 / 23；OP 0.7952 → 按 OPCTB 排第 9 / 23**；且 **SEG 0.6636 可排到 CE 第 8**（已高于 JAN-US 的 0.5993、IGFL-FR 的 0.6313）；
- 若把前端换成实例感知（Distance/Boundary Head，仓库 C5 计划的 +0.079 空间兑现一半），**OP 有望进入 CE 前 5**。

**动作建议**：注册 CTC → 提交 CE（先 regular）→ 同时评估是否可以走 generalizable（需要一套参数跑全部 13 个数据集，门槛高但竞争少、仅 5 个提交）。这是把"我跟 SOTA 比过"从说法变成事实的唯一途径。

---

### 5.6 推荐的"三基准对比矩阵"（可直接当实验章骨架）

1. **CTC 3D（榜单合法性）**：**CE + CHO**（regular；若有资源做泛化档则加 A549 / H157 / MDA231）。报 `DET / SEG / TRA / OPCTB` + **AOGM 六项**；**上界档另报 LNK**（对标 PAST-FR 0.9822 / Trackastra 0.9714）。
2. **Biohub Kaggle（分裂感知 + 大规模公开可比）**：用官方 scorer 报 `adjusted_edge_jaccard / division_jaccard / 总分`——**这是你分裂卖点的主指标**，也是唯一能与 4,020 个公开解直接同分的口径。
3. **BlastoSPIM（宽松许可 + 跨系统迁移）**：报 IoU≥0.5 的 F1 / panoptic quality / **谱系边 F1**（对标已发表的 **0.996**）；并可延伸它的"跨系统迁移"实验设计。
4. **压力测试（审稿人最爱）**：把**唯一各向同性的 TRIF（1.0×）重采样到 CE 的 11.1×**，报退化曲线；再做 Δt 下采样扫描（20 s → 80 min 跨度 ~240×）。
5. **零样本臂（可选但建议）**：Cellpose-SAM 掩码 + 简单 linker，或 SAM4CellTracking。
6. **统计与发表**：全部 ≥3 种子、双序列、噪声地板倍数、序列级 bootstrap CI；**公开权重/派生数据前先解决 §0 第 16 条的许可问题**。

---

## 6. 陷阱清单（都是本仓库已经踩过或差一步踩到的）

| 陷阱 | 具体表现 | 规避 |
| --- | --- | --- |
| **TRA 饱和幻觉** | CE 前 6 名 TRA ≥ 0.967，Δ0.005 看起来像进步 | 用 AOGM 六项 + OPCTB + CT/BC 看结构变化；Δ < 噪声地板一律写"不可区分" |
| **拿 GT-marker 档比榜单** | 榜单方法必须自己检测 | 报告必须标档；论文里把上界档明确写成"隔离追踪贡献的诊断档" |
| **训练序列 vs 隐藏测试序列** | 你自评的数字与榜单不是同一份数据 | 提交 CTC；或在论文里显式声明"自评于训练序列" |
| **CE 的 SEG 是 silver truth** | SEG 上限 0.84（人类），不是 1.0 | 报 SEG 时同时给人类上界；别用"SEG 只到 0.67"自责 |
| **各向异性 11× 未按物理单位** | 体素距离 ≈ 把 z 当成和 xy 一样细 | 所有距离/EDT/门限按 µm（仓库 R3、C5.0） |
| **`Fluo-N3DL-*` 只评部分细胞** | 多检 → CTB 记 FP 重罚；CSB 不罚 | 跨数据集对比前先读官方"Important note" |
| **匿名榜面的不可复现性** | CE 第 1 名 AMOLF-NL 无公开方法描述、PDF 404 | 论文里对标"有代码的方法"（Trackastra/Ultrack/linajea），匿名条目只作为"存在更高结果"的引用 |
| **MOTA 与 AOGM 取向相反** | MOTA 重罚 FP，AOGM 重罚 FN(10) | 两个都报，并解释取向差异 |
| **把帧当独立样本做检验** | 同一视频的帧高度相关 | 按视频/序列做配对检验或 bootstrap |
| **泛化声明越界** | seq02 有 79% 帧被前端见过 | 用 0003 决策的口径：区分"参数可迁移"与"跨序列泛化" |
| **指标混用** | CTC TRA（AOGM + marker GT）/ Biohub（GEFF 边 Jaccard + 分裂项）/ BlastoSPIM（谱系边 F1）/ Cell-HOTA **不可比** | 每行明确标指标；或在统一 GT 上重算同一指标 |
| **忽略 CTC 的"额外细胞不对称"** | DRO/TRIC/TRIF：多检在 CTB 里**算错**、在 CSB 里**被静默过滤** | 不要用 CSB 视角调参再声称 CTB 成绩 |
| **各向异性不做重采样就谈泛化** | CE **11.1×** vs TRIF **1.0×**（全 CTC 唯一各向同性）；Δt 从 20 s 跨到 80 min（~240×） | 门限一律按 µm；报告重采样策略；可做"把 TRIF 重采样到 CE 各向异性"的压力测试 |
| **许可越界（新发现）** | CTC 条款：公开非 CTC 用途需官方许可、禁止克隆数据集与标注 | 公开权重 / 派生数据前向 CTC 组委会书面确认 |
| **引用"CTC 2025 届"** | CTC 自 2017 起就是**每月滚动**评测，最后一届固定截稿是 **ISBI 2024**（引入 CLB），2025/2026 没有新届、没有新 3D 数据集 | 引用时按"持续开放基准"表述 |
| **把静态任务当追踪** | cryo-ET Kaggle = 静态 3D 目标识别；FlyWire / MICrONS = 静态体数据的划分与人工校对 | 都不含时间轴，不能作为时序追踪基准引用 |

---

## 附录 A：一手数据的来源与复核方法

**来源 URL**

- 评测方法（DET/SEG/TRA/LNK/BIO/OP 定义）：<https://celltrackingchallenge.net/evaluation-methodology/>
- 3D+Time 数据集规格：<https://celltrackingchallenge.net/3d-datasets/>
- 榜单页（含历史快照）：<https://celltrackingchallenge.net/latest-ctb-results/>
- 官方数据文件：
  - `http://public.celltrackingchallenge.net/documents/CellTrackingBenchmark.xlsx`
  - `http://public.celltrackingchallenge.net/documents/CellTrackingBenchmark-Generalizability.xlsx`
  - `http://public.celltrackingchallenge.net/documents/CellSegmentationBenchmark.xlsx`
  - `http://public.celltrackingchallenge.net/documents/CellLinkingBenchmark.xlsx`
- 参与者与算法描述：<https://celltrackingchallenge.net/participants/>
- **指标演进与批评（一手核实摘要）**：
  - CHOTA（Kaiser, **Ulman**, Rosenhahn, ECCV 2024 BIC workshop）：<https://arxiv.org/abs/2408.11571>
  - Cell-TRACTR + Cell-HOTA（PLoS Comput Biol 2025）：<https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1013071>
  - HOTA（Luiten et al., IJCV 2020）：<https://arxiv.org/abs/2009.07736>，参考实现 <https://github.com/JonathonLuiten/TrackEval>
  - AOGM 原始定义（Matula et al., PLoS ONE 2015）：<https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0144959>
  - CTC 10 年报告（含 17.8%/99.1% 覆盖、泛化与复用研究）：<https://pmc.ncbi.nlm.nih.gov/articles/PMC10333123/>
  - 官方/官方附属 Python 工具链：[py-ctcmetrics](https://github.com/CellTrackingChallenge/py-ctcmetrics)（CTC+AOGM 六项+HOTA+CHOTA+`ctc_noise`）、[traccuracy](https://traccuracy.readthedocs.io/en/stable/metrics/ctc.html)、[panoptica](https://github.com/BrainLesion/panoptica)（3D PQ）
  - 指标选择权威指南：Metrics Reloaded，Nat Methods 2024，<https://www.nature.com/articles/s41592-023-02151-z>
- 方法类：Trackastra <https://arxiv.org/abs/2405.15700>（CTC 标签 EPFL-CH，参与者页：<https://celltrackingchallenge.net/participants/EPFL-CH/>）；Ultrack <https://www.nature.com/articles/s41592-025-02778-0>；Ultrack 的 CTC 提交 <https://github.com/royerlab/ultrack_CTC_submission>；linajea <https://github.com/funkelab/linajea>；
  **HOCT**（2026，边中心 transformer）<https://arxiv.org/abs/2607.11754>；
  **CELLECT**（THU-CN (3)）<https://www.nature.com/articles/s41592-025-02886-x>；
  **ELEPHANT**（IGFL-FR）<https://elifesciences.org/articles/69380>；
  **Cellpose-SAM / Cellpose 4.0** <https://www.biorxiv.org/content/10.1101/2025.04.28.651001v1>；
  **SAM4CellTracking**（零样本基线）<https://arxiv.org/abs/2509.09943>
- CLB（连接榜）数据文件：`http://public.celltrackingchallenge.net/documents/CellLinkingBenchmark.xlsx`；页面 <https://celltrackingchallenge.net/latest-clb-results/>
- **CTC 提交与许可**：提交要求 <https://celltrackingchallenge.net/submission-of-results/>；数据使用条款（"公开非 CTC 用途需官方许可、禁止克隆"）<https://celltrackingchallenge.net/datasets/>；标签改名映射 `TagMapping.xlsx`（如 `AC (6)` → CZB-US、`AC (8)` → LUH-GE）
- **§2.8 的两个额外 3D 基准**：
  - Biohub – Cell Tracking During Development（Kaggle，已结束）：<https://www.kaggle.com/c/biohub-cell-tracking-during-development>；官方数据/指标/基线 repo：<https://github.com/royerlab/kaggle-cell-tracking-competition>（README + `metrics.md`）
  - BlastoSPIM 数据：<https://users.flatironinstitute.org/~awatters/blastospim/html/series.html>；论文 Nunley et al., Development 2024：<https://pmc.ncbi.nlm.nih.gov/articles/PMC11574361/>
  - Zebrahub（Ultrack 生成轨迹）：<https://zebrahub.sf.czbiohub.org/imaging>
- **体内成像（域偏移可选臂）**：LTDB 白细胞追踪库（Sci Data 2018）<https://pmc.ncbi.nlm.nih.gov/articles/PMC6049032/>（含 11 类命名失败模式，可作鲁棒性章节的词表）

**复核方法（可重跑）**

1. 下载上述 4 个 xlsx；
2. 按**方法名**（不是行号——各 workbook 的行集合不同，按行号对齐会串行取错）从 `TRA`/`SEG`/`DET` 三个分表取视频 01/02 的值并求均值；
3. 与 `Cell Tracking Benchmark` / `Cell Segmentation Benchmark` 分表的"per dataset mean"交叉校验；
4. AOGM 权重直接从官方评测器输出日志的表头读取（本项目 `experiments/*/official_logs/TRA_log.txt` 前 6 行）。

**本项目自身数字的出处**

| 数字 | 出处 |
| --- | --- |
| GT-marker 档 TRA 0.996513 / 0.996369 | `experiments/B1_gnn_paper_ce01/`、`experiments/B3_seq02_validation/` |
| Oracle 实例上界 TRA 0.984784 / DET 0.991581 / SEG 0.688720 | `experiments/C4O_oracle_upper_bound/` |
| 真实检测 + 再切 k=1.6：seq01 TRA 0.926890 / seq02 0.926320 | `experiments/C5.0e_postproc_rules/`、`experiments/C5.0f_resplit_sweep/` |
| paperpipe 严格口径双序列：seq01 0.898249 / seq02 0.899119，SEG 0.6732 / DET 0.9388 | `experiments/P4_final_gnn_ce01/`、`experiments/P4_final_gnn_ce02/` |
| 噪声地板 0.0007 | `experiments/B2_methodology/` |

**本次未完成的核对项（留白，避免误用）**

1. AMOLF-NL / AC (9) / MPI-GE (CBG) (3) 的方法细节：官方参与者页无描述、PDF 实测 404 → **无法核实**（这是榜面最大的不可复现风险）；BGU-IL (5) 软件 "upon request"；
2. **Trackastra 在 CE 上只提交了 linking-only（CLB, EPFL-CH）**——所以 CTB 榜上**没有它的 TRA/SEG**；其论文中的 3D 支持（nD masks、3D RoPE）与逐数据集数字**据检索，未读全文**，付印前需复核；
3. Ultrack 的具体 TRA 数字（Nature 站点跨域跳转被拦）→ 本文引用的 Ultrack 细节来自其 PMC 镜像；
4. linajea / CELLECT / ELEPHANT 的出处与 CE 数字：linajea 的 CE 数字是榜面（一手），论文出处（Nat Biotech 2023）为**据检索**；CELLECT（Nat Methods 2025）与 ELEPHANT（eLife 2022）的**论文→CTC 标签映射为据检索**，机制描述需读原文确认；
5. HOCT：摘要与作者/日期为**一手核实**；正文的逐数据集 CLB/LNK/BIO 数字、细菌分裂基准表、`royerlab/hoct` 仓库与权重均为**据检索，未逐一打开核实**；
6. Cellpose-SAM 的 3D 能力与版本号（Cellpose 4.0 / bioRxiv 2025.04.28.651001）为**据检索**；
7. CTC 官方 `py-ctcmetrics` 与官方二进制在本机内核上的兼容性 → **未实测**（本项目已知官方静态二进制与本地新内核不兼容，需在云端跑）；
8. CHOTA / Cell-HOTA 在 **3D 各向异性** 数据上的行为（两者原文都只在 2D 数据上验证）→ 用作辅助指标、不作主指标前建议先在自己数据上做一致性检查；
9. **可微/神经 OT 细胞追踪器"不存在"**这一结论来自检索未命中（negative result）——不能证明绝对不存在，只能说明**没有成为公开 SOTA**；撰写新颖性声明时措辞要保守。
10. **Biohub Kaggle 的指标公式与 baseline 细节**：来自官方 repo 的 README/`metrics.md`（**子代理核实**）；我本环境**未能独立复核**（`raw.githubusercontent.com` 取不到、GitHub 页面 JS 渲染）→ 付印前请自行打开 repo 核对 `score = adjusted_edge_jaccard + 0.1×division_jaccard`、7 µm 匹配阈值与过检测惩罚项；
11. BlastoSPIM 的**下载脚本/固定划分**与 CTC 的许可条款原文：均为二手转述（URL 已给），使用前请自行打开确认。

**已复核的关键事实（本轮逐一验证）**

| 事实 | 验证方式 |
| --- | --- |
| AOGM 权重 NS=5 / FN=10 / FP=1 / ED=1 / EA=1.5 / EC=1 | 项目自身官方评测日志表头（一手）+ CHOTA/traccuracy 文档（交叉印证） |
| AOGM 的六操作（split/delete/add 顶点 + delete/add/alter 边）与"权重可调、小改动下排名稳健"的设计说明 | Matula et al. 2015 PLoS ONE 摘要原文（一手） |
| CE 只有 1 份人工追踪标注、inter-annotator 为 NA | 官方 CTB 页面原文（一手） |
| gold 分割真值平均覆盖 17.8%、silver 99.1%、tracking gold 除大型胚胎外全覆盖 | CTC 10 年报告 PMC10333123 原文（一手） |
| CHOTA 为谱系级 HOTA、代码在 CTC 官方 py-ctcmetrics、ECCV 2024 BIC workshop | arXiv:2408.11571 摘要（一手） |
| **EPFL-CH 就是 Trackastra** | 官方参与者页（作者 Gallusser & Weigert + 源码 `weigertlab/trackastra`）（一手） |
| CLB 泛化总榜：PAST-FR LNK 0.9840 (#1)、EPFL-CH/Trackastra 0.9774 (#2) | 官方 `CellLinkingBenchmark.xlsx` Generalizability 分表（一手） |
| CE 的官方 LNK/BIO：PAST-FR 0.9822/0.8619、Trackastra 0.9714/0.7825、RWTH-GE 0.9620/0.6949 | 官方 `CellLinkingBenchmark.xlsx`（一手） |
| HOCT 存在，且摘要明确称"候选图拓扑对 GNN 不携带有用信息" | arXiv:2607.11754 摘要（一手） |
| **泛化档 = 13 数据集 × 6 种训练配置（GT/ST/GT+ST/allGT/allST/allGT+allST）× 2 视频 = 78 例；3D 成员仅 CE / CHO / A549 / H157 / MDA231**（SIM+ 与 N3DL 系列**不在**泛化档） | 本机读官方 Generalizability 工作簿的表头与数据集清单（一手） |
| 9 个 3D+Time 数据集体素尺寸/时间间隔/GT 类型 | 官方 3D 数据集页（一手） |
| CE/CHO/DRO/TRIC/TRIF/A549/H157/MDA231/SIM+ 的榜面 TRA/SEG/DET | 官方 xlsx 本机重算（一手） |
| 泛化档 5 个方法的 78 例聚合指标 | 官方 Generalizability xlsx 的 Ranking 分表（一手） |
| 人类 SEG/TRA 上界 | 官方 Generalizability xlsx 的 Extras 分表（一手，含 CE 单标注的解读限制） |
