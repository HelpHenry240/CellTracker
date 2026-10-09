# 细胞核分割 + 细胞追踪（时序关联/谱系）公开数据集调研（CTC 之外）

调研方式：web_search / web_fetch + curl 直取官网页面与 API（Zenodo / figshare / Dryad / Mendeley / IDR / Kaggle baseline repo）。
规则：**只收录本人实际检索到并能给出 URL 的条目**；凡未在页面/API 中直接证实的字段一律标注「未核实」。
本次调研日期：2026-10-08。

> 注意：本调研**排除 Cell Tracking Challenge（celltrackingchallenge.net）**本身；仅在必要处提及"派生自 CTC"的条目（如 BBBC035）。

---

## 1. 总表（带时序 + 轨迹/谱系标注 —— 可直接进入追踪训练）

| # | 名称 / 官方链接 | 维度·模态·规模 | 轨迹/谱系 + 分割掩码（稀疏/全量） | 格式 | 许可 / 注册 | 对接管线成本 | 量级 |
|---|---|---|---|---|---|---|---|
| T1 | **DynamicNuclearNet (DNN)** — Tracking + Segmentation 两个子集<br>https://deepcell.readthedocs.io/en/master/data-gallery/dynamicnuclearnet.html | 2D 荧光核，时序（具体帧数/图像数未核实） | **有**：`lineages` 记录 cell id、出现的帧、父→子分裂链接；同时给核分割掩码 y。标注为全量（每个 batch 的 y 与 lineage 对应） | Python API 返回数组（X/y/lineage 对象），需自行导出 TIFF+zarr | 修改版 Apache，**仅非商业学术**；需 API key（https://users.deepcell.org/login/ ） | **低–中**：有现成 lineage → 可直接转成 (帧, 节点, 边, 分裂) 图标签；输出不是 CTC 格式，需写导出脚本 | 未核实 |
| T2 | **BlastoSPIM（= 小分析提到的 MINS / preimplantation mouse embryo）**<br>https://plus.figshare.com/articles/dataset/Dataset_supporting_Nuclear_instance_segmentation_and_tracking_for_preimplantation_mouse_embryos_/26540593<br>http://blastospim.flatironinstitute.org/ | **3D** 光片（SPIM），H2B-miRFP720；BlastoSPIM 1.0 = 573 张 3D 图（11,708 核 + 116 极体），2.0 = 80 张（6,628 核），合计 18,336 核；xy 0.208 µm / z 2.0 µm；早期胚胎序列 15 min/帧、约 29h45m | **分割掩码：全量 3D 实例掩码**（逐 z 切片勾轮廓）。**轨迹/谱系：数据集页面只描述分割**（标题含 tracking）→ **时序轨迹标注未核实** | `*_image_0001.npy.gz` + `*_masks_0001.npy.gz`（每序列一对） | **CC0**；无需注册（figshare/官网直接下载） | **中**：npy.gz → TIFF 需转换；若把同一胚胎按时间序串起来可做"检测+关联"预训练；分裂标签需自行构造或找配套论文补充 | **23.16 GB**（figshare API 实测 size=23156220925） |
| T3 | **Biohub – Cell Tracking During Development（Kaggle 竞赛）**<br>https://www.kaggle.com/competitions/biohub-cell-tracking-during-development<br>官方 baseline：https://github.com/royerlab/kaggle-cell-tracking-competition | 斑马鱼胚胎 **3D+time 光片**；每个样本 (100, 64, 256, 256) uint16（维度来自第三方博客，未在官方页核实）；z=1.625 µm、xy=0.40625 µm（同上） | **有，但稀疏（sparse ground truth）**：节点=细胞中心 (t,z,y,x)，边=跨帧链接，分裂=1 节点→t+1 两节点；**只标注一部分细胞**（baseline README 明示 sparse supervision）；分割掩码：无（只有点/图） | `.geff`（tracksdata 图格式）；Kaggle 提交为 CSV | 需 Kaggle 注册参赛；许可条款未核实；无现金奖（积分/奖牌） | **低–中**：图结构（节点/边/分裂）与 GNN 边级三分类**天然同构**，是本清单里最贴近你 pipeline 的标注形式；但需处理"稀疏标注"（未标注细胞不可当负样本） | 未核实（Zarr 体数据，量级较大） |
| T4 | **Zebrahub imaging（CZ Biohub / Royer lab）**<br>https://zebrahub.sf.czbiohub.org/imaging<br>数据目录：https://public.czbiohub.org/royerlab/zebrahub/imaging/single-objective/ | 斑马鱼胚胎，7 套光片时序（ZSNS001/001_tail/002/003/004/005 单物镜 + ZMNS001/002 多视角）；OME-Zarr | **有细胞追踪数据**（官网原文：7 套 time-lapse + "cell tracking data"）；追踪用 Ultrack 生成，**是否含分裂/谱系字段未核实**；分割掩码：未核实 | **OME-Zarr**（图像）+ **tracks.csv**（ZSNS001 849 MiB、ZSNS001_tail 464 MiB、ZSNS003 199 MiB、ZSNS004 360 MiB、ZSNS005 326 MiB）+ 每套一个 .json 元数据；另有 `tracks_benchmark/` | 页面未给出明确许可 → **未核实**；大部分数据公开 HTTP 直下；`photo-manipulation/` 子集需填 Google 表单 | **低–中**：CSV 轨迹 + OME-Zarr 图像，转换量可控；适合做"检测→关联"训练与跨数据集泛化测试 | 未核实（体数据，GB–TB 级） |
| T5 | **BioEmergences-wt3（Mendeley Data）**<br>https://data.mendeley.com/datasets/vcgbmpr366/1 | 斑马鱼野生型胚胎原肠期，**3D+time**，核通道 + 膜通道 | **有数字细胞谱系**（BioEmergences workflow 计算，DOI 10.1038/ncomms9674）；分割掩码：未核实（VTK 场数据） | **VTK**（3D+time 场数据），单文件 wt3.zip | **CC BY 4.0**；无需注册（Mendeley 直接下载，实测 1,933,480,674 B） | **高**：VTK 非主流训练格式，需写转换（体素化 + 从谱系重建逐帧实例标签） | **约 1.93 GB** |
| T6 | **IDR idr0013 = MitoCheck（Neumann et al. 2010）**<br>https://idr.openmicroscopy.org/study/idr0013/<br>（镜像：BioImage Archive S-BIAD865） | HeLa H2B-GFP，2D+time 全基因组 RNAi 筛选；191,368 个 5D 图像 / 17,851,318 个 2D 平面；平均维度 1344×1024×1×1×**93 帧** | **无轨迹/谱系 GT**（只有表型分类表；分割掩码无）。→ 只能做**分割/检测预训练 + 自监督伪标签** | OME-Zarr / OME-TIFF（IDR 提供 Zarr 下载与原图下载） | **CC0 1.0**；无需注册 | **高（就追踪而言）**：无实例标签；需先用现成分割模型产伪标签，再自建轨迹 | **18.98 TB** |
| T7 | **IDR idr0167 = CellCycleNet**<br>https://idr.openmicroscopy.org/study/idr0167/ | 3D 单细胞核染色图像（1024×1024×150×3ch，593 个 5D 图像，1 TB 级） | 标注为**细胞周期分期**；是否含逐帧轨迹/谱系 → **未核实**；分割掩码未核实 | OME-Zarr / OME-TIFF + 注释 CSV | **CC BY 4.0**；无需注册 | 中（若确有时序即可用；先下载小样本核实） | 约 1–4 TB（页面列 1 / 4 TB 两实验） |
| T8 | **IDR 其他长时序成像研究**（原始图像，无 GT 轨迹）<br>idr0002（330 帧，CC BY 4.0）、idr0068-shah-zebrafishlightsheet（420 帧，CC BY 4.0）、idr0118-keenan-flylightsheet（95 帧，CC BY 4.0）、idr0109-zaritsky-melanoma（607 帧，CC BY 4.0）、idr0111-lee-cellmigration（128 帧）、idr0061-wolf-spindlepositioning（180 帧）、idr0104-goglia-erkdynamics（121 帧）、idr0108-sabinina-nuclearporecomplex（37,735 帧） | 3D/2D + time，荧光/明场，胚胎与细胞系 | **无轨迹/谱系/分割 GT**（IDR 通常只托管图像 + ROIs） | OME-Zarr（新版强制）/ OME-TIFF | 逐研究 CC BY 4.0 或 CC0（上述已核实 CC BY 4.0）；无需注册 | **高**：只能当"未标注时序视频"，用于自监督/伪标签/域适应 | 0.02–5.9 TB/研究 |
| T9 | **StarryNite / AceTree（C. elegans 谱系追踪）**<br>https://starrynite.sourceforge.net/ | C. elegans 胚胎，3D+time 荧光核；PNAS 2006 / Nat Protoc 2006 | 设计目标是**全胚胎细胞谱系追踪**（输入原始 4D → 输出谱系）；样本数据链接 http://waterston.gs.washington.edu/StarryNite.html | 自有文本/图像输出格式 | 开源工具（SourceForge）；**样本数据可用性未核实（链接为 2010 年代旧地址，很可能已失效）** | 高（工具链古老、数据可用性待确认） | 未核实 |

---

## 2. 总表（仅分割掩码 / 检测点 —— 用于分割预训练）

| # | 名称 / 官方链接 | 维度·模态·规模 | 轨迹/谱系 | 分割掩码（稀疏/全量） | 格式 | 许可 / 注册 | 对接管线成本 | 量级 |
|---|---|---|---|---|---|---|---|---|
| S1 | **LIVECell**<br>https://sartorius-research.github.io/LIVECell/<br>https://github.com/sartorius-research/LIVECell | 2D 相差（label-free）活细胞；8 种细胞系（A172/BT474/BV2/Huh7/MCF7/SHSY5Y/SkBr3/SKOV3），>160 万细胞 | 无（图像名含时间戳，但每图独立标注，无轨迹） | **全量实例分割**（人工标注+专家校验） | 图像 TIFF（`images.zip`）+ **COCO json** 标注 + Detectron2 权重 | **CC BY-NC 4.0**；无需注册 | **低**：最干净的 2D 分割预训练集；COCO→掩码一行脚本；但**无时序、无核**（是整细胞） | 图像 **1.3 GB** + 标注 **2.1 GB**（模型另 15 GB） |
| S2 | **TissueNet**<br>https://deepcell.readthedocs.io/en/master/data-gallery/tissuenet.html | 2D 荧光，**组织**（非培养皿），核 + 全细胞双标注；train≈2600×512²、val≈3000×256²、test≈1200×256² | 无 | **全量**（核 mask + 细胞 mask），TissueNet 1.1 经过额外人工 QC | Python API（deepcell.datasets），底层 npz/数组 | 修改版 Apache，**仅非商业学术**；需 DeepCell API key | **低**：组织域泛化预训练首选；需 API key 与导出脚本 | 未核实 |
| S3 | **Cellpose 官方数据集**<br>https://www.cellpose.org/dataset | 2D/3D 混合（cyto 系列为 2D 相差/荧光；nuclei 为 3D 鼠脑核） | 无 | 全量实例掩码（"Cellpose annotated dataset"） | 官方打包 zip（具体内部格式需注册后查看） | **需接受 HHMI 研究条款 + 填机构邮箱**；README 明示 **CC-BY-NC**（https://github.com/MouseLand/cellpose ） | 低（若是标准图像+掩码）；**子集清单与规模未核实**（官方页需注册后才给出下载） | 未核实 |
| S4 | **Broad Bioimage Benchmark Collection（BBBC）**<br>https://bbbc.broadinstitute.org/image_sets | 以**静态单帧**为主；荧光/DIC/明场；从 6 图（BBBC001）到 690,000 图（BBBC051） | 基本无（BBBC035 派生自 CTC，属于例外） | 视条目而定：counts / foreground / outlines / biological（BBBC038、BBBC039、BBBC050 含分割或计数 GT） | 多为 TIFF/PNG + zip；标注为 txt/PNG/csv | 逐条不同 → **未逐一核实**；无需注册 | **低–中**：核分割预训练可用（**BBBC038 = Kaggle 2018 Data Science Bowl，670 图**；BBBC039 = 200 图 U2OS 核；BBBC050 = 165 图小鼠胚胎核；BBBC003/032/033 = 小鼠胚胎）；时序追踪不可用 | 单条目 MB–GB 级 |
| S5 | **PanNuke**<br>https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/ | 2D H&E 病理，pan-cancer，核实例 + 5 类细胞类型；3 个 fold zip | 无 | **全量实例 + 类型** | zip（fold_1/2/3.zip）+ HoVer-Net 权重 | **CC BY-NC-SA 4.0**，仅研究；无需注册 | 低（分割预训练）；域与荧光显微相差远 → 只作辅助 | 未核实（GB 级） |
| S6 | **NuCLS**<br>https://nucls.grand-challenge.org/ | 2D H&E（TCGA 乳腺癌），核分类+定位+分割，众包+深度学习方法 | 无 | 全量（多级标注） | grand-challenge 下载包（另有 HuggingFace 镜像） | 未核实 | 低（分割预训练） | 未核实 |
| S7 | 其他 2D 静态核分割集：**MoNuSeg** https://monuseg.grand-challenge.org/ ；**CoNSeP**（HoVer-Net 页）https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/ ；**CryoNuSeg** https://zenodo.org/records/3713970 ；**TNBC** https://zenodo.org/records/1175282 | 2D，H&E 或冷冻切片 | 无 | 全量实例掩码 | 各站 zip / npz | 各自条款（未核实） | 低（分割预训练） | KB–GB |
| S8 | **IDR idr0062 = Blin NesSys**<br>https://idr.openmicroscopy.org/study/idr0062/ | **3D** 荧光，胚胎（mid-gestation）与 3D 培养/神经玫瑰花环；22 个 5D 图像（618×548×168×3） | 无 | 研究类型标注为 **image segmentation**，提供 annotation CSV（3D 核分割 GT） | OME-Zarr/OME-TIFF + annotation CSV | **CC BY 4.0**；无需注册 | 中（3D 核分割预训练很对口；需从 CSV/ROI 重建掩码） | 0.02 TB |
| S9 | **Allen Cell Explorer（Allen Institute for Cell Science）**<br>https://www.allencell.org/data-downloading.html | **3D** 荧光 hiPSC（31,987 个 3D 细胞图像），结构蛋白标记 | 无（单时间点为主） | 部分数据有分割（Allen Cell & Structure Segmenter 配套 GT） | Quilt / S3 程序化访问；桌面软件包 | Allen Institute 使用条款（页面显示 "terms of use"）：**非商业学术**（具体条款未逐条核实） | 中（3D 域泛化预训练；Quilt 取数需写脚本） | 未核实（TB 级） |
| S10 | **OpenCell**（CZ Biohub）<br>https://opencell.czbiohub.org/ ；数据集卡：https://virtualcellmodels.cziscience.com/dataset/opencell-microscopy-images | 2D/3D 荧光，1,000+ 内源 GFP 标记细胞系；每个蛋白 4–6 个视野、z 堆栈约 100 层、2 通道（DNA + GFP） | 无（静态） | 下载页提供标注：https://opencell.czbiohub.org/download （数据集卡提示 "Usage not covered by the license"） | 图像（z 堆栈 + MIP）；标注格式未核实 | **需自行到下载页确认**（存在使用限制）；**未核实** | 中（亚细胞/核分割预训练；非时序） | 未核实 |
| S11 | **HuBMAP**：数据门户 https://portal.hubmapconsortium.org/ ；Kaggle 竞赛：<br>· HuBMAP + HPA – Hacking the Human Body https://www.kaggle.com/competitions/hubmap-organ-segmentation<br>· HuBMAP – Hacking the Kidney https://www.kaggle.com/competitions/hubmap-kidney-segmentation<br>· HuBMAP – Hacking the Human Vasculature https://www.kaggle.com/competitions/hubmap-hacking-the-human-vasculature | 2D 全切片/组织（H&E、PAS、荧光），静态 | 无 | 竞赛提供全量实例/结构掩码 | Kaggle 数据集（TIFF + json/rle）；门户为 OME-TIFF/zarr | Kaggle 竞赛条款 + HuBMAP 数据使用协议；需注册 | 中（组织域，非细胞培养核；仅分割预训练） | 未核实（TB 级） |
| S12 | **NeurIPS 2022 Cell Segmentation Challenge**<br>https://neurips22-cellseg.grand-challenge.org/<br>（结果论文：https://zenodo.org/records/10718351 ） | 多模态高分辨率显微图像（含 ~10,000×10,000 全切片），弱监督设定 | 无 | 部分标注（弱监督：少量标注图 + 大量无标注图） | grand-challenge 提交包 | **需注册**（须用 NeurIPS 注册邮箱）；许可未核实 | 中（弱监督分割预训练；流程较重） | 未核实 |
| S13 | **SpotNet**（DeepCell）<br>https://deepcell.readthedocs.io/en/master/data-gallery/spotnet.html | 2D 荧光，**点/斑点**（非完整细胞核） | 无 | **坐标点标注**（检测 GT，不是掩码） | Python API（X + 坐标 y） | 修改版 Apache，非商业学术；需 API key | 低（若你的管线要"检测点"分支可参考；与核实例分割不同任务） | 未核实 |
| S14 | Kaggle **2018 Data Science Bowl** = BBBC038 https://www.kaggle.com/competitions/data-science-bowl-2018 ；Kaggle **Sartorius Cell Instance Segmentation (2021)** = LIVECell https://www.kaggle.com/competitions/sartorius-cell-instance-segmentation | 同 S1/S4 | 无 | 全量 | Kaggle 数据集 | Kaggle 条款（Sartorius 数据本体为 CC BY-NC 4.0） | 低 | 同 S1/S4 |

---

## 3. 检索到但**不建议**投入的条目（附理由）

| 名称 / 链接 | 为什么不建议 |
|---|---|
| **DynamicAtlas / Morphodynamic atlas for Drosophila development** — Dryad https://datadryad.org/dataset/doi:10.25349/D9WW43 （论文 https://www.nature.com/articles/s41592-025-02897-8 ） | 真实果蝇胚胎光片形变图谱，但 **170.81 GB（2024 版）/ 181.46 GB（2022 版）tar.lz4**、文件按基因型/标记分散（53+ tar）、页面**未给出实例掩码或轨迹 GT** → 需要极大转换成本且追踪标签未核实 |
| **Zenodo 22807312（斑马鱼 253 时间点 Gaussian splats）** https://zenodo.org/records/22807312 | **不是原始体素数据**（是拟合出的 3D Gaussian splats，2 个 zip 约 1.1 GB + 5.9 GB），**无追踪标注**，只适合可视化/渲染 |
| **Fly-QMA** | 实为 "Fly-QMA: Automated analysis of mosaic imaginal discs in Drosophila"（**成虫盘克隆嵌合分析**，不是胚胎时序谱系追踪）；GitHub 上未检索到公开代码/数据仓库（`api.github.com/search/repositories?q=Fly-QMA` 无命中）→ **数据下载地址未核实**，与你的任务不匹配 |
| **BioImage.IO** https://bioimage.io/ | 是**模型库（Model Zoo）**，不是数据集仓库；页面为 JS 应用，无法通过抓取核实其是否托管时序追踪数据（未核实） |
| **IDR 大型图谱类研究**（如 idr0043-uhlen-humanproteinatlas 159.39 TB、idr0090-ashdown-malaria 23.92 TB） | 静态/非追踪、体量过大，性价比极低 |
| **DeepCell 旧版（≤0.11）内置数据集** | 现版本 `deepcell.datasets` 仅暴露 TissueNet / DynamicNuclearNet / SpotNet（已核实源码 `deepcell/datasets/__init__.py`）；旧版清单未核实，不建议依赖 |

---

## 4. 分档建议（针对"nnU-Net 分割 → 实例化 → 相邻帧图 → GNN 边级三分类 → 谱系"）

### 第一档：**最值得直接接入追踪训练**（有逐帧身份 + 跨帧关联/分裂）
1. **DynamicNuclearNet-Tracking（T1）** —— 唯一一个同时给"核分割掩码 + lineage（cell id / 出现帧 / 父-子分裂链接）"的现成数据集，语义上与你的 TRA 标签几乎一一对应。代价是 DeepCell API key + 导出脚本。**建议作为除 CTC 之外的第一优先。**
2. **Biohub – Cell Tracking During Development（T3）** —— 标注形式（节点 (t,z,y,x) / 边 / 分裂）与 GNN 边级三分类完全同构，且官方 baseline 就是"检测 + 跨帧关联"，可当参照实现。**关键差异：标注是稀疏的**（未标注细胞不能当负样本），正好可以检验你 pipeline 在稀疏监督下的退化程度。
3. **Zebrahub tracks（T4）** —— 7 套斑马鱼光片时序 + Ultrack 轨迹 CSV，体量可控、可直接 HTTP 下载；适合做"跨物种/跨模态泛化"的第二个测试集。需先确认 CSV 是否含分裂字段与许可条款。

### 第二档：**可用但要写转换器 / 只能做弱监督追踪**
4. **BioEmergences-wt3（T5）** —— 有数字细胞谱系、CC BY 4.0、仅 1.93 GB，是"便宜"的时序谱系数据；但 VTK 格式需自己体素化并重建逐帧实例标签。
5. **BlastoSPIM（T2）** —— 3D 核实例掩码质量高且 CC0（23 GB），**但轨迹/谱系标注未核实**：先按 npy.gz 下载 1–2 个序列核实是否有时序 id，再决定它是"追踪数据"还是"3D 分割预训练数据"。

### 第三档：**只适合分割预训练（把 nnU-Net 前端做强）**
- 同类域（培养细胞、2D）：**LIVECell（S1）**、**Cellpose cyto 系列（S3）**、**BBBC038 / BBBC039 / BBBC050（S4）**
- 组织/病理域：**TissueNet（S2）**、**PanNuke（S5）**、**NuCLS（S6）**、**MoNuSeg/CoNSeP/CryoNuSeg/TNBC（S7）**、**HuBMAP 竞赛（S11）**、**NeurIPS 2022 CellSeg（S12）**
- 3D 核：**IDR idr0062（S8）**、**BlastoSPIM 分割部分（T2）**、**Allen Cell Explorer（S9）**
- ⚠ 这些**全部没有时序轨迹**，不能用于追踪监督；且**域差（H&E vs 荧光 vs 相差）与"核 vs 整细胞"必须显式区分**——你的任务目标是**细胞核掩码**，LIVECell/TissueNet 的细胞膜/全细胞标签不能直接当核标签用。

### 第四档：**只当未标注视频做自监督/域适应**
- **IDR 长时序研究（T8）**，尤其 **idr0013 MitoCheck（T6，CC0，18.98 TB，93 帧/片）**：体量大、无 GT，但 CC0 + 真实 HeLa 有丝分裂动态，适合做伪标签 + 时序一致性自监督。

### 第五档：**不建议**（见第 3 节）

### 关键结论（一句话）
除 CTC 外，**没有**第二个数据集能同时提供"密集逐帧实例 + 谱系 + 分割掩码"的 TRA 级真值：
- 想要**追踪监督** → DNN（T1）、Biohub Kaggle（T3）为主，Zebrahub（T4）为辅；
- 想要**分割强度** → LIVECell / Cellpose / BBBC / IDR idr0062 / BlastoSPIM 预训练，再回 CTC 微调；
- 想要**体量** → MitoCheck（T6，CC0）做自监督。

### 与现有 pipeline 的接口提示
- 若要把上述数据喂进现有 GNN，需要产出的最小结构就是 `h5`：每帧节点（位置 + 强度/尺寸特征）+ 相邻帧候选边 + `move/divide/no-link` 三分类标签 + 父子关系。**T1 / T3 是唯二能直接生成这套标签的数据集**（T3 稀疏、T1 全量）。
- 注意 R11（训练/测试检测来源同分布）：这些数据集上若用 GT 掩码训练 GNN，就必须同时用 GT 掩码建图推理，不能混预测实例。
- 注意 R9（双序列）：可用 T1 的多个 batch / T3 的多个胚胎 / T4 的多套 timelapse 做"双序列"式复验，而不是只看单一序列。

---

## 5. 未核实项清单（明确留白，避免误用）

1. DynamicNuclearNet 的**图像数/帧数/GB 数**（需 API key 后实测）。
2. BlastoSPIM 的**逐帧轨迹与谱系标注**（figshare 描述只写 3D 实例分割）。
3. Zebrahub 的**许可条款**与 tracks CSV 的**分裂/谱系字段定义**。
4. Biohub Kaggle 数据的**总体量**、**官方许可**、样本维度（维度数字来自第三方博客转述）。
5. Cellpose 官方数据集的**子集清单与规模**（页面需接受条款+邮箱后才给下载）。
6. TissueNet / Cellpose / Allen / OpenCell / PanNuke / NuCLS / HuBMAP / NeurIPS2022 的**精确 GB 数与逐条许可原文**。
7. StarryNite 样本数据的**现行可用性**（链接指向 `waterston.gs.washington.edu`，年代久远）。
8. Fly-QMA 的**数据集下载地址**（工具论文存在，数据仓库未检索到）。
9. IDR idr0167（CellCycleNet）是否含**逐帧轨迹**。
10. BBBC 各条目的**许可**（站点按条目区分，本次未逐条核实）。
