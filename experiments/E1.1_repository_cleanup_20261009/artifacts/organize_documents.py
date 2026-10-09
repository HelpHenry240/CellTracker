#!/usr/bin/env python3
"""文档按职责分类；旧文档完整归档，实验数值与原始证据保持不变。"""
from pathlib import Path
import json
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[3]


def write(name, content):
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + '\n')


def main():
    previous = sorted(p for p in (ROOT / 'docs').rglob('*.md'))
    retained = {'docs/decisions/0001-mass-uniform-vs-size.md',
                'docs/decisions/0004-ideas-pipeline-rebuild.md',
                'docs/reports/ideas_pipeline_rebuild_20261009.md',
                'docs/reports/gt_oracle_encoder_20261009.md',
                'docs/reports/nnunet_oracle_encoder_20261009.md',
                'docs/reports/dataset_survey_nuclei_seg_tracking.md',
                'docs/reports/sota_3d_cell_tracking_survey_20261009.md'}
    moves = {'docs/decisions/0004-ideas-pipeline-rebuild.md': 'docs/decisions/0002-ideas-pipeline-rebuild.md',
             'docs/reports/dataset_survey_nuclei_seg_tracking.md': 'docs/references/dataset_survey.md',
             'docs/reports/sota_3d_cell_tracking_survey_20261009.md': 'docs/references/tracking_survey_20261009.md'}
    migration = []
    for path in previous:
        name = str(path.relative_to(ROOT))
        target = moves.get(name, name) if name in retained else None
        migration.append({'original': name, 'current': target,
                          'original_archive': 'experiments/archives/00_source_and_documents.tar.gz'})
        if name not in retained:
            path.unlink()
        elif name in moves:
            (ROOT / moves[name]).parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(ROOT / moves[name]))
    for path in (ROOT / 'docs/reports').glob('*.md'):
        content = path.read_text()
        replacements = {
            'scripts/run_ideas_pipeline.py': 'scripts/run_pipeline.py',
            '`paperpipe/FORMULA_MAP.md`': '`docs/architecture/formula_map.md`',
            '`paperpipe/configs/ce_calibrated_20261009.yaml`': '`configs/ce_calibrated_20261009.yaml`',
            '`paperpipe/README.md`': '`docs/guides/running.md`',
            '方法实现为 `paperpipe/`': '方法实现为 `src/papertrack/`',
            '外层旧追踪入口保留为历史基线。': '公共组件集中于 `src/celltracker/`，历史对照 API 保留供回归。',
        }
        for old, new in replacements.items():
            content = content.replace(old, new)
        content = content.replace('/artifacts/evidence_formal/archive_manifest.json)', '/../../archives/catalog.json)')
        content = content.replace('/../../archives/catalog.json)', '/../archives/catalog.json)')
        content += '\n整理说明（2026-10-09）：入口与目录链接已更新；实验数值和当时的测试计数保持原样。'
        content += '完整原件见 [归档目录](../../experiments/archives/README.md)，保留文件的逐项校验见 [清单](../../experiments/retention_manifest.json)。\n'
        path.write_text(content)
    p = ROOT / 'docs/decisions/0002-ideas-pipeline-rebuild.md'
    p.write_text(p.read_text().replace('# 0004：', '# 0002：') +
                 '\n2026-10-09 目录整理：实现集中到 src/papertrack，共享组件在 src/celltracker；主入口为 scripts/run_pipeline.py。原编号0004与全文原件保留在归档。\n')
    p = ROOT / 'docs/decisions/0001-mass-uniform-vs-size.md'
    p.write_text(p.read_text() + '\n当前适用范围：本决定只适用于 GT marker 检测档；nnU-Net 真实实例与掩码 Oracle 使用体积质量，不能把 marker 决定推广到真实细胞区域。\n')
    write('docs/history/document_migration.json', json.dumps(migration, ensure_ascii=False, indent=2))
    with tarfile.open(ROOT / 'experiments/archives/00_source_and_documents.tar.gz') as handle:
        formula = handle.extractfile('paperpipe/FORMULA_MAP.md').read().decode()
    formula = formula.replace('paperpipe/src/papertrack/', 'src/papertrack/')
    formula = formula.replace('paperpipe/src/vendor/celltracker/', 'src/celltracker/')
    formula = formula.replace('vendor/celltracker/ot/', '../celltracker/ot/')
    formula = formula.replace('分别记录，不能把旧实现的 TRA 当作新实现已达到的结果。',
                              '分别记录，不能把旧实现的 TRA 当作新实现已达到的结果。旧实验完整归档，任务索引见 ../../experiments/INDEX.md。')
    write('docs/architecture/formula_map.md', formula)

    write('README.md', '''
# CellTracker

按 `ideas.pdf` 实现的 3D 细胞追踪：预测实例 → 冻结 nnU-Net encoder → OT/FGW → 多帧图 → GNN → 轨迹重建 → 官方 CTC 评测。

源码统一位于 `src/`：`papertrack` 是当前论文主链路，`celltracker` 提供共享组件及明确保留的历史对照 API。已移除重复 vendor 与 `paperpipe/` 目录。

| 入口 | 内容 |
| --- | --- |
| [文档导航](docs/README.md) | 架构、运行、评测、决策与研究资料 |
| [项目进度](docs/status.md) | 已完成验证、结论边界及后续任务 |
| [运行说明](docs/guides/running.md) | 统一 CLI、训练、配置与消融 |
| [任务实验索引](experiments/INDEX.md) | 按进度排列的当前证据 |
| [合并实验记录](experiments/RECORDS.md) | 历史到当前的阶段结论 |
| [脚本清单](scripts/README.md) | 保留脚本的用途 |

```bash
conda activate celltracker
python -m pytest
python scripts/run_pipeline.py --list-modules
```

完整算法模板为 `configs/paper_default.yaml`；当前真实预测实例预设为 `configs/ce_calibrated_20261009.yaml`。数据、冻结 nnU-Net 大权重和完整历史压缩包不随 Git 提交；当前正式 GNN 小权重与逐项指纹保留在 `experiments/`。
''')
    write('docs/README.md', '''
# 文档导航

每类信息只维护一个入口。阶段报告保留当时的数字，当前状态以进度页和实验索引为准。

| 分类 | 文档 | 用途 |
| --- | --- | --- |
| 进度 | [status.md](status.md) | 已完成项目、限制、下一步 |
| 架构 | [pipeline.md](architecture/pipeline.md)、[formula_map.md](architecture/formula_map.md) | 数据流、模块职责、论文公式 |
| 使用 | [running.md](guides/running.md) | 配置、统一入口、消融、长任务 |
| 评测 | [evaluation.md](guides/evaluation.md) | 检测来源、官方指标、噪声、对照 |
| 实验管理 | [experiments.md](guides/experiments.md) | 八件套、归档、恢复与追加记录 |
| 决策 | [marker质量](decisions/0001-mass-uniform-vs-size.md)、[论文重建](decisions/0002-ideas-pipeline-rebuild.md)、[评测协议](decisions/0003-evaluation-protocol.md) | 决策依据与适用范围 |
| 正式报告 | [真实预测实例](reports/ideas_pipeline_rebuild_20261009.md)、[GT marker Oracle](reports/gt_oracle_encoder_20261009.md)、[NN掩码 Oracle](reports/nnunet_oracle_encoder_20261009.md) | 三档双序列、双种子结果 |
| 研究资料 | [数据集调查](references/dataset_survey.md)、[追踪研究调查](references/tracking_survey_20261009.md) | 保留资料的调查日期，不代表持续更新 |
| 历史 | [复盘](history/README.md)、[迁移清单](history/document_migration.json) | 被合并文档的原位置与归档入口 |

完整数字和原始日志见 [实验索引](../experiments/INDEX.md)。
''')
    write('docs/status.md', '''
# 项目进度（2026-10-09）

当前 pipeline 已按论文重建并跑通；真实检测和两类 Oracle 均有双序列、双种子官方验证。此次文件整理只调整代码位置、文档和证据保留层级，未新增训练或修改方法。

| 阶段 | 状态 | 证据 |
| --- | --- | --- |
| 仓库/数据/经典基线 | 完成，作为历史参照 | E0.1–E0.3、E1旧基线、E5分割训练 |
| 早期追踪与前端试验 | 已归档，不能移用旧模块结论 | A/B/C、E2–E4、P 系列 |
| 论文审计与重建 | 完成；v2探索含已修复问题，已归档 | E0.4–E0.6 |
| 冻结前端/encoder/同来源映射 | 完成 | E0.7 |
| 真实预测实例正式评测 | 完成，4配置×2种子、20次官方评测 | E0.8 |
| GT marker + encoder | 完成，2种子、5次官方评测 | E0.9 |
| nnU-Net掩码 + GT种子Oracle + encoder | 完成，2种子、5次官方评测 | E1.0 |
| 源码合并与记录整理 | 工程整理，验收另列 | E1.1_repository_cleanup |

| 当前检测来源 | seq01 DET | seq01 SEG | seq01 TRA均值 | seq02 DET | seq02 SEG | seq02 TRA均值 |
| --- | --- | --- | --- | --- | --- | --- |
| 真实预测实例，k=1.6，保留孤立实例 | 0.955785 | 0.663599 | 0.9325185 | 0.951608 | 0.667235 | 0.9308145 |
| GT marker + encoder | 1.000000 | 不适用 | 0.9979770 | 1.000000 | 不适用 | 0.9976020 |
| NN掩码 + GT种子Oracle + encoder | 0.998328 | 0.683938 | 0.9960460 | 0.993644 | 0.698012 | 0.9907585 |

数值来自各阶段 `metrics_final.json`，每档各有两个独立种子及完整重复。三档设置不同，不能把跨档差值归因于 encoder 或单个后处理模块。

当前瓶颈仍是实例归属：真实实例的粘连/误检使其明显落后于 Oracle。但 Oracle 使用 GT 辅助且改变体积过滤、候选和空洞策略，只能作为诊断条件。

后续按依赖顺序推进：

1. 真实预测档的实际部署候选审计：分别报告移动与分裂召回。E0.8的seq01标定记录总体召回约99.8%，但k=1.6分裂仅667/685，不能用总体值掩盖少数类损失；先复核实际部署图，保留既有正式指标。
2. 在同一检测来源/当前版本上做核心模块消融：FGW、运动、多尺度、tracklet 等每项独立重训，至少双序列、双种子、官方指标和重复评测。旧版扩展“无收益”结论不能代替当前版本的消融。
3. 合成 FN/FP/过分割/欠分割与遮挡案例，量化鲁棒性及空洞恢复。接口和补画实现已有，论文级验证仍待完成。
4. 前端实例感知目标、Distance/Boundary 方向需另行方法决策；涉及 nnU-Net 重训前询问用户。当前冻结分割器不变。
5. 建立分割器也未见过的数据划分，完成端到端泛化、多序列终表和论文图表。

手工特征对照按用户要求暂缓。既有 nnU-Net 训练见过部分 seq02（150/190帧），当前 seq02 仅对 GNN 保持跨序列，不能声称端到端盲测。详细限制见 [评测协议](guides/evaluation.md)。
''')
    write('docs/architecture/pipeline.md', '''
# 架构与源码职责

`src/` 是唯一源码根目录；两个命名空间共同安装，保留 `papertrack` 名称以兼容已有模型。共享求解器只保留一份。

```mermaid
flowchart LR
    A[原图及冻结NN前景] --> B[实例H5]
    B --> C[冻结encoder侧车]
    C --> D[测度与帧内结构]
    D --> E[粗追踪运动及相邻OT/FGW]
    E --> F[多尺度精炼]
    F --> G[多帧图]
    G --> H[GNN边判定]
    H --> I[轨迹重建及tracklet]
    I --> J[流式CTC导出与校验]
    J --> K[官方DET/SEG/TRA]
```

| 代码位置 | 职责 |
| --- | --- |
| `src/papertrack/representation` | 物理测度、代价、帧内结构 |
| `src/papertrack/coupling` | 相邻/跨帧耦合编排 |
| `src/papertrack/temporal` | 多尺度完整目标及回溯 |
| `src/papertrack/longrange` | 运动与二层tracklet |
| `src/papertrack/graph` | 命名节点/边特征、候选、监督与上下文 |
| `src/papertrack/gnn` | 论文二分类 BCE/OT 损失、训练、推理 |
| `src/papertrack/reconstruction` | 生死、分裂、冲突、空洞与导出 |
| `src/papertrack/runtime` | 契约、标定、消融、主流程、官方评测 |
| `src/celltracker/{data,detect,cost,ot,track,eval,experiment}` | 共享数据/实例化、求解器、容器、IO、留痕 |
| `src/celltracker/gnn/data.py`、`pipeline/motion.py` | 共享批处理与运动估计 |
| `src/celltracker/{pipeline,gnn,graph}`中的历史控制实现、`track/ot_tracker.py` | 早期对照与回归用途；当前 CLI 不调用旧编排器 |

`celltracker.gnn` 与 `celltracker.pipeline` 的顶层历史接口按需加载，避免导入公共数据模块时拉入整条旧链路。所有共享计算模块保持原实现；没有 vendor 路径覆盖。

H5记录检测来源、spacing、实例label及可选GT重叠映射；encoder侧车记录逐帧身份和模型指纹。图、缓存、GNN权重均校验上游契约。`--h5` 是检测来源，`--gt-h5` 只提供监督、标定与评测真值；GT辅助Oracle必须单独标注。

论文逐式对应与工程补充见 [公式映射](formula_map.md)。
''')
    write('docs/guides/running.md', '''
# 运行、配置与消融

在仓库根目录使用已有 `celltracker` 环境。源码部署须包含完整 `src/`、`scripts/` 与 `configs/`，不再单独打包旧 `paperpipe/`。

```bash
conda activate celltracker
python -m pytest
python scripts/run_pipeline.py --list-modules
python scripts/run_pipeline.py --config configs/ce_calibrated_20261009.yaml --list-modules
```

`paper_default.yaml` 是完整方法模板，需针对输入来源标定。`ce_calibrated_20261009.yaml` 是E0.8真实预测实例、k=1.6、保留孤立实例的现有预设；不能直接用于marker或新Oracle。`--set section.field=value`覆盖参数，`--ablate name`按当前状态反向切换。

已有前端产物的实例H5路径因本地/云端位置不同，可先按实际位置设置以下任务变量：

```bash
CT_DETECTIONS=data/interim/Fluo-N3DH-CE_01_pred_resplit_v2.h5
CT_TRUTH=data/interim/Fluo-N3DH-CE_01.h5
CT_ENCODER=data/interim/encoder_01_resplit_v2.npz
CT_CONFIG=configs/ce_calibrated_20261009.yaml
```

标定应先打印物理位移、原始C、Γ与候选召回分布；只用seq01设置阈值：

```bash
python scripts/calibrate_params.py --h5 "$CT_DETECTIONS" --gt-h5 "$CT_TRUTH" \
  --config "$CT_CONFIG" --set node.encoder_feat_path="$CT_ENCODER" \
  --out data/interim/new_calibration.json --apply-out data/interim/new_calibration.yaml
```

独立训练时先完整上游建图，再运行统一训练入口。以下为新实验命令模板；完整队列应按当前服务器实际耗时在云端后台执行，本地只做小样本：

```bash
python scripts/run_pipeline.py --h5 "$CT_DETECTIONS" --gt-h5 "$CT_TRUTH" --seq 01 \
  --exp-id NEW_build --config "$CT_CONFIG" --set node.encoder_feat_path="$CT_ENCODER" \
  --dump-graphs data/interim/graphs_NEW --build-only
python scripts/train_gnn.py --graphs data/interim/graphs_NEW --out data/interim/model_NEW \
  --config "$CT_CONFIG" --epochs 60 --seed 20261008 --device cpu --exp-id NEW_train
python scripts/run_pipeline.py --h5 "$CT_DETECTIONS" --gt-h5 "$CT_TRUTH" --seq 01 \
  --exp-id NEW_eval --config "$CT_CONFIG" --set node.encoder_feat_path="$CT_ENCODER" \
  --ckpt data/interim/model_NEW/best.pt --official
```

复用现有E0.8权重时保持同一前端及上游配置，仅替换数据与对应encoder位置；权重可在 `experiments/E0.8_ideas_final_v3_20261009/artifacts/models/conservative_k16/` 找到。两类Oracle的正式配置和权重分别在E0.9/E1.0中。改变来源或上游契约后必须独立重训。

矩阵入口 `scripts/run_ablation_matrix.py` 默认只生成计划，`--execute`才执行，`--resume`恢复已完成任务。传入 `--h5-01/02`、`--gt-01/02`、`--encoder01/02`、`--config`、`--out`、`--exp-prefix`，使用两个种子并独立训练每项变体。`--recalibrate`只在seq01标定各变体。具体参数以 `--help` 为准。

云端长任务使用 `nohup`、独立日志与PID；GPU启动前/运行中/结束后检查 `nvidia-smi`，完成后核验非零checkpoint。SSH连接由 `scripts/cloud_run.sh` 与本地凭据文档管理；备用连接使用 `CT_CLOUD_CONN=backup`，凭据不入库。nnU-Net重训需用户确认；本次整理未启动云任务。

模块开关与实现范围见 [公式映射](../architecture/formula_map.md)，结果口径见 [评测说明](evaluation.md)。
''')
    write('docs/guides/evaluation.md', '''
# 评测与对照口径

官方 CTC DET/SEG/TRA 是结论依据，本地诊断仅定位问题。官方二进制在云端运行；统一 `run_pipeline.py --official` 在导出后强制格式校验。可用 `--official-tools`、`--official-gt-dir`直接调用当前服务器工具；否则走 `cloud_eval.py`上传并在完成后清理临时评测目录。

| 检测来源 | GT作用 | 可报告的结论 |
| --- | --- | --- |
| GT marker Oracle | 提供检测位置/标记及边监督 | DET恒1，SEG无分割含义；隔离追踪贡献 |
| nnU-Net真实预测实例 | 仅训练映射、标定及评测 | 实际前端条件下的全链路结果 |
| NN掩码 + GT种子Oracle | 额外提供实例化种子 | 诊断性上界条件，不能称为可部署检测 |
| 合成退化 | 定义可控FN/FP/分裂合并 | 鲁棒性，目前仍待正式验证 |

训练与推理的检测来源、实例化参数及encoder必须一致。节点不得含GT身份编码；映射只用作标签、对齐和诊断。marker使用均匀质量/关闭体积守恒，真实区域使用体积质量；策略差异必须写入报告。

结论至少需要双序列；模块有效性需至少两个独立重训种子。每种新设置进行同权重完整重复，分别列重复差和种子波动，不把两者混为一类噪声。历史噪声地板0.0007仅适用于旧设置；当前比较分辨率至少0.001，并取当前重复差、种子全距中的更大值。ΔTRA低于分辨率只能报告不可区分。

候选召回至少分列移动、分裂、桥接，报告各自分母；总体高召回不能保证分裂安全。标定候选与实际部署候选也需分开审计。AOGM六项NS/FN/FP/ED/EA/EC帮助归因，错误方向应与判据匹配。

现有GNN仅seq01训练，两种子各60轮，seq02用于跨序列评测；既有nnU-Net训练含158个seq01和150个seq02帧，不能声称端到端盲测。E0.8配置选择也查看了seq02。Oracle使用GT种子，额外不满足盲测条件。

尚未进行同协议手工特征对照，因此当前Oracle与历史无encoder结果的差异不能解释为encoder独立收益。详细结果见 [正式报告](../README.md) 与 [实验记录](../../experiments/RECORDS.md)。
''')
    write('docs/guides/experiments.md', '''
# 实验记录与历史恢复

任务入口统一为 [INDEX](../../experiments/INDEX.md)，阶段叙述统一为 [RECORDS](../../experiments/RECORDS.md)。原始官方结果的来源和散列登记在 `experiments/ledger.json`；它是证据清单，不能把重复拷贝计为独立实验。

新实验创建独立目录，保留config、command、env、git_commit、metrics、logs、figures、notes八件套；已有正式指标不覆盖。当前阶段以 `metrics_final.json` 为最终数字，历史执行日志中的绝对路径属于当时工作环境。

此次按用户要求一次性重建旧索引并合并重复文档；以后新增结果继续追加独立记录。已有正式指标、官方日志与小权重保持字节不变，校验见 `retention_manifest.json`；整理过程、迁移清单及测试见E1.1目录。

历史错误、偏离、中断和被替代实验完整压缩保存，没有抹去失败证据。压缩包逐文件校验通过才清理原件，包清单、SHA256、源commit和恢复命令见 [archives](../../experiments/archives/README.md)。

大型压缩包仅在本地保存，Git保留目录、散列和恢复说明。克隆后如需原始未入库文件，必须另行拷贝对应压缩包；旧提交只能恢复曾经入库的部分。原始数据与nnU-Net大权重仍使用既有数据备份。

中断恢复先查看git log/status、阶段目录和云端日志，再判断续跑；不要根据记忆重复训练。正式小权重是推理副本，不能恢复优化器；继续训练使用原云端完整断点。
''')
    write('docs/decisions/0003-evaluation-protocol.md', '''
# 0003：统一当前评测协议

日期：2026-10-09。此文件合并旧seq02协议的有效约束，原全文保存在历史归档。

决定：当前GNN仅seq01训练/标定，同一权重分别评测完整seq01与seq02；至少两个独立种子并补一次完整重复。官方指标与原始日志保持原样。检测来源分别报告，Oracle不当作部署结果，marker的SEG不用于分割结论。

依据：旧的in-sequence结果与当前跨序列结果不可直接混用；分割器曾使用seq02，GNN跨序列不等于端到端盲测。当前没有同协议手工特征对照，不声称encoder独立收益。完整执行约束集中在 [评测说明](../guides/evaluation.md)。
''')
    write('docs/history/README.md', '''
# 历史文档与复盘

原实验计划、逐日总结、旧代码映射、测试手册、论文核对长表和审计报告存在重复且描述不同版本。它们已完整归档，当前职责分别集中于status、architecture、guides和正式报告。逐文件迁移见 [document_migration.json](document_migration.json)。

保留的教训：

- 论文公式、模块顺序和训练输入来源需在训练前审计；旧λ加权融合/三分类追踪不能替代当前论文二分类实现。
- 物理spacing要覆盖EDT、代价和门限；修复后排查同族量，不仅修一个调用。
- 总体候选召回会掩盖分裂损失；欠分割指标可能奖励错误方向，判据需匹配失败模式。
- 单序列、单种子或噪声内改进不足以支持模块有效性。旧版FGW/运动等探索不代表新版已完成消融。
- 在原检测/原特征上训练的权重不能用于新来源；Oracle种子身份转换曾合并相邻GT标签，已修复并进入输入契约。
- 孤立性删除会增加FN；当前真实检测预设保留孤立实例，不能照搬论文文字的有效性假设。

旧文档原件和被删除脚本位于 [完整归档](../../experiments/archives/README.md)，原Git提交为426e41b。历史数值按原协议保存，不能挪作当前性能。
''')
    write('experiments/INDEX.md', '''
# 按任务进度整理的实验索引

当前阶段先读 [合并记录](RECORDS.md)；原始官方数值清单见 [ledger](ledger.json)。旧索引的完整原件在00归档，已按用户要求合并重排。

| 顺序 | 阶段 | 状态/用途 | 日常证据入口 |
| --- | --- | --- | --- |
| 1 | 仓库、数据、指标自校验 | 历史基础，保留摘要 | [E0.1](E0.1_repo_skeleton/notes.md)、[E0.2](E0.2_data_stats/notes.md)、[E0.3](E0.3_eval_selfcheck/notes.md) |
| 2 | 匈牙利/greedy基线、冻结nnU-Net训练 | 重要历史参照，完整记录归档 | [阶段记录](RECORDS.md)、[01归档](archives/README.md) |
| 3 | 早期A/B与E2–E4、P追踪 | 旧实现、偏离及探索，已合并归档 | [02归档](archives/README.md) |
| 4 | 早期C前端/真实检测/Oracle | 历史前端机制证据，已合并归档 | [03归档](archives/README.md) |
| 5 | E0.4–E0.6审计、重建、本地/云端探索 | 有已修复bug和被替代结果，不能作为新版消融 | [04归档](archives/README.md) |
| 6 | 冻结前端、encoder、检测映射 | 完成准备，无新增UNet训练 | [E0.7](E0.7_ideas_frontend_20261009/metrics_preparation_finished.json) |
| 7 | 真实预测实例正式双序列×双种子 | 当前可部署档，20次官方评测 | [E0.8最终指标](E0.8_ideas_final_v3_20261009/metrics_final.json)、[报告](../docs/reports/ideas_pipeline_rebuild_20261009.md) |
| 8 | GT marker Oracle + encoder | 隔离追踪贡献，5次官方评测 | [E0.9最终指标](E0.9_gt_oracle_encoder_20261009/metrics_final.json)、[报告](../docs/reports/gt_oracle_encoder_20261009.md) |
| 9 | NN掩码 + GT种子Oracle + encoder | 前端归属诊断，5次官方评测 | [E1.0最终指标](E1.0_nnunet_oracle_encoder_20261009/metrics_final.json)、[报告](../docs/reports/nnunet_oracle_encoder_20261009.md) |
| 10 | 项目文件整理 | 工程验收，无新增性能实验 | [E1.1](E1.1_repository_cleanup_20261009/notes.md) |

后续任务见 [项目进度](../docs/status.md)。正式证据只保留关键配置、模型指纹、官方日志、图和诊断；完整原件在05归档。所有本地压缩包、原commit和恢复方式见 [归档说明](archives/README.md)。
''')
    write('experiments/RECORDS.md', '''
# 项目实验合并记录

此文件按任务推进顺序记录结论及证据；逐项数字仍在原始JSON/日志，避免维护多份互相矛盾的实验总表。

## 1. 基础、早期实现及前端探索

E0.1–E0.3完成仓库、数据统计及评测自校验；早期匈牙利基线seq01/02 TRA为0.995594/0.995052。E5.1完成现有nnU-Net语义分割训练，验证Dice约0.9605；该值不代表实例可分离性。

E2–E4及P系列包含纯OT、阈值、旧GNN/融合/上下文等探索；A/B按当时口径重搭框架与消融，B1 marker档TRA为0.996513/0.996369。它们的实现、质量/阈值、监督与特征协议不同于当前版本，数值只作历史参照。旧版模块有效性/无效性结论须在新版本重验。

C系列确定语义前景粘连/实例归属瓶颈，包含spacing、假分裂标签、孤立过滤、实例再切等归因。旧C4真实档0.905706/0.909340、k=1.6再切0.926890/0.926320；旧C4O仅seq01、TRA0.984784，使用不同Oracle过滤和种子转换。不能把这些数字当作当前无encoder对照。

完整历史已按基础/追踪/前端分为01/02/03压缩包；[ledger](ledger.json)保存查得的原始官方结果来源与SHA256。

## 2. 论文重建和被替代结果

E0.4审计后，E0.5/E0.6贯通完整论文链路，修复求解目标、物理单位、多尺度梯度/线搜索、上下文连通、二分类BCE、来源契约等。v2云端筛查存在编号0伪运动和实例再切背景污染等已修复问题，其核心模块筛查不能支持v3的结论。中断、失败、偏离和被替代记录一并归档到04，原日志未抹去。

## 3. 冻结前端与真实预测正式结果

E0.7完成385帧（195/190）检测映射及冻结stage2 encoder特征，nnU-Net权重保持不变。

E0.8按“原实例/再切k=1.6 × 孤立过滤开/关”四档，每档独立训练两个种子，共8次训练、20次官方评测（含4次完整重复）。推荐保留孤立实例+k=1.6：

| 序列 | DET | SEG | TRA均值 | TRA两种子范围 |
| --- | --- | --- | --- | --- |
| 01 | 0.955785 | 0.663599 | 0.9325185 | 0.932366–0.932671 |
| 02 | 0.951608 | 0.667235 | 0.9308145 | 0.930680–0.930949 |

与相同后端原实例相比，TRA提升0.0233165/0.0182925，SEG下降0.009597/0.023632；NS由2199/1506降到1265/759。当前结论是这组前端和孤立保留设置下的全链路表现，不能解释为某个OT扩展独立有效。当前真实档分裂候选安全仍需实际部署图复核。

## 4. 两类Oracle + encoder

E0.9为GT marker位置/掩码 + 冻结encoder，排除检测误差；E1.0为NN前景 + GT标签种子，前景逐体素不变，保留未匹配连通域且min_volume=1。各自独立训练2种子，双序列官方评测并重复seq02，共各5次官方评测。

| 档位 | seq01 TRA均值 | seq02 TRA均值 | 边界 |
| --- | --- | --- | --- |
| GT marker Oracle | 0.9979770 | 0.9976020 | DET恒1，SEG不适用，质量均匀 |
| NN掩码+GT种子Oracle | 0.9960460 | 0.9907585 | DET0.998328/0.993644，SEG0.683938/0.698012，质量按体积 |

Oracle种子转换修复相邻不同GT身份被合并的问题，并纳入契约。两档候选总体/分裂独立核验，marker top-k=3保留716/716分裂边；NN Oracle seq01保留704/704分裂，seq02只读核验704/709。官方JSON、AOGM日志和12份当前推理权重保持原字节。

GT辅助不属于部署方法；现有分割训练见过部分seq02；没有同协议手工特征对照，不归因encoder收益。历史C4O与E1.0间差异包含种子转换、体积过滤和追踪版本，不能称为encoder消融。

## 5. 整理与后续工作

E1.1合并源码根、移除vendor和旧入口，文档分类去重；原始全部实验先完整归档校验，再保留精简的正式证据。未启动训练、云任务或修改方法。整理后的测试和逐项保持校验见E1.1。

后续先做真实档部署候选漏斗，再做当前版本核心模块的双种子消融和鲁棒性验证；前端实例感知训练另行决策。任务依赖与限制集中于 [项目进度](../docs/status.md)，实验入口见 [INDEX](INDEX.md)。
''')
    write('experiments/archives/README.md', '''
# 完整历史归档

此次按用户要求压缩合并历史实验。每个压缩包包含逐文件SHA256、字节数、权限、原始路径和源commit的内嵌清单；打包并逐项校验成功后才清理源目录。已保留的正式文件校验见 [retention_manifest](../retention_manifest.json)。

| 包 | 内容 |
| --- | --- |
| 00_source_and_documents.tar.gz | 整理前src/paperpipe/scripts/configs/docs/tests、README和原INDEX |
| 01_foundations.tar.gz | 基础、数据统计、指标自校验、经典基线与nnU-Net训练记录 |
| 02_historical_tracking.tar.gz | 早期E2–E4、A/B、P追踪与消融 |
| 03_historical_frontend.tar.gz | C阶段前端、旧真实检测与旧Oracle |
| 04_rebuild_audit_and_failed_runs.tar.gz | E0.4–E0.6审计/重建/被替代与中断探索，含未入库中间产物 |
| 05_formal_evidence_complete.tar.gz | E0.7–E1.0正式阶段整理前的全部证据 |

每包散列、大小、文件数、来源目录与commit见 [catalog.json](catalog.json)。大压缩包保存在本地且不入Git；若要迁移机器，应额外复制该目录下的压缩包。源码旧提交能恢复曾入库文件，不能替代对未入库文件的压缩包备份。

```bash
python scripts/manage_archives.py verify experiments/archives/04_rebuild_audit_and_failed_runs.tar.gz
python scripts/manage_archives.py restore experiments/archives/04_rebuild_audit_and_failed_runs.tar.gz \
  --out /tmp/celltracker_history_restore
```

恢复目标必须是新目录，验证失败不会解包；不覆盖当前文件。恢复后维持原仓库相对路径。若本地压缩包不可用，可从整理前提交426e41b提取原先入库的文件：

```bash
mkdir -p /tmp/celltracker_git_history
git archive 426e41b experiments docs paperpipe scripts | tar -x -C /tmp/celltracker_git_history
```

归档与删除的工程证据见 [E1.1](../E1.1_repository_cleanup_20261009/notes.md)。
''')
    write('scripts/README.md', '''
# 脚本职责

| 分类 | 脚本 | 用途 |
| --- | --- | --- |
| 主链路 | `run_pipeline.py` | 统一论文推理/监督建图/CTC导出/官方评测 |
| 学习 | `train_gnn.py` | 配置驱动的论文GNN训练与恢复 |
| 消融 | `run_ablation_matrix.py` | 双序列多种子计划、独立训练与恢复 |
| 标定 | `calibrate_params.py` | 完整上游的绝对阈值、物理分布与候选损失 |
| 实例前端 | `predict_to_h5.py`、`relabel_detections.py`、`resplit_detections.py`、`calibrate_instance_split.py` | 流式实例化、GT映射、再切与拆分诊断 |
| 冻结分割/特征 | `nnunet/` | 数据转换、冻结推理、encoder侧车、训练入口；重训另行确认 |
| 官方工具 | `cloud_eval.py`、`analyze_tra_log.py` | 官方评测与AOGM分解 |
| 云端连接 | `cloud_run.py`、`cloud_run.sh` | 执行与传输，凭据只从本地文档读取 |
| 对照 | `run_baseline.py` | 匈牙利/greedy历史基线 |
| 历史诊断 | `diagnostics/detection_ceiling.py` | 早期体素门限、单GT映射诊断；仅回归/历史用途，不定当前结论 |
| 数据/留痕 | `download_data.sh`、`pdownload.sh`、`new_experiment.sh`、`record_env.sh` | 已有数据获取与八件套辅助 |
| 归档 | `manage_archives.py` | 打包、逐文件校验及恢复到新目录 |

已删除的旧编排器、旧GNN/OT参数扫描、一次性作图、独立帧对旧候选扫描及旧paperpipe打包脚本，原件均在00归档。删除/迁移清单见 `experiments/E1.1_repository_cleanup_20261009/artifacts/source_migration.json`。实验内的一次性执行脚本同样完整归档，不再充当日常入口。
''')


if __name__ == '__main__':
    main()
