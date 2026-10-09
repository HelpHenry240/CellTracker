"""将完整双种子官方证据写成可复核报告；无手工特征对照时不推断encoder收益。"""
from pathlib import Path
import json
import statistics

base=Path(__file__).resolve().parents[1]
raw=base/'artifacts/evidence_formal/logs/oracle_metrics_final.json'
result=json.loads(raw.read_text())
if result['jobs_complete']!=9 or len(result['runs'])!=5:
    raise RuntimeError('正式实验未全部完成')
for row in result['runs']:
    if row['DET']!=1 or not row['format_validation']['ok']:
        raise RuntimeError('Oracle检测或格式验收失败')
    row['AOGM_weighted']=sum(row['AOGM'][k]*w for k,w in {'NS':5,'FN':10,'FP':1,'ED':1,'EA':1.5,'EC':1}.items())
repeat=next(row for row in result['runs'] if row['repeat'])
original=next(row for row in result['runs'] if row['seq']=='02' and row['seed']==20261008 and not row['repeat'])
result['repeat_AOGM_absolute_difference']={key:abs(repeat['AOGM'][key]-original['AOGM'][key]) for key in original['AOGM']}
for seq in ('01','02'):
    scores=[row['TRA'] for row in result['runs'] if row['seq']==seq and not row['repeat']]
    result['summary'][seq]['TRA']['sample_std']=statistics.stdev(scores)
    result['summary'][seq]['TRA']['half_range']=(max(scores)-min(scores))/2
frozen=json.loads((base/'artifacts/evidence_formal/logs/oracle_frozen_model_identity.json').read_text())
if not frozen['unchanged'] or frozen['before']!=frozen['after']:
    raise RuntimeError('冻结encoder模型身份校验未通过')
result['frozen_model_identity']=frozen
result['candidate_topk']=3
result['training_graph_audit']=json.loads((base/'artifacts/evidence_formal/experiments/P8_gt_marker_encoder_conservative_matrix_20261009/training_graph_audit.json').read_text())
result['verification']={'local_pytest':{'passed':151,'skipped':0},'cloud_clean_archive_pytest':{'passed':149,'skipped':2},
    'all_official_binaries_identical':all(row['binary_sha256']==result['runs'][0]['binary_sha256'] for row in result['runs']),
    'GT_marker_frames_pixel_equal':385,'handcrafted_control':False,'unet_training':False}
output=base/'metrics_final.json'
if output.exists():raise FileExistsError(output)
output.write_text(json.dumps(result,ensure_ascii=False,indent=2))
lines=['# GT marker Oracle + 冻结 encoder：双序列、双种子官方验证','',
'本轮已完成GT marker输入的冻结encoder特征提取、两个独立GNN训练、双序列官方评测和一次完整seq02重复。',
'没有加入手工特征对照，也没有重训nnU-Net。结果衡量当前pipeline在排除检测误差后的追踪表现。','',
'## 实验协议','',
'GT Oracle指官方TRA marker的位置与标记掩码，不是nnU-Net前景内用GT种子拆分的实例Oracle，也不是完整细胞分割。',
'原检测H5共385帧（seq01/02为195/190帧、23802/22418个节点），逐帧与官方TRA标记像素一致。',
'使用既有nnU-Net fold0的checkpoint_best，冻结stage2 encoder；从原图在GT质心采样128维外观。',
'GT身份仅用于特征侧车对齐及边监督，不作为身份编码输入模型。实际节点133维=质心3+尺寸1+encoder128+归一化时间1。','',
'候选/生死阈值标定及GNN训练只使用seq01；种子20261008和20261009各训练60轮，内部按时间块划分训练/验证。',
'每个权重独立评测完整seq01和seq02，再用seed20261008权重重复完整seq02推理、导出与官方评测。',
'固定uniform质量，不做marker体积守恒筛选，沿用决策0001；保留孤立检测，空洞使用split，不补画新检测。',
'FGW、运动、多尺度、同帧图、时序上下文、GNN和二层tracklet按现有全模块配置运行。','',
'## 候选安全审计与修复','',
'初版仅检查总体候选召回：部署图保留23757/23798真实关联，但只保留679/716分裂边（94.83%）。',
'这个隐性问题会被占多数的移动边掩盖。已新增少数类独立标定警告和回归测试，',
'并在本轮训练前分别检查标定和实际部署图的总体、分裂保留率，任一低于99%即阻断。','',
'仅在seq01实际部署耦合上扫描已有top-k开关（0..5），没有根据官方TRA选参。',
'使用已有工程保底top-k=3：23797/23798真实关联（99.9958%）、716/716分裂（100%）；',
'194张实际训练图，跨空洞候选0。初版已精确PID取消，权重、已完成指标及日志保留为诊断，',
'正式配置重新建图、独立训练两个种子，不复用初版GNN权重。','',
'物理spacing=[1,0.09,0.09]µm；r_max=6.4µm覆盖seq01最大真实位移5.30224µm。',
'λ_OT=0.001；标定真实C_p99.9=38.27491，λ_OT×C_p99.9=0.038275；',
'实际训练正边C_p99.9=34.80925、max=75.23230，确认OT正则尺度不会压倒正边监督。','',
'## 官方结果','',
'| 种子 | 序列 | DET | TRA | 轨迹数 | 加权AOGM |','| --- | --- | --- | --- | --- | --- |']
for row in result['runs']:
    label=str(row['seed'])+('（重复）' if row['repeat'] else '')
    lines.append(f"| {label} | {row['seq']} | {row['DET']:.6f} | {row['TRA']:.6f} | {row['format_validation']['n_tracks']} | {row['AOGM_weighted']:.1f} |")
lines+=['','| 序列 | TRA均值 | 两种子范围 | 样本标准差（n=2） |','| --- | --- | --- | --- |']
for seq in ('01','02'):
    s=result['summary'][seq]['TRA'];lines.append(f"| {seq} | {s['mean']:.7f} | {s['min']:.6f}–{s['max']:.6f} | {s['sample_std']:.7f} |")
noise=result['repeat_absolute_difference']['TRA']
lines+=['',f'完整seq02重复的TRA差={noise:.6f}（官方六位小数精度）；两种子波动另列，二者不是同一种噪声。',
    f"本轮比较分辨率取max(0.001,重复差,种子全距)={result['comparison_resolution_TRA']:.6f}；两次重复不足以证明所有设置零噪声。",
    '全部5次提交格式通过，无幽灵轨迹或空洞；GT输入下DET恒1。',
    '官方SEG原始输出0已保留，但marker不是细胞分割，不能把该值解释成分割能力。','',
    '## AOGM归属','',
    '| 种子 | 序列 | NS | FN | FP | ED | EA | EC |','| --- | --- | --- | --- | --- | --- | --- | --- |']
for row in result['runs']:
    if not row['repeat']:lines.append('| '+str(row['seed'])+' | '+row['seq']+' | '+' | '.join(str(row['AOGM'][k]) for k in ('NS','FN','FP','ED','EA','EC'))+' |')
if all(all(row['AOGM'][key]==0 for key in ('NS','FN','FP')) for row in result['runs']):
    lines+=['','全部运行的检测错误项NS/FN/FP均为0；残余失分来自边冗余、边缺失与边语义（ED/EA/EC）。']
lines+=['','当前结果支持：GT检测来源下，该pipeline可以完成高精度追踪并在GNN未训练过的seq02上运行。',
'本轮没有同版本、同协议的手工特征对照，不能归因出encoder本身的提升或下降。',
'历史GT数值来自不同实现与参数，不作为本轮特征消融基准；初版top-k=0也只作为候选安全诊断。',
'既有nnU-Net训练包含150/190个seq02帧；seq02只对GNN保持跨序列，不能声称端到端盲测泛化。','',
'## 复核入口','',
'本地151项测试通过；干净源码云端149通过、2跳过（缺POT参考包、源码归档没有Git索引）。',
'正式训练来源500322cc01433477cea866654faa035d1d617398，190项文件散列校验通过；',
'源包SHA256=cd3cceff56ebac66e6f78cbe420e30dfea53d516e8c2c0fd646c256bf7f9401e。',
'首次GT encoder提取来源f35e123；上游检测与冻结模型散列校验通过后正式阶段复用特征。',
'冻结nnU-Net前后SHA256均为'+frozen['before']+'。','',
'正式后台PID756340，工作目录/root/CellTracker_oracle_encoder_conservative_20261009；日志位于logs/oracle_encoder_pipeline.log。',
'原体数据、缓存和完整GNN断点保留在云端；两份逐张量校验的小型推理权重及md5/sha256另存八件套。',
'大型运行诊断仅压缩长列表，官方指标与日志原样保留；完整证据归档及每项原文件SHA256可追溯。','',
'- [最终指标](../../experiments/E0.9_gt_oracle_encoder_20261009/metrics_final.json)',
'- [正式配置](../../experiments/E0.9_gt_oracle_encoder_20261009/pipeline_conservative.yaml)',
'- [正式证据](../../experiments/E0.9_gt_oracle_encoder_20261009/artifacts/evidence_formal/archive_manifest.json)',
'- [推理权重指纹](../../experiments/E0.9_gt_oracle_encoder_20261009/artifacts/models/manifest.json)','',
'![双种子官方TRA](../../experiments/E0.9_gt_oracle_encoder_20261009/figures/oracle_official_TRA.png)','',
'![官方误差归属](../../experiments/E0.9_gt_oracle_encoder_20261009/figures/oracle_AOGM.png)','',
'![候选安全漏斗](../../experiments/E0.9_gt_oracle_encoder_20261009/figures/oracle_candidate_safety.png)','',
'后续显式待办：后续候选消融报告继续分列移动/分裂分母；关闭保底时的候选损失不可误归因于GNN。','']
repo=base.parents[1]
(repo/'docs/reports/gt_oracle_encoder_20261009.md').write_text('\n'.join(lines))
print(json.dumps({'report_written':True,'summary':result['summary'],'repeat':result['repeat_absolute_difference']},ensure_ascii=False))
