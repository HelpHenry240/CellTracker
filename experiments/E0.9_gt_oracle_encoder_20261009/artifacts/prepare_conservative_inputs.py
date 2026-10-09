"""引用已经核验的GT检测和冻结encoder，保留首次提取的GPU与身份校验证据。"""
from pathlib import Path
import json
import shutil
import hashlib

old=Path('/root/CellTracker_oracle_encoder_20261009')
new=Path('/root/CellTracker_oracle_encoder_conservative_20261009')
original=Path('/root/autodl-tmp/CellTracker_rebuild_20261008/data/interim')
(new/'logs').mkdir(exist_ok=True)
(new/'data/interim').mkdir(parents=True,exist_ok=True)
records=[]
for seq in ('01','02'):
    sources=[original/f'Fluo-N3DH-CE_{seq}.h5',
             old/f'data/interim/encoder_{seq}_gt_marker_20261009.npz',
             old/f'data/interim/encoder_{seq}_gt_marker_20261009.json']
    for source in sources:
        dest=new/'data/interim'/source.name
        if dest.exists():
            raise FileExistsError(dest)
        dest.symlink_to(source)
        records.append({'source':str(source),'reference':str(dest),'bytes':source.stat().st_size})
shutil.copy2(old/'logs/oracle_input_audit.json',new/'logs/oracle_input_audit.json')
for source in old.glob('logs/oracle_encoder_*'):
    if source.is_file() and source.suffix in {'.txt','.json','.log'}:
        shutil.copy2(source,new/'logs'/('original_'+source.name))
shutil.copy2(old/'logs/oracle_topk_safety.json',new/'logs/oracle_topk_safety.json')
(new/'logs/frozen_encoder_reuse.json').write_text(json.dumps({
    'reason':'GT检测和encoder未变，仅候选图配置变化；不复用初版GNN权重',
    'original_source_release':json.loads((old/'SOURCE_VERSION.json').read_text()),
    'references':records},ensure_ascii=False,indent=2))
# 状态快照只追加；初版矩阵不得被恢复成正式结果。
snapshot=old/'logs/oracle_v1_superseded_status.json'
if not snapshot.exists():
    state=json.loads((old/'experiments/P8_gt_marker_encoder_matrix_20261009/status.json').read_text())
    snapshot.write_text(json.dumps({'status_snapshot':state,'cancelled_pipeline':True,
        'reason':'总体召回掩盖分裂边损失，切换已有top-k=3后重新建图训练',
        'results_role':'diagnostic only; not final Oracle protocol'},ensure_ascii=False,indent=2))
print(json.dumps({'inputs_prepared':len(records),'new_root':str(new)}))
