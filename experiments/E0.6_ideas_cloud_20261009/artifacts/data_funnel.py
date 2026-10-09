"""流式核对检测身份覆盖与训练图候选损失；这些量只作归因。"""
from pathlib import Path
import argparse,json
import h5py
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args()
r=Path(a.root); report={}
for seq in ['01','02']:
    total_gt=total_pred=covered=unmatched=merged=merged_excess=0
    with h5py.File(r/f'data/interim/Fluo-N3DH-CE_{seq}.h5') as truth,h5py.File(r/f'data/interim/Fluo-N3DH-CE_{seq}_rebuild.h5') as predicted:
        for key in predicted['frames']:
            ids=np.asarray(predicted[f'frames/{key}/gt_ids'])
            gt_ids=set(np.asarray(truth[f'frames/{key}/label']).tolist())-{0}
            mapped=set(ids[ids>0].tolist())
            sizes=(ids>0).sum(axis=1)
            total_gt+=len(gt_ids); total_pred+=len(ids); covered+=len(gt_ids&mapped)
            unmatched+=int((sizes==0).sum());merged+=int((sizes>1).sum())
            merged_excess+=int(np.maximum(sizes-1,0).sum())
    report[seq]={'gt_nodes':total_gt,'pred_nodes':total_pred,'covered_gt_nodes':covered,
        'missing_gt_nodes':total_gt-covered,'unmatched_pred_nodes':unmatched,
        'multi_identity_pred_nodes':merged,'merged_identity_excess':merged_excess,
        'gt_coverage':covered/max(total_gt,1),'note':'CTC majority-marker mapping; diagnostics, not official DET'}
for matrix in sorted((r/'experiments').glob('P7_ideas_*matrix*')):
    for graph_dir in sorted((matrix/'graphs').glob('*')):
        totals={'adjacent_candidates':0,'adjacent_positive':0,'division_positive':0,'bridge_candidates':0,'bridge_positive':0}
        for path in sorted(graph_dir.glob('pair_*.npz')):
            with np.load(path) as graph:
                adjacent=graph['cand_gap']==1
                totals['adjacent_candidates']+=int(adjacent.sum())
                totals['adjacent_positive']+=int(graph['label'][adjacent].sum())
                totals['division_positive']+=int(graph['cand_label_div'][adjacent].sum())
                totals['bridge_candidates']+=int((~adjacent).sum())
                totals['bridge_positive']+=int(graph['label'][~adjacent].sum())
        report[f'{matrix.name}/{graph_dir.name}']=totals
Path(a.out).write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
