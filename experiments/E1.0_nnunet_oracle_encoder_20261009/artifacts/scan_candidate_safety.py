"""仅用seq01真实Oracle实例审计已有top-k保底，不使用官方分数选参。"""
from pathlib import Path
from dataclasses import replace
import argparse,json,sys


def scan(root):
    import h5py,numpy as np
    root=Path(root);sys.path.insert(0,str(root/'paperpipe/src'))
    from papertrack.config import load_config
    from papertrack.coupling.pairwise import load_coupling
    from papertrack.graph.build import _candidates
    matrix=root/'experiments/P9_nnunet_oracle_encoder_matrix_20261009'
    cfg=load_config(matrix/'configs/baseline_calibrated.yaml')
    counts={k:dict(true_pairs=0,retained_true=0,division_pairs=0,retained_division=0,candidate_pairs=0) for k in (0,3,4,5,6,8,12)}
    with h5py.File(root/'data/interim/Fluo-N3DH-CE_01.h5') as truth,h5py.File(root/'data/interim/Fluo-N3DH-CE_01_nnunet_oracle.h5') as h5,np.load(root/'experiments/P9_nnunet_oracle_encoder_baseline_build01/artifacts/pipeline/couplings.npz') as plans:
        parent={int(r['label']):int(r['parent']) for r in truth['tracks'][:]};keys=sorted(h5['frames'])
        for pos,(a,b) in enumerate(zip(keys[:-1],keys[1:])):
            coupling=load_coupling(matrix/f'cache/baseline/01/adjacent_{int(a):04d}.npz');coupling.plan=plans[f'plan_{pos}']
            source={int(label):i for i,ids in enumerate(h5['frames'][a]['gt_ids'][:]) for label in ids if label>0}
            pairs=[]
            for j,ids in enumerate(h5['frames'][b]['gt_ids'][:]):
                for label in ids:
                    if label<=0:continue
                    label=int(label);previous=label if label in source else parent.get(label,0)
                    if previous in source:pairs.append((source[previous],j,previous!=label))
            for k,r in counts.items():
                edges,_,_=_candidates(coupling,replace(cfg.graph,cand_topk=k),cfg.coupling.r_max);accepted={tuple(e) for e in edges.tolist()};r['candidate_pairs']+=len(edges)
                for i,j,division in pairs:
                    r['true_pairs']+=1;r['division_pairs']+=int(division);r['retained_true']+=int((i,j) in accepted);r['retained_division']+=int(division and (i,j) in accepted)
    for r in counts.values():
        r['true_recall']=r['retained_true']/r['true_pairs'];r['division_recall']=r['retained_division']/r['division_pairs']
    safe=[k for k in (3,4,5,6,8,12) if counts[k]['true_recall']>=.99 and counts[k]['division_recall']>=.99]
    return {'counts':counts,'selected_topk':min(safe) if safe else None,'selection':'existing top-k, seq01 candidate safety only; frozen before any formal training'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args();result=scan(a.root);Path(a.out).write_text(json.dumps(result,indent=2));print(json.dumps(result))
