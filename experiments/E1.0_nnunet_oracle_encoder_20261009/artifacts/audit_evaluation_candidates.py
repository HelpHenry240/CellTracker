"""只读核验正式评测的候选漏斗；seq02结果不用于改变已经冻结的参数。"""
from pathlib import Path
import argparse
import json
import sys


def audit(root):
    import h5py
    import numpy as np

    root = Path(root).resolve()
    sys.path.insert(0, str(root / 'paperpipe/src'))
    from papertrack.config import load_config
    from papertrack.coupling.pairwise import load_coupling
    from papertrack.graph.build import _candidates

    matrix = root / 'experiments/P9_nnunet_oracle_encoder_matrix_20261009'
    cfg = load_config(matrix / 'configs/baseline_calibrated.yaml')
    result = {'protocol': 'read-only diagnostic; frozen seq01-calibrated configuration',
              'r_max_um': cfg.coupling.r_max, 'candidate_topk': cfg.graph.cand_topk,
              'sequences': {}}
    for seq in ('01', '02'):
        evaluation = root / f'experiments/P9_nnunet_oracle_encoder_baseline_seed20261008_eval{seq}'
        counts = dict(true_pairs=0, within_gate=0, retained_true=0,
                      division_pairs=0, retained_division=0, candidate_pairs=0)
        missing = []
        with h5py.File(root / f'data/interim/Fluo-N3DH-CE_{seq}.h5') as truth, \
                h5py.File(root / f'data/interim/Fluo-N3DH-CE_{seq}_nnunet_oracle.h5') as h5, \
                np.load(evaluation / 'artifacts/pipeline/couplings.npz') as plans:
            parent = {int(row['label']): int(row['parent']) for row in truth['tracks'][:]}
            keys = sorted(h5['frames'])
            for pos, (a, b) in enumerate(zip(keys[:-1], keys[1:])):
                coupling = load_coupling(matrix / f'cache/baseline/{seq}/adjacent_{int(a):04d}.npz')
                coupling.plan = plans[f'plan_{pos}']
                edges, _, _ = _candidates(coupling, cfg.graph, cfg.coupling.r_max)
                accepted = {tuple(edge) for edge in edges.tolist()}
                counts['candidate_pairs'] += len(accepted)
                source = {int(label): i for i, ids in enumerate(h5['frames'][a]['gt_ids'][:])
                          for label in ids if label > 0}
                for j, ids in enumerate(h5['frames'][b]['gt_ids'][:]):
                    for label in ids:
                        if label <= 0:
                            continue
                        label = int(label)
                        predecessor = label if label in source else parent.get(label, 0)
                        if predecessor not in source:
                            continue
                        i = source[predecessor]
                        division = predecessor != label
                        retained = (i, j) in accepted
                        counts['true_pairs'] += 1
                        counts['within_gate'] += int(np.isfinite(coupling.cost[i, j]))
                        counts['retained_true'] += int(retained)
                        counts['division_pairs'] += int(division)
                        counts['retained_division'] += int(division and retained)
                        if not retained:
                            missing.append({'source_frame': int(a), 'source_gt': predecessor,
                                            'target_gt': label, 'division': division,
                                            'displacement_um': float(coupling.d_cur[i, j]),
                                            'outside_gate': not bool(np.isfinite(coupling.cost[i, j]))})
        counts['true_recall'] = counts['retained_true'] / counts['true_pairs']
        counts['division_recall'] = counts['retained_division'] / max(counts['division_pairs'], 1)
        result['sequences'][seq] = {'counts': counts, 'missing_true_pairs': missing}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    result = audit(args.root)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({seq: value['counts'] for seq, value in result['sequences'].items()}))
