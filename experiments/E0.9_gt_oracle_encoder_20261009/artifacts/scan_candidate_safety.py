"""在seq01实际部署耦合上量化已有top-k保底，按父子边单独检查不可逆损失。"""
from pathlib import Path
from dataclasses import replace
import argparse
import json
import sys


def scan(root):
    import h5py
    import numpy as np
    root = Path(root)
    sys.path.insert(0, str(root / 'paperpipe/src'))
    from papertrack.config import load_config
    from papertrack.coupling.pairwise import load_coupling
    from papertrack.graph.build import _candidates
    matrix = root / 'experiments/P8_gt_marker_encoder_matrix_20261009'
    cfg = load_config(matrix / 'configs/baseline_calibrated.yaml')
    counts = {k: {'true_pairs': 0, 'retained_true': 0, 'division_pairs': 0,
                  'retained_division': 0, 'candidate_pairs': 0} for k in (0, 1, 2, 3, 4, 5)}
    with h5py.File(root / 'data/interim/Fluo-N3DH-CE_01.h5') as h5, np.load(
            root / 'experiments/P8_gt_marker_encoder_baseline_build01/artifacts/pipeline/couplings.npz') as plans:
        parents = {int(row['label']): int(row['parent']) for row in h5['tracks'][:]}
        keys = sorted(h5['frames'])
        for pos, (a, b) in enumerate(zip(keys[:-1], keys[1:])):
            coupling = load_coupling(matrix / f'cache/baseline/01/adjacent_{int(a):04d}.npz')
            coupling.plan = plans[f'plan_{pos}']
            source = {int(label): i for i, label in enumerate(h5['frames'][a]['label'][:])}
            truth = []
            for j, label in enumerate(h5['frames'][b]['label'][:]):
                label = int(label)
                predecessor = label if label in source else parents.get(label, 0)
                if predecessor in source:
                    truth.append((source[predecessor], j, predecessor != label))
            for k, report in counts.items():
                edges, _, _ = _candidates(coupling, replace(cfg.graph, cand_topk=k), cfg.coupling.r_max)
                accepted = {tuple(edge) for edge in edges.tolist()}
                report['candidate_pairs'] += len(edges)
                for i, j, division in truth:
                    report['true_pairs'] += 1
                    report['division_pairs'] += int(division)
                    report['retained_true'] += int((i, j) in accepted)
                    report['retained_division'] += int(division and (i, j) in accepted)
    audit = json.loads((matrix / 'training_graph_audit.json').read_text())
    if counts[0]['retained_true'] != audit['positive_edges'] or counts[0]['retained_division'] != audit['division_positive']:
        raise RuntimeError('扫描耦合与实际部署训练图不一致')
    for report in counts.values():
        report['true_recall'] = report['retained_true'] / report['true_pairs']
        report['division_recall'] = report['retained_division'] / report['division_pairs']
    safe = [k for k in (3, 4, 5) if counts[k]['true_recall'] >= .99 and counts[k]['division_recall'] >= .99]
    if not safe:
        raise RuntimeError('已有top-k保底仍不足；检查具体真实边，禁止开始新训练')
    result = {'protocol': 'seq01 training input only; existing top-k switch; actual post-multiscale couplings',
              'counts': counts, 'selected_topk': min(safe),
              'selection': 'start at existing engineering default3, increase only if division retention requires it'}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    result = scan(args.root)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
