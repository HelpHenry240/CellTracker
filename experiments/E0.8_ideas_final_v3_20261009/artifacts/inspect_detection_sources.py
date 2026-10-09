"""逐帧读取实例表核对两种前端的 GT 身份覆盖，避免实例总数掩盖欠分割。"""
from pathlib import Path
import argparse
import json
import h5py
import numpy as np


def inspect(root):
    root = Path(root)
    report = {}
    for seq in ('01', '02'):
        for source in ('rebuild', 'resplit_v2'):
            path = root / 'data/interim' / f'Fluo-N3DH-CE_{seq}_{source}.h5'
            totals = {'gt_nodes': 0, 'pred_nodes': 0, 'covered_gt_nodes': 0,
                'unmatched_pred_nodes': 0, 'multi_identity_pred_nodes': 0, 'merged_identity_excess': 0}
            rows = []
            with h5py.File(root / f'data/interim/Fluo-N3DH-CE_{seq}.h5') as truth, h5py.File(path) as predicted:
                for key in sorted(predicted['frames']):
                    ids = predicted[f'frames/{key}/gt_ids'][:]
                    expected = set(truth[f'frames/{key}/label'][:].tolist()) - {0}
                    observed = set(ids[ids > 0].tolist())
                    counts = (ids > 0).sum(axis=1)
                    row = {'frame': int(key), 'gt_nodes': len(expected), 'pred_nodes': len(ids),
                        'covered_gt_nodes': len(expected & observed), 'unmatched_pred_nodes': int((counts == 0).sum()),
                        'multi_identity_pred_nodes': int((counts > 1).sum()),
                        'merged_identity_excess': int(np.maximum(counts - 1, 0).sum())}
                    for name in totals:
                        totals[name] += row[name]
                    rows.append(row)
            totals['missing_gt_nodes'] = totals['gt_nodes'] - totals['covered_gt_nodes']
            totals['gt_identity_coverage'] = totals['covered_gt_nodes'] / totals['gt_nodes']
            report[f'{seq}_{source}'] = {'totals': totals, 'frames': rows,
                'definition': 'complete GT marker majority overlap; diagnostic, not official DET'}
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    report = inspect(args.root)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({name: value['totals'] for name, value in report.items()}, indent=2))
