"""在全部真实帧上核对显式排除自身的修复，是否改变本次基准使用的邻域。"""
from pathlib import Path
import argparse
import json
import sys
import h5py
import numpy as np


def verify(root):
    root = Path(root)
    sys.path.insert(0, str(root / 'paperpipe/src'))
    import papertrack
    from celltracker.cost.features import pairwise_distance
    reports = {}
    for seq in ('01', '02'):
        for source in ('rebuild', 'resplit_v2'):
            path = root / 'data/interim' / f'Fluo-N3DH-CE_{seq}_{source}.h5'
            comparisons = mismatches = duplicate_nodes = 0
            with h5py.File(path) as file:
                spacing = file.attrs['spacing_zyx']
                for frame in file['frames'].values():
                    xy = frame['centroid'][:]
                    duplicate_nodes += len(xy) - len(np.unique(xy, axis=0))
                    distance = pairwise_distance(xy, xy, spacing)
                    order = np.argsort(distance, axis=1)
                    fixed = order[order != np.arange(len(xy))[:, None]].reshape(len(xy), max(len(xy) - 1, 0))
                    for neighbors in (4, 6):
                        count = min(neighbors, max(len(xy) - 1, 0))
                        comparisons += 1
                        mismatches += int(not np.array_equal(order[:, 1:count + 1], fixed[:, :count]))
            reports[f'{seq}_{source}'] = {'duplicate_centroids': duplicate_nodes,
                'frame_neighbor_comparisons': comparisons, 'different_neighbor_sets_or_orders': mismatches}
            if mismatches:
                raise RuntimeError('当前真实数据受 kNN 修复影响；不得复用修复前的性能证据')
    return reports


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    report = verify(args.root)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
