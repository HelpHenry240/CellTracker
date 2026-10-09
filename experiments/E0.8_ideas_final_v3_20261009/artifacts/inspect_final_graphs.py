"""只读检查部署阈值下实际训练图的候选边与代价，不把标定阶段的代理数当作最终图。"""
from pathlib import Path
import argparse
import json
import numpy as np


def inspect(root):
    reports = {}
    for matrix in sorted((Path(root) / 'experiments').glob('P7_ideas_v3_*matrix*')):
        count = {'adjacent_candidates': 0, 'adjacent_positive': 0,
            'division_positive': 0, 'bridge_candidates': 0, 'bridge_positive': 0}
        costs = []
        for path in sorted((matrix / 'graphs/baseline').glob('pair_*.npz')):
            with np.load(path) as graph:
                adjacent = graph['cand_gap'] == 1
                positive = graph['label'] > 0
                count['adjacent_candidates'] += int(adjacent.sum())
                count['adjacent_positive'] += int((adjacent & positive).sum())
                count['division_positive'] += int(graph['cand_label_div'][adjacent].sum())
                count['bridge_candidates'] += int((~adjacent).sum())
                count['bridge_positive'] += int((~adjacent & positive).sum())
                costs.append(graph['cand_feat'][positive, 0])  # 当前命名布局第 0 列是原始 C。
        cost = np.concatenate(costs) if costs else np.empty(0)
        count['graphs'] = len(costs)
        count['positive_C_percentiles'] = dict(zip(('p50', 'p95', 'p99', 'p99.9', 'max'),
            map(float, np.percentile(cost, [50, 95, 99, 99.9, 100])))) if len(cost) else None
        reports[matrix.name] = count
    return reports


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    result = inspect(args.root)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
