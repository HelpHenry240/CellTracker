"""由正式官方结果生成可独立导出的图；点为两个种子，误差线为最小值到最大值。"""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NAMES = ('paper_all_k0', 'conservative_k0', 'paper_all_k16', 'conservative_k16')
LABELS = ('Paper / k=0', 'Retain / k=0', 'Paper / k=1.6', 'Retain / k=1.6')
COLORS = ('#536878', '#148b92', '#ba762f', '#4b865c')
WEIGHTS = {'NS': 5, 'FN': 10, 'FP': 1, 'ED': 1, 'EA': 1.5, 'EC': 1}


def save(fig, out, name):
    fig.savefig(out / f'{name}.svg', bbox_inches='tight')
    fig.savefig(out / f'{name}.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot(summary_path, sources_path, output):
    result = json.loads(Path(summary_path).read_text())
    sources = json.loads(Path(sources_path).read_text())
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, seq in zip(axes, ('01', '02')):
        for index, name in enumerate(NAMES):
            runs = [run for run in result['profiles'][name]['runs'] if run['seq'] == seq and not run['repeat']]
            values = np.array([run['TRA'] for run in runs])
            mean = values.mean()
            ax.errorbar(index, mean, yerr=[[mean - values.min()], [values.max() - mean]],
                        fmt='o', color=COLORS[index], capsize=5, markersize=7)
            ax.scatter(index + np.array([-.08, .08]), values, color=COLORS[index], s=18)
        ax.set_title(f'Full sequence {seq} / official TRA')
        ax.set_xticks(range(4), LABELS, rotation=18, ha='right')
        ax.grid(axis='y', alpha=.2)
        ax.ticklabel_format(axis='y', style='plain', useOffset=False)
    axes[0].set_ylabel('TRA')
    fig.suptitle('Two independently trained seeds per configuration; range is not a confidence interval')
    fig.tight_layout()
    save(fig, out, 'official_TRA_two_seeds')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, seq in zip(axes, ('01', '02')):
        bottom = np.zeros(4)
        for category in WEIGHTS:
            values = []
            for name in NAMES:
                runs = [run for run in result['profiles'][name]['runs'] if run['seq'] == seq and not run['repeat']]
                values.append(np.mean([run['AOGM'][category] for run in runs]) * WEIGHTS[category])
            ax.bar(range(4), values, bottom=bottom, label=category)
            bottom += values
        ax.set_title(f'Full sequence {seq}')
        ax.set_xticks(range(4), LABELS, rotation=18, ha='right')
        ax.grid(axis='y', alpha=.15)
    axes[0].set_ylabel('Weighted edit counts (mean of two seeds)')
    axes[1].legend(ncol=3, frameon=False)
    fig.suptitle('AOGM attribution from official TRA logs; repository CTC default weights')
    fig.tight_layout()
    save(fig, out, 'official_AOGM_attribution')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, seq in zip(axes, ('01', '02')):
        original, split = sources[f'{seq}_rebuild']['frames'], sources[f'{seq}_resplit_v2']['frames']
        frames = [row['frame'] for row in original]
        ax.plot(frames, [row['gt_nodes'] for row in original], color='#323232', label='GT identities')
        ax.plot(frames, [row['pred_nodes'] for row in original], color=COLORS[0], label='Original instances')
        ax.plot(frames, [row['pred_nodes'] for row in split], color=COLORS[2], label='Re-split instances')
        ax.set_title(f'Sequence {seq} / diagnostic counts')
        ax.set_xlabel('Frame')
        ax.grid(alpha=.15)
    axes[0].set_ylabel('Instances or GT identities')
    axes[1].legend(frameon=False)
    fig.suptitle('Instance counts alone cannot distinguish over-segmentation from under-segmentation')
    fig.tight_layout()
    save(fig, out, 'frontend_instances_by_frame')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', required=True)
    parser.add_argument('--sources', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    plot(args.summary, args.sources, args.out)
