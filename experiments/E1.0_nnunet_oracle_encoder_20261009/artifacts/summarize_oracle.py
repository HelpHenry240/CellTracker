"""汇总 Oracle 实验的官方结果与重复误差；不把历史不同实现当作特征消融。"""
from pathlib import Path
import argparse
import importlib.util
import json
import statistics


def summarize(root, repository):
    root = Path(root)
    matrix = root / 'experiments/P9_nnunet_oracle_encoder_matrix_20261009'
    plan = json.loads((matrix / 'plan.json').read_text())
    status = json.loads((matrix / 'status.json').read_text())
    if any(status.get(job['id'], {}).get('status') != 'complete' for job in plan['jobs']):
        raise RuntimeError('正式任务尚未全部完成')
    spec = importlib.util.spec_from_file_location('tra_counts', Path(repository) / 'scripts/analyze_tra_log.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = []
    for job in plan['jobs']:
        if job['kind'] != 'evaluate':
            continue
        path = Path(job['outputs'][0])
        trial = root / 'experiments' / path.parent.name
        metric = json.loads((trial / path.name).read_text())
        seq = metric['seq']
        check = metric['format_validation']
        if not check['ok'] or check['n_ghost_tracks'] or check['n_tracks_with_holes']:
            raise RuntimeError('正式提交格式未通过')
        rows.append({'id': job['id'], 'seq': seq,
            'seed': int(job['id'].split('_seed')[1][:8]), 'repeat': job['id'].endswith('_repeat'),
            **{name: metric[name] for name in ('DET', 'SEG', 'TRA')},
            'AOGM': module.parse(trial / 'official_logs/TRA_log.txt')['counts'],
            'format_validation': check, 'binary_sha256': metric['binary_sha256']})
    summary = {}
    for seq in ('01', '02'):
        values = [row for row in rows if row['seq'] == seq and not row['repeat']]
        summary[seq] = {name: {'mean': statistics.mean(row[name] for row in values),
            'min': min(row[name] for row in values), 'max': max(row[name] for row in values),
            'seed_spread': max(row[name] for row in values) - min(row[name] for row in values)}
            for name in ('DET', 'SEG', 'TRA')}
    original = next(row for row in rows if row['seq'] == '02' and row['seed'] == 20261008 and not row['repeat'])
    repeat = next(row for row in rows if row['repeat'])
    noise = {name: abs(original[name] - repeat[name]) for name in ('DET', 'SEG', 'TRA')}
    return {'protocol': {'source': 'nnU-Net semantic predictions with GT-label seed Oracle instances', 'features': 'frozen nnU-Net stage2 encoder, 128 dimensions',
        'training': 'seq01 only', 'seeds': [20261008, 20261009], 'epochs': 60,
        'repeat': 'same seed20261008 checkpoint, complete seq02 inference/export/official evaluation',
        'instance_mass': 'volume', 'volume_conservation': True,
        'foreground_preserved': True, 'unseeded_policy': 'keep_component', 'min_volume': 1, 'hole_policy': 'split',
        'SEG_interpretation': 'valid segmentation metric; instance ownership is GT-seed-assisted', 'unet_training': False,
        'handcrafted_feature_control': False}, 'jobs_complete': len(plan['jobs']),
        'source_release': json.loads((matrix / 'source_release.json').read_text()),
        'runs': rows, 'summary': summary, 'repeat_absolute_difference': noise,
        'comparison_resolution_TRA': max(.001, noise['TRA'],
            *(summary[seq]['TRA']['seed_spread'] for seq in ('01', '02'))),
        'limitations': ['No matched handcrafted-feature control: cannot claim encoder improvement',
            'Prior nnU-Net training included some seq02 frames: not end-to-end blind generalization',
            'Historical GT marker and drop/min-volume300 Oracle protocols differ from this strict-foreground Oracle']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--repository', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    Path(args.out).write_text(json.dumps(summarize(args.root, args.repository), ensure_ascii=False, indent=2))
