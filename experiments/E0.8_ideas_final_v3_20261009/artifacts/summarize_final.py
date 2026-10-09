"""汇总已落盘的双序列官方指标、重复运行误差和误差归属，不使用本地代理指标。"""
from pathlib import Path
import argparse
import importlib.util
import json
import statistics


def summarize(root, repository):
    root = Path(root)
    spec = importlib.util.spec_from_file_location('tra_counts', Path(repository) / 'scripts/analyze_tra_log.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = {'profiles': {}, 'definition': {
        'seeds': [20261008, 20261009], 'training': 'seq01 only',
        'repeat': 'same seed20261008 checkpoint, complete seq02 inference/export/official evaluation',
        'minimum_claim_threshold_TRA': 0.001,
        'seed_spread': 'two-seed max minus min; not a confidence interval'}}
    for name in ('paper_all_k0', 'conservative_k0', 'paper_all_k16', 'conservative_k16'):
        matrix = root / 'experiments' / f'P7_ideas_v3_{name}_matrix_20261009'
        plan = json.loads((matrix / 'plan.json').read_text())
        status = json.loads((matrix / 'status.json').read_text())
        if any(status.get(job['id'], {}).get('status') != 'complete' for job in plan['jobs']):
            raise RuntimeError(f'{name}: 任务未全部完成')
        profile = {'matrix_jobs_complete': len(plan['jobs']), 'runs': [], 'summary': {}}
        for job in plan['jobs']:
            if job['kind'] != 'evaluate':
                continue
            path = Path(job['outputs'][0])
            # 云端绝对路径转为归档中对应位置；原路径仍保留在计划证据中。
            trial = root / 'experiments' / path.parent.name
            metric = json.loads((trial / path.name).read_text())
            counts = module.parse(trial / 'official_logs/TRA_log.txt')['counts']
            profile['runs'].append({'id': job['id'], 'seq': path.stem[-2:],
                'seed': int(job['id'].split('_seed')[1][:8]), 'repeat': job['id'].endswith('_repeat'),
                **{key: metric[key] for key in ('DET', 'SEG', 'TRA')}, 'AOGM': counts})
        for seq in ('01', '02'):
            runs = [run for run in profile['runs'] if run['seq'] == seq and not run['repeat']]
            profile['summary'][seq] = {key: {'mean': statistics.mean(run[key] for run in runs),
                'min': min(run[key] for run in runs), 'max': max(run[key] for run in runs),
                'seed_spread': max(run[key] for run in runs) - min(run[key] for run in runs)}
                for key in ('DET', 'SEG', 'TRA')}
        original = next(run for run in profile['runs'] if run['seq'] == '02' and run['seed'] == 20261008 and not run['repeat'])
        repeated = next(run for run in profile['runs'] if run['repeat'])
        profile['repeat_absolute_difference'] = {key: abs(original[key] - repeated[key]) for key in ('DET', 'SEG', 'TRA')}
        profile['comparison_resolution_TRA'] = max(0.001, profile['repeat_absolute_difference']['TRA'],
            *(profile['summary'][seq]['TRA']['seed_spread'] for seq in ('01', '02')))
        result['profiles'][name] = profile
    result['comparisons'] = {}
    for before, after in (('paper_all_k0', 'conservative_k0'), ('paper_all_k16', 'conservative_k16'),
                          ('paper_all_k0', 'paper_all_k16'), ('conservative_k0', 'conservative_k16')):
        left, right = result['profiles'][before], result['profiles'][after]
        resolution = max(left['comparison_resolution_TRA'], right['comparison_resolution_TRA'])
        deltas = {}
        for seq in ('01', '02'):
            pairs = []
            for seed in (20261008, 20261009):
                a = next(run for run in left['runs'] if run['seed'] == seed and run['seq'] == seq and not run['repeat'])
                b = next(run for run in right['runs'] if run['seed'] == seed and run['seq'] == seq and not run['repeat'])
                pairs.append(b['TRA'] - a['TRA'])
            deltas[seq] = {'paired_seed_deltas_TRA': pairs, 'mean_delta_TRA': statistics.mean(pairs),
                'above_resolution_both_seeds': all(delta > resolution for delta in pairs)}
        result['comparisons'][f'{before} -> {after}'] = {'resolution_TRA': resolution, 'sequences': deltas}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--repository', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    Path(args.out).write_text(json.dumps(summarize(args.root, args.repository), ensure_ascii=False, indent=2))
