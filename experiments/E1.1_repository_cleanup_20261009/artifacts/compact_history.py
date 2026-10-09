#!/usr/bin/env python3
"""一次性整理记录：先完整归档并校验，再恢复供日常阅读的正式证据。"""
from pathlib import Path
import importlib.util
import json
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[3]
STAGE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('archives', ROOT / 'scripts/manage_archives.py')
archive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archive)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def inventory(path):
    files = [p for p in path.rglob('*') if p.is_file()]
    return {'files': len(files), 'bytes': sum(p.stat().st_size for p in files)}


def keep_formal(file, folder):
    rel = file.relative_to(folder)
    # 原始掩码、重复执行脚本、初次中断证据均可从完整归档恢复。
    if 'evidence_initial' in rel.parts or '__pycache__' in rel.parts:
        return False
    if len(rel.parts) == 1:
        if file.name == 'metrics.json':
            return not ((folder / 'metrics_final.json').exists() or (folder / 'metrics_preparation_finished.json').exists())
        return file.suffix == '.yaml' or file.name in {'metrics_preparation_finished.json', 'config.yaml', 'command.sh', 'env.txt', 'git_commit.txt',
                             'notes.md', 'metrics_final.json'}
    if rel.parts[0] == 'figures':
        return True
    if 'models' in rel.parts:
        return True
    if 'official_logs' in rel.parts or file.name.startswith('metrics_official_'):
        return True
    if 'figures' in rel.parts or file.suffix in {'.py', '.sh', '.pyc', '.pid', '.npy', '.npz', '.tif', '.h5'}:
        return False
    if file.name in {'env.txt', 'git_commit.txt', 'notes.md', 'run_info.json', 'metrics.json',
                     'repository_manifest.json', 'archive_manifest.json', 'source_evidence_manifest.json'}:
        return False
    return file.suffix in {'.json', '.yaml', '.txt', '.log', '.md', '.complete'}


def main():
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    original = sorted(p for p in (ROOT / 'experiments').iterdir() if p.is_dir() and p != STAGE)
    formal_names = {'E0.7_ideas_frontend_20261009', 'E0.8_ideas_final_v3_20261009',
                    'E0.9_gt_oracle_encoder_20261009', 'E1.0_nnunet_oracle_encoder_20261009'}
    foundations = {p.name for p in original if p.name not in formal_names
                   and p.name.startswith(('E0.1_', 'E0.2_', 'E0.3_', 'E1.', 'E5.'))}
    groups = [
        ('01_foundations', [p for p in original if p.name in foundations]),
        ('02_historical_tracking', [p for p in original if p.name.startswith(('A', 'B', 'E2.', 'E3.', 'E4.', 'P'))]),
        ('03_historical_frontend', [p for p in original if p.name.startswith('C')]),
        ('04_rebuild_audit_and_failed_runs', [p for p in original if p.name.startswith(('E0.4_', 'E0.5_', 'E0.6_'))]),
        ('05_formal_evidence_complete', [p for p in original if p.name in formal_names]),
    ]
    assigned = [p for _, paths in groups for p in paths]
    assert len(assigned) == len(set(assigned)) and set(assigned) == set(original), '每个历史目录只能归档一次'
    before = {name: inventory(ROOT / name) for name in ['src', 'paperpipe', 'scripts', 'docs', 'experiments']}
    source_hashes = {str(p.relative_to(ROOT)): archive.file_digest(p)
                     for base in ['src', 'paperpipe/src/papertrack']
                     for p in (ROOT / base).rglob('*.py')}
    write(STAGE / 'artifacts/before.json', {'inventory': before, 'source_hashes': source_hashes,
                                          'experiment_directories': len(original)})
    # 正式指标在归档前登记来源，重复拷贝仍记录为证据副本而非独立实验。
    ledger = []
    for group, folders in groups:
        for folder in folders:
            for file in sorted(folder.rglob('*.json')):
                if not ('official' in file.name or file.name == 'metrics.json'):
                    continue
                try:
                    value = json.loads(file.read_text())
                except (ValueError, UnicodeError):
                    continue
                if isinstance(value, dict) and all(isinstance(value.get(k), (int, float)) for k in ['DET', 'SEG', 'TRA']):
                    ledger.append({'original_path': str(file.relative_to(ROOT)), 'source_commit': commit,
                                   'sha256': archive.file_digest(file), 'archive': f'{group}.tar.gz',
                                   'category': ('superseded_candidate_diagnostic' if 'evidence_initial' in file.parts else
                                                'superseded_rebuild_exploration' if group.startswith('04_') else
                                                'current' if folder.name in formal_names else 'historical'),
                                   'metrics': {k: value[k] for k in ['DET', 'SEG', 'TRA']}})
    write(STAGE / 'artifacts/ledger_original.json', ledger)
    staging = Path(tempfile.mkdtemp(prefix='celltracker_cleanup_retained_'))
    keepers = {}
    for folder in original:
        for file in folder.rglob('*'):
            if not file.is_file():
                continue
            selected = (keep_formal(file, folder) if folder.name in formal_names else folder.name in foundations
                        and ((file.name in {'config.yaml', 'command.sh', 'env.txt', 'git_commit.txt', 'metrics.json', 'notes.md'}
                              and file.parent == folder) or file.relative_to(folder).parts[0] in {'figures', 'logs'}))
            if not selected:
                continue
            name = str(file.relative_to(ROOT))
            keepers[name] = archive.file_digest(file)
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, target)
    write(STAGE / 'artifacts/retained_before.json', keepers)
    write(STAGE / 'artifacts/recovery_state.json', {'staging': str(staging), 'source_commit': commit,
                                                  'instruction': '若中断，只恢复校验通过的保留文件，不盲目重跑整理。'})
    receipts = []
    target_dir = ROOT / 'experiments/archives'
    snapshot = ['src', 'paperpipe', 'scripts', 'configs', 'docs', 'tests', 'pyproject.toml',
                'README.md', '.gitignore', 'experiments/INDEX.md']
    receipts.append(archive.pack(ROOT, snapshot, target_dir / '00_source_and_documents.tar.gz', commit))
    write(target_dir / 'catalog.json', receipts)
    for group, folders in groups:
        receipt = archive.pack(ROOT, [str(p.relative_to(ROOT)) for p in folders],
                               target_dir / f'{group}.tar.gz', commit, prune=True)
        receipts.append(receipt)
        write(target_dir / 'catalog.json', receipts)
        print(f'{group}: {receipt["files"]} files verified, {receipt["bytes"]} archive bytes', flush=True)
    for name, expected in keepers.items():
        source = staging / name
        assert archive.file_digest(source) == expected
        target = ROOT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        assert archive.file_digest(target) == expected
    write(ROOT / 'experiments/retention_manifest.json',
          {'source_commit': commit, 'complete_archives': 'archives/catalog.json', 'files': keepers})
    write(ROOT / 'experiments/ledger.json', {'source_commit': commit,
          'note': '历史数值保留原始口径；同路径副本与重复评测不得当作独立种子。当前结论以 metrics_final.json 为准。',
          'official_records': ledger})
    write(STAGE / 'artifacts/archive_summary.json', {'archives': receipts, 'retained_files': len(keepers),
                                                  'official_records': len(ledger)})
    shutil.rmtree(staging)
    print(f'Completed: {len(receipts)} verified archives, {len(keepers)} retained files', flush=True)


if __name__ == '__main__':
    main()
