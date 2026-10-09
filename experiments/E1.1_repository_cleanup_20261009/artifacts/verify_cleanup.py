#!/usr/bin/env python3
"""检查归档、算法源文件与现有权重；不训练模型、不运行云任务。"""
from pathlib import Path
import importlib.util
import json
import subprocess
import sys
import tempfile
import shutil

ROOT = Path(__file__).resolve().parents[3]
STAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
spec = importlib.util.spec_from_file_location('archives', ROOT / 'scripts/manage_archives.py')
archive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archive)


def main():
    catalog = json.loads((ROOT / 'experiments/archives/catalog.json').read_text())
    checked = []
    for row in catalog:
        path = ROOT / row['archive']
        assert archive.file_digest(path) == row['sha256'], path
        content = archive.verify(path)
        assert len(content['files']) == row['files']
        checked.append({'archive': row['archive'], 'verified_files': row['files'], 'sha256': row['sha256']})
        print(f'Archive verified: {path.name}', flush=True)
    before = json.loads((STAGE / 'artifacts/before.json').read_text())
    changed = []
    unchanged = []
    expected = {'src/celltracker/gnn/__init__.py', 'src/celltracker/pipeline/__init__.py',
                'paperpipe/src/papertrack/__init__.py', 'paperpipe/src/papertrack/_paths.py',
                'paperpipe/src/papertrack/gnn/train.py',
                'paperpipe/src/papertrack/representation/__init__.py',
                'paperpipe/src/papertrack/longrange/__init__.py'}
    for original, digest in before['source_hashes'].items():
        current = ROOT / original.replace('paperpipe/src/', 'src/')
        if current.is_file() and archive.file_digest(current) == digest:
            unchanged.append(str(current.relative_to(ROOT)))
        else:
            assert original in expected, f'发现非预期算法改动：{original}'
            changed.append(original)
    # 从原源码归档实际恢复到新目录，证明恢复工具可用且每项字节不变。
    base = Path(tempfile.mkdtemp(prefix='celltracker_archive_restore_'))
    restored = base / 'original'
    archive.restore(ROOT / catalog[0]['archive'], restored)
    for original, digest in before['source_hashes'].items():
        assert archive.file_digest(restored / original) == digest
    shutil.rmtree(base)
    from papertrack.gnn.infer import load_model
    weights = []
    for folder in ['E0.8_ideas_final_v3_20261009', 'E0.9_gt_oracle_encoder_20261009',
                   'E1.0_nnunet_oracle_encoder_20261009']:
        directory = ROOT / 'experiments' / folder / 'artifacts/models'
        manifest = json.loads((directory / 'manifest.json').read_text())
        for row in manifest:
            matches = list(directory.rglob(Path(row['path']).name))
            # E0.8同一种子有四个profile，先按profile识别文件夹。
            path = next(p for p in matches if p.parent.name == row['profile'])
            assert archive.file_digest(path) == row['sha256']
            model = load_model(path, 'cpu')
            assert model.input_contract and next(model.parameters()).device.type == 'cpu'
            weights.append({'path': str(path.relative_to(ROOT)), 'sha256': row['sha256'],
                            'cpu_load': 'passed', 'parameters': sum(p.numel() for p in model.parameters())})
            del model
    assert len(weights) == 12
    for file in (ROOT / 'scripts').rglob('*.sh'):
        subprocess.run(['bash', '-n', str(file)], check=True)
    for name in ['run_pipeline', 'train_gnn', 'calibrate_params', 'run_ablation_matrix']:
        subprocess.run([sys.executable, str(ROOT / f'scripts/{name}.py'), '--help'],
                       stdout=subprocess.DEVNULL, check=True)
    report = {'archives': checked, 'source_restore': 'passed', 'algorithm_files_byte_identical': unchanged,
              'layout_only_source_edits': changed, 'current_weights': weights,
              'shell_syntax': 'passed', 'canonical_cli_help': 'passed', 'unet_training': False,
              'new_official_evaluations': 0}
    (STAGE / 'artifacts/verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(f'Passed: {len(checked)} archives, {len(unchanged)} unchanged algorithm files, {len(weights)} weights', flush=True)


if __name__ == '__main__':
    main()
