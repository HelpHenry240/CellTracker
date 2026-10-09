"""源码归档可独立运行；文档入口和正式证据不依赖已清理目录。"""
from pathlib import Path
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
from urllib.parse import unquote

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_current_document_links_resolve():
    paths = [ROOT / 'README.md', ROOT / 'scripts/README.md', ROOT / 'experiments/INDEX.md',
             ROOT / 'experiments/RECORDS.md', ROOT / 'experiments/archives/README.md',
             *(ROOT / 'docs').rglob('*.md')]
    missing = []
    for file in paths:
        for target in re.findall(r'\]\(([^\s)]+)', file.read_text()):
            target = unquote(target).strip('<>').split('#')[0]
            if not target or '://' in target or target.startswith(('mailto:', 'data:', 'app:')):
                continue
            if not (file.parent / target).exists():
                missing.append((str(file.relative_to(ROOT)), target))
    assert not missing, f'文档链接失效：{missing}'


def test_retained_formal_evidence_is_byte_identical():
    manifest_path = ROOT / 'experiments/retention_manifest.json'
    if not manifest_path.exists():
        pytest.skip('轻量源码包不携带实验记录')
    manifest = json.loads(manifest_path.read_text())
    missing, changed = [], []
    for name, expected in manifest['files'].items():
        file = ROOT / name
        if not file.is_file():
            missing.append(name)
            continue
        # 证据为小文件和文本，分块读取以便日后增加更大的日志。
        value = hashlib.sha256()
        with file.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                value.update(block)
        if value.hexdigest() != expected:
            changed.append(name)
    assert not missing and not changed, f'原始证据缺失={missing}，字节改变={changed}'


def test_retained_evidence_is_in_git_index():
    probe = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', '--show-toplevel'], capture_output=True, text=True)
    if probe.returncode or Path(probe.stdout.strip()).resolve() != ROOT:
        pytest.skip('源码包无Git索引')
    manifest = json.loads((ROOT / 'experiments/retention_manifest.json').read_text())
    indexed = set(subprocess.check_output(['git', '-C', str(ROOT), 'ls-files'], text=True).splitlines())
    assert set(manifest['files']) <= indexed, '保留证据仍被忽略，克隆后将缺失'


def test_exported_cli_and_matrix_use_unified_layout(tmp_path):
    probe = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', '--show-toplevel'], capture_output=True, text=True)
    if probe.returncode or Path(probe.stdout.strip()).resolve() != ROOT:
        pytest.skip('源码包无Git索引')
    tree = subprocess.check_output(['git', '-C', str(ROOT), 'write-tree'], text=True).strip()
    payload = subprocess.check_output(['git', '-C', str(ROOT), 'archive', tree, 'src', 'scripts', 'configs'])
    package = tmp_path / 'package'
    package.mkdir()
    with tarfile.open(fileobj=io.BytesIO(payload)) as handle:
        handle.extractall(package, filter='data')
    env = dict(os.environ, PYTHONPATH=str(package / 'src'))
    result = subprocess.run([sys.executable, str(package / 'scripts/run_pipeline.py'), '--list-modules'],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert len(json.loads(result.stdout)) == 25
    plan_dir = tmp_path / 'plan'
    command = [sys.executable, str(package / 'scripts/run_ablation_matrix.py'),
               '--config', str(package / 'configs/ce_calibrated_20261009.yaml'),
               '--h5-01', 'prediction01.h5', '--h5-02', 'prediction02.h5',
               '--gt-01', 'truth01.h5', '--gt-02', 'truth02.h5',
               '--encoder01', 'encoder01.npz', '--encoder02', 'encoder02.npz',
               '--out', str(plan_dir), '--ablations', 'fgw', '--recalibrate']
    planned = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60)
    assert planned.returncode == 0, planned.stderr
    plan = json.loads((plan_dir / 'plan.json').read_text())
    kinds = set()
    for job in plan['jobs']:
        entry = Path(job['command'][1])
        assert entry.parent == package / 'scripts' and entry.is_file()
        assert 'paperpipe' not in str(entry)
        kinds.add(job['kind'])
        if job['kind'] == 'evaluate':
            assert all(Path(p).is_relative_to(package / 'experiments') for p in job['outputs'])
    assert kinds == {'calibrate', 'build', 'train', 'evaluate'}
    assert not (package / 'experiments').exists(), '生成计划不得运行训练或评测'
