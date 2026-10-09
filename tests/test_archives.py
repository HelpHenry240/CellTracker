"""历史记录可恢复；校验失败或原件改变时禁止清理。"""
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import pytest

spec = importlib.util.spec_from_file_location('manage_archives', Path(__file__).resolve().parents[1] / 'scripts/manage_archives.py')
archive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archive)


def test_archive_roundtrip_preserves_bytes_and_executable_mode(tmp_path):
    source = tmp_path / 'old'
    source.mkdir()
    data = source / 'metrics.json'
    data.write_bytes(b'{"TRA":0.984784}\n')
    command = source / 'command.sh'
    command.write_bytes(b'#!/bin/bash\nexit 0\n')
    command.chmod(0o755)
    target = tmp_path / 'archives/old.tar.gz'
    receipt = archive.pack(tmp_path, ['old'], target, 'previous-commit', prune=True)
    assert receipt['files'] == 2 and not source.exists()
    archive.restore(target, tmp_path / 'restored')
    assert (tmp_path / 'restored/old/metrics.json').read_bytes() == b'{"TRA":0.984784}\n'
    assert (tmp_path / 'restored/old/command.sh').stat().st_mode & 0o777 == 0o755


def test_changed_source_cancels_pruning(tmp_path):
    source = tmp_path / 'old'
    source.mkdir()
    data = source / 'result.txt'
    data.write_text('original')
    target = tmp_path / 'archive.tar.gz'
    archive.pack(tmp_path, ['old'], target, 'previous-commit')
    data.write_text('updated')
    with pytest.raises(ValueError, match='原件改变'):
        archive.prune_sources(tmp_path, target)
    assert data.read_text() == 'updated'


def test_added_source_file_cancels_pruning(tmp_path):
    source = tmp_path / 'old'
    source.mkdir()
    (source / 'old.txt').write_text('original')
    target = tmp_path / 'archive.tar.gz'
    archive.pack(tmp_path, ['old'], target, 'previous-commit')
    (source / 'new.txt').write_text('new record')
    with pytest.raises(ValueError, match='文件集合改变'):
        archive.prune_sources(tmp_path, target)
    assert (source / 'new.txt').is_file()


def test_changed_source_parent_cannot_prune_external_file(tmp_path):
    root = tmp_path / 'project'
    source = root / 'old'
    source.mkdir(parents=True)
    (source / 'result.txt').write_text('original')
    target = root / 'archives/old.tar.gz'
    archive.pack(root, ['old/result.txt'], target, 'previous-commit')
    external = tmp_path / 'outside'
    source.rename(external)
    source.symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match='越界'):
        archive.prune_sources(root, target)
    assert (external / 'result.txt').read_text() == 'original'


def test_restore_rejects_traversal_before_creating_destination(tmp_path):
    target = tmp_path / 'unsafe.tar.gz'
    with tarfile.open(target, 'w:gz') as handle:
        for name, data in [('__archive_manifest__.json', b'{"files":{}}'), ('../outside', b'invalid')]:
            member = tarfile.TarInfo(name)
            member.size = len(data)
            handle.addfile(member, io.BytesIO(data))
    with pytest.raises(ValueError, match='不安全'):
        archive.restore(target, tmp_path / 'restored')
    assert not (tmp_path / 'restored').exists()


@pytest.mark.parametrize('name', ['a//b', 'a/./b', './a', 'a/', '.', '/a', '../a'])
def test_archive_paths_cannot_alias_or_escape(name):
    with pytest.raises(ValueError, match='不安全'):
        archive.safe_name(name)
