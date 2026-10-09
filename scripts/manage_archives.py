#!/usr/bin/env python3
"""压缩历史文件并校验逐文件SHA256；恢复到新目录，避免覆盖现有结果。"""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import io
import json
import shutil
import tarfile

MANIFEST = '__archive_manifest__.json'


def digest(stream):
    value = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        value.update(block)
    return value.hexdigest()


def file_digest(path):
    with Path(path).open('rb') as stream:
        return digest(stream)


def safe_name(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or not path.parts or str(path) != name:
        raise ValueError(f'不安全的归档路径：{name}')
    return path


def verify(archive):
    """解包前验证所有条目；拒绝链接、重复路径、额外条目和损坏的文件。"""
    with tarfile.open(archive, 'r:gz') as handle:
        members = handle.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)) or MANIFEST not in names:
            raise ValueError('归档清单缺失或条目重复')
        for member in members:
            safe_name(member.name)
            if not member.isfile():
                raise ValueError('归档只允许普通文件')
        manifest = json.load(handle.extractfile(MANIFEST))
        if set(names) != set(manifest['files']) | {MANIFEST}:
            raise ValueError('归档条目与清单不一致')
        for member in members:
            if member.name == MANIFEST:
                continue
            expected = manifest['files'][member.name]
            if member.size != expected['bytes'] or digest(handle.extractfile(member)) != expected['sha256']:
                raise ValueError(f'文件校验失败：{member.name}')
    return manifest


def pack(root, sources, output, source_commit, prune=False):
    """完整归档、验证成功后才允许清理源目录；文件变化时保留原件。"""
    root, output = Path(root).resolve(), Path(output).resolve()
    paths = []
    for name in sources:
        safe_name(str(name))
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root) or path.resolve() == root:
            raise ValueError('源路径必须位于项目内，且不能是链接或项目根目录')
        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_dir() and output.is_relative_to(path):
            raise ValueError('归档不能写入将被清理的源目录')
        paths.append(path)
    files = sorted({file for path in paths for file in ([path] if path.is_file() else path.rglob('*'))
                    if file.is_file() or file.is_symlink()})
    if any(file.is_symlink() for file in files):
        raise ValueError('源文件含链接，必须先核对其所有权')
    if output.exists():
        raise FileExistsError(output)
    pending = output.with_name(output.name + '.part')
    if pending.exists():
        raise FileExistsError(pending)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {'source_commit': source_commit, 'source_paths': [str(p.relative_to(root)) for p in paths],
                'files': {}}
    with tarfile.open(pending, 'w:gz', compresslevel=3) as handle:
        for file in files:
            name = str(file.relative_to(root))
            stat = file.stat()
            manifest['files'][name] = {'sha256': file_digest(file), 'bytes': stat.st_size,
                                       'mode': stat.st_mode & 0o777}
            handle.add(file, arcname=name, recursive=False)
        data = json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode()
        entry = tarfile.TarInfo(MANIFEST)
        entry.size = len(data)
        handle.addfile(entry, io.BytesIO(data))
    verify(pending)
    pending.replace(output)
    receipt = {'archive': str(output.relative_to(root)) if output.is_relative_to(root) else str(output),
               'sha256': file_digest(output),
               'bytes': output.stat().st_size, 'source_commit': source_commit,
               'source_paths': manifest['source_paths'], 'files': len(files),
               'original_bytes': sum(row['bytes'] for row in manifest['files'].values())}
    if prune:
        prune_sources(root, output)
    return receipt


def prune_sources(root, archive):
    root = Path(root).resolve()
    manifest = verify(archive)
    sources = [root / safe_name(name) for name in manifest['source_paths']]
    for source in sources:
        if source.is_symlink() or not source.resolve().is_relative_to(root) or source.resolve() == root:
            raise ValueError('归档后源路径变为链接或越界，取消清理')
    for name, row in manifest['files'].items():
        path = root / safe_name(name)
        if path.is_symlink() or not path.is_file() or file_digest(path) != row['sha256']:
            raise ValueError(f'归档后原件改变，取消清理：{name}')
    # 先核验整个文件集合，防止归档期间新产生的文件随目录一起被删除。
    current = {str(p.relative_to(root)) for source in sources
               for p in ([source] if source.is_file() else source.rglob('*'))
               if p.is_file() or p.is_symlink()}
    if current != set(manifest['files']):
        raise ValueError('归档后文件集合改变，取消清理')
    for source in sorted(sources, key=lambda p: len(p.parts), reverse=True):
        if source.is_dir():
            shutil.rmtree(source)
        elif source.exists():
            source.unlink()


def restore(archive, destination):
    manifest = verify(archive)
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive, 'r:gz') as handle:
        for member in handle:
            if member.name == MANIFEST:
                continue
            target = destination / safe_name(member.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                shutil.copyfileobj(handle.extractfile(member), stream, length=1024 * 1024)
            target.chmod(manifest['files'][member.name]['mode'])
    return {'restored_files': len(manifest['files']), 'destination': str(destination)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    create = sub.add_parser('pack')
    create.add_argument('--root', default='.')
    create.add_argument('--sources', nargs='+', required=True)
    create.add_argument('--out', required=True)
    create.add_argument('--source-commit', required=True)
    create.add_argument('--prune', action='store_true')
    check = sub.add_parser('verify')
    check.add_argument('archive')
    extract = sub.add_parser('restore')
    extract.add_argument('archive')
    extract.add_argument('--out', required=True)
    args = parser.parse_args()
    if args.action == 'pack':
        result = pack(args.root, args.sources, args.out, args.source_commit, args.prune)
    elif args.action == 'restore':
        result = restore(args.archive, args.out)
    else:
        result = verify(args.archive)
        result = {key: value for key, value in result.items() if key != 'files'} | {'verified_files': len(result['files'])}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
