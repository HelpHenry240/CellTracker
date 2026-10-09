"""收集可复核的云端证据；每个文件只读一次，避免活动日志的散列与归档不一致。"""
from pathlib import Path
import argparse
import hashlib
import io
import json
import tarfile


def collect(root, output):
    root = Path(root).resolve()
    paths = set()
    for experiment in (root / 'experiments').glob('P7_ideas*'):
        candidates = [experiment] if experiment.is_file() else experiment.rglob('*')
        for path in candidates:
            if not path.is_file() or '__pycache__' in path.parts:
                continue
            if path.suffix in {'.tif', '.h5', '.npz', '.npy', '.tgz', '.gz', '.pyc'}:
                continue
            if path.suffix == '.pt' and path.name != 'best.pt':
                continue
            if path.stat().st_size <= 10_000_000:
                paths.add(path)
    for path in (root / 'logs').glob('*'):
        if path.is_file() and path.name.startswith(('ideas_', 'frontend_', 'SOURCE_VERSION_')):
            paths.add(path)
    for path in (root / 'data/interim').glob('*.json'):
        if 'rebuild' in path.name or 'resplit_v2' in path.name:
            paths.add(path)
    paths.add(root / 'SOURCE_VERSION.json')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    files = {}
    with tarfile.open(output, 'w:gz') as archive:
        for path in sorted(paths):
            data = path.read_bytes()
            name = str(path.relative_to(root))
            files[name] = hashlib.sha256(data).hexdigest()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mtime = int(path.stat().st_mtime)
            archive.addfile(info, io.BytesIO(data))
        manifest = json.dumps({'files': files, 'count': len(files)}, indent=2).encode()
        info = tarfile.TarInfo('evidence_manifest.json')
        info.size = len(manifest)
        archive.addfile(info, io.BytesIO(manifest))
    output.with_suffix('.manifest.json').write_bytes(manifest)
    print(json.dumps({'files': len(files), 'bytes': output.stat().st_size}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    collect(args.root, args.out)
