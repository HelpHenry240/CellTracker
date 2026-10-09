"""只读收集 Oracle 的可复核证据；原始体数据、缓存与训练断点继续保留在云端。"""
from pathlib import Path
import argparse
import hashlib
import io
import json
import tarfile


def collect(root, output):
    root = Path(root).resolve()
    paths = []
    for base in [*root.glob('experiments/P8_gt_marker_encoder*'), root / 'logs']:
        candidates = [base] if base.is_file() else base.rglob('*')
        for path in candidates:
            if not path.is_file() or any(part in {'cache', 'calibration_cache', 'graphs', '__pycache__'} for part in path.parts):
                continue
            if path.suffix in {'.h5', '.npz', '.npy', '.tif', '.tiff', '.pt', '.gz'}:
                continue
            paths.append(path)
    paths.append(root / 'SOURCE_VERSION.json')
    for path in (root / 'data/interim').glob('encoder_*gt_marker*.json'):
        paths.append(path)
    files = {}
    with tarfile.open(output, 'w:gz') as archive:
        for path in sorted(set(paths)):
            data = path.read_bytes()
            name = str(path.relative_to(root))
            files[name] = hashlib.sha256(data).hexdigest()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mtime = int(path.stat().st_mtime)
            archive.addfile(info, io.BytesIO(data))
        data = json.dumps({'files': files, 'count': len(files)}, indent=2).encode()
        info = tarfile.TarInfo('evidence_manifest.json')
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    print(json.dumps({'files': len(files), 'bytes': Path(output).stat().st_size}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    collect(args.root, args.out)
