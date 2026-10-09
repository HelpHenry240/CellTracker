"""将证据归档解包到新目录，并验证每个文件的 SHA256；拒绝链接和目录穿越。"""
from pathlib import Path
import argparse
import hashlib
import json
import tarfile


def unpack(archive_path, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive_path) as archive:
        for member in archive:
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination) or not member.isfile():
                raise ValueError(f'归档含不合法条目：{member.name}')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.extractfile(member).read())
    manifest = json.loads((destination / 'evidence_manifest.json').read_text())
    for name, expected in manifest['files'].items():
        actual = hashlib.sha256((destination / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'证据散列不一致：{name}')
    print(json.dumps({'verified_files': len(manifest['files']), 'destination': str(destination)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    unpack(args.archive, args.out)
