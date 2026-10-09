"""从已校验的原始归档选择小型复核材料，保留大数组原件的路径与散列。"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil


def compact(value):
    if isinstance(value, dict):
        return {key: compact(item) for key, item in value.items()}
    if isinstance(value, list):
        if len(value) > 64:
            return {'items_in_original': len(value), 'storage': 'original_verified_archive'}
        return [compact(item) for item in value]
    return value


def select(source, destination, archive):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    selected = {}
    for path in sorted(source.rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        is_trial = relative.parts[0] == 'experiments' and relative.parts[1].startswith('P7_ideas_v3_')
        is_log = relative.parts[0] == 'logs' and path.name.startswith((
            'ideas_final_', 'ideas_frontend_encoder_resume_', 'ideas_frozen_',
            'ideas_detection_', 'ideas_graph_', 'frontend_resume_', 'ideas_cloud_test'))
        is_version = relative.as_posix() == 'SOURCE_VERSION.json'
        if not (is_trial or is_log or is_version):
            continue
        if any(item in {'cache', 'calibration_cache', 'graphs', '__pycache__'} for item in relative.parts):
            continue
        if path.suffix in {'.pt', '.h5', '.npz', '.npy', '.tif', '.tiff', '.tgz'}:
            continue
        output = destination / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        data = path.read_bytes()
        original_hash = hashlib.sha256(data).hexdigest()
        if path.name == 'run_info.json' or (path.name == 'metrics.json' and '_build' in str(relative)):
            output = output.with_name(f'{output.stem}_summary.json')
            payload = {'original_path': str(relative), 'original_sha256': original_hash,
                       'original_archive': archive, 'summary': compact(json.loads(data))}
            output.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            shutil.copyfile(path, output)
        selected[str(output.relative_to(destination))] = {
            'sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
            'original_path': str(relative), 'original_sha256': original_hash,
            'bytes': output.stat().st_size}
    manifest = {'archive': archive, 'files': selected, 'count': len(selected),
                'excluded': 'reproducible caches, graphs, volumes and full training checkpoints; originals retained'}
    (destination / 'repository_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'files': len(selected), 'bytes': sum(row['bytes'] for row in selected.values())}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--archive', required=True)
    args = parser.parse_args()
    select(args.source, args.out, args.archive)
