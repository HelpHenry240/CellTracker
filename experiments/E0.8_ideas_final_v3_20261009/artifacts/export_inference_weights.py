"""保留原训练断点，另存便于复现实验的小型推理权重并逐张量校验。"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import torch


def digest(path, algorithm):
    return hashlib.new(algorithm, Path(path).read_bytes()).hexdigest()


def export(root, destination, repository):
    root, destination = Path(root), Path(destination)
    sys.path.insert(0, str(Path(repository) / 'paperpipe/src'))
    from papertrack.gnn.infer import load_model
    metadata = []
    for name in ('paper_all_k0', 'conservative_k0', 'paper_all_k16', 'conservative_k16'):
        source_release = json.loads((root / 'experiments' / f'P7_ideas_v3_{name}_matrix_20261009/source_release.json').read_text())
        for seed in (20261008, 20261009):
            source = root / 'experiments' / f'P7_ideas_v3_{name}_matrix_20261009/models/baseline/seed{seed}/best.pt'
            original = torch.load(source, map_location='cpu', weights_only=False)
            payload = {key: original[key] for key in ('model', 'model_config', 'train_config', 'epoch', 'input_contract')}
            payload['artifact_type'] = 'inference_only'
            payload['source_checkpoint_sha256'] = digest(source, 'sha256')
            payload['source_release'] = source_release
            output = destination / name / f'seed{seed}_inference.pt'
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.exists():
                raise FileExistsError(output)
            torch.save(payload, output)
            loaded = load_model(output, 'cpu')
            if not all(torch.equal(value, loaded.state_dict()[key]) for key, value in original['model'].items()):
                raise RuntimeError('推理权重与原训练检查点不同')
            row = {'profile': name, 'seed': seed, 'path': str(output), 'bytes': output.stat().st_size,
                'md5': digest(output, 'md5'), 'sha256': digest(output, 'sha256'),
                'source_checkpoint_sha256': payload['source_checkpoint_sha256'], 'epoch': original['epoch'],
                'input_contract_signature': original['input_contract']['signature'],
                'git_commit': source_release['git_commit'], 'tensor_identity_verified': True}
            output.with_suffix('.json').write_text(json.dumps(row, indent=2))
            metadata.append(row)
    (destination / 'manifest.json').write_text(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--repository', required=True)
    args = parser.parse_args()
    export(args.root, args.out, args.repository)
