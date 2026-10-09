"""收集实验留痕与小型 GNN 权重，排除可再生的体数据、图缓存和训练断点备份。"""
from pathlib import Path
import argparse
import hashlib
import json
import tarfile

parser=argparse.ArgumentParser()
parser.add_argument('--root',required=True)
parser.add_argument('--out',required=True)
args=parser.parse_args()
root=Path(args.root).resolve()
paths=[]
for experiment in sorted((root/'experiments').glob('*')):
    if not experiment.name.startswith(('P7_ideas','GNN_')):
        continue
    for path in sorted(experiment.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts:
            continue
        if path.suffix in {'.tif','.h5','.npz','.npy','.tgz','.gz','.pyc'}:
            continue
        if path.suffix=='.pt' and path.name!='best.pt':
            continue
        if path.stat().st_size>10_000_000:
            continue
        paths.append(path)
for path in sorted((root/'logs').glob('*')):
    if path.is_file() and ('ideas_' in path.name or 'gpu_' in path.name):
        paths.append(path)
if (root/'SOURCE_VERSION.json').exists():
    paths.append(root/'SOURCE_VERSION.json')
files={str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
manifest=Path(args.out).with_suffix('.manifest.json')
manifest.write_text(json.dumps({'files':files,'count':len(files)},indent=2))
with tarfile.open(args.out,'w:gz') as archive:
    for path in paths:
        archive.add(path,arcname=str(path.relative_to(root)),recursive=False)
    archive.add(manifest,arcname='evidence_manifest.json')
print(json.dumps({'files':len(files),'bytes':Path(args.out).stat().st_size}))
