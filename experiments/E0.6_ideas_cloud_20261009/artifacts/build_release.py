from pathlib import Path
import hashlib,json,subprocess,tarfile
root=Path.cwd(); commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
paths=[]
for directory in ['paperpipe/src','paperpipe/scripts','paperpipe/configs','paperpipe/tests','src','tests','scripts']:
 for p in sorted((root/directory).rglob('*')):
  if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc': paths.append(p)
for relative in ['pyproject.toml','scripts/relabel_detections.py','scripts/resplit_detections.py','scripts/nnunet/export_encoder_features.py','scripts/run_ideas_pipeline.py','scripts/analyze_tra_log.py']:
 paths.append(root/relative)
files={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))}
identity=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
manifest={'git_commit':commit,'source_sha256':identity,'definition':'SHA256 of sorted JSON path-to-SHA256 mapping','files':files}
m=Path('/tmp/SOURCE_VERSION.json');m.write_text(json.dumps(manifest,indent=2))
a=Path(f'/tmp/celltracker_ideas_source_{commit[:7]}.tgz')
with tarfile.open(a,'w:gz') as t:
 for p in sorted(set(paths)):t.add(p,arcname=str(p.relative_to(root)))
 t.add(m,arcname='SOURCE_VERSION.json')
Path(f'experiments/E0.6_ideas_cloud_20261009/artifacts/SOURCE_VERSION_{commit[:7]}.json').write_text(m.read_text())
print(json.dumps({'commit':commit,'source_sha256':identity,'archive_bytes':a.stat().st_size,'file_count':len(files)}))
