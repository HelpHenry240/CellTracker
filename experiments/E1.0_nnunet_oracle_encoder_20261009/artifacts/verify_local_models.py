"""核验回传的小权重指纹，并在本地CPU加载，检查全部参数为有限值。"""
from pathlib import Path
import hashlib
import json
import sys
import torch

root = Path(__file__).resolve().parents[3]
base = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'paperpipe/src'))
from papertrack.gnn.infer import load_model

manifest = json.loads((base / 'artifacts/models/manifest.json').read_text())
rows = []
for item in manifest:
    path = base / 'artifacts/models' / item['profile'] / f"seed{item['seed']}_inference.pt"
    data = path.read_bytes()
    assert len(data) == item['bytes']
    assert hashlib.md5(data).hexdigest() == item['md5']
    assert hashlib.sha256(data).hexdigest() == item['sha256']
    model = load_model(path, 'cpu')
    assert all(torch.isfinite(value).all() for value in model.state_dict().values())
    payload = torch.load(path, map_location='cpu', weights_only=False)
    assert payload['input_contract']['signature'] == item['input_contract_signature']
    assert payload['source_checkpoint_sha256'] == item['source_checkpoint_sha256']
    assert payload['source_release']['git_commit'] == item['git_commit']
    rows.append({'seed': item['seed'], 'bytes': len(data), 'cpu_load': True,
                 'hashes_match_manifest': True, 'all_parameters_finite': True,
                 'parameters': sum(value.numel() for value in model.parameters()),
                 'input_contract_signature': item['input_contract_signature']})
assert len(rows) == 2
out = base / 'logs/model_portability.json'
if out.exists():
    raise FileExistsError(out)
out.write_text(json.dumps(rows, indent=2))
print(json.dumps(rows))
