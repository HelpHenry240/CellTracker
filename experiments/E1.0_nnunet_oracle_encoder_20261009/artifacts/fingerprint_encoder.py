"""记录本轮实际encoder文件的身份与内容散列，不修改特征或原始检测。"""
from pathlib import Path
import argparse,hashlib,json

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);a=p.parse_args()
root=Path(a.root);records={}
for seq in ('01','02'):
    feature=root/f'data/interim/encoder_{seq}_nnunet_oracle_20261009.npz'
    records[seq]={'file':str(feature),'resolved_file':str(feature.resolve()),
                  'bytes':feature.stat().st_size,'sha256':hashlib.sha256(feature.read_bytes()).hexdigest(),
                  'metadata':json.loads(feature.with_suffix('.json').read_text())}
output=root/'logs/oracle_feature_fingerprints.json'
if output.exists():raise FileExistsError(output)
output.write_text(json.dumps(records,indent=2));print(json.dumps(records))
