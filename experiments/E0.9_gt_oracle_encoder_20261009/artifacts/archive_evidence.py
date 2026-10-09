"""保留官方原始证据，并压缩大型运行诊断；原文件散列与云端完整归档均可追溯。"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil


def compact(value):
    if isinstance(value,dict):
        return {key:compact(item) for key,item in value.items()}
    if isinstance(value,list):
        if len(value)>64:
            return {'diagnostic_list_omitted':True,'items':len(value),
                    'full_record':'immutable cloud evidence archive; see archive_manifest.json'}
        return [compact(item) for item in value]
    return value


def archive(source,destination,compressed,remote):
    source,destination=Path(source),Path(destination)
    destination.mkdir(parents=True,exist_ok=False)
    expected=json.loads((source/'evidence_manifest.json').read_text())['files']
    items={}
    for name,sha in expected.items():
        path=source/name; data=path.read_bytes()
        if hashlib.sha256(data).hexdigest()!=sha:
            raise ValueError(f'原证据散列不符：{name}')
        target=destination/name; target.parent.mkdir(parents=True,exist_ok=True)
        reduced=path.suffix=='.json' and len(data)>1_000_000
        target.write_text(json.dumps(compact(json.loads(data)),ensure_ascii=False,indent=2)) if reduced else target.write_bytes(data)
        items[name]={'source_sha256':sha,'source_bytes':len(data),'compacted_diagnostics':reduced,
                    'repository_sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
    (destination/'archive_manifest.json').write_text(json.dumps({'files':items,'count':len(items),
        'full_cloud_archive':remote,'archive_sha256':hashlib.sha256(Path(compressed).read_bytes()).hexdigest(),
        'rule':'only diagnostic lists >64 in JSON files >1MB summarized; official results/logs unchanged'},indent=2))
    shutil.copy2(source/'evidence_manifest.json',destination/'source_evidence_manifest.json')
    print(json.dumps({'files':len(items),'destination':str(destination)}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--archive',required=True);parser.add_argument('--remote',required=True)
    args=parser.parse_args();archive(args.source,args.out,args.archive,args.remote)
