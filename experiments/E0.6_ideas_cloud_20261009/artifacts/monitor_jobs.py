"""只读显示后台队列、官方结果及误差归属，不把本地指标用于定论。"""
from pathlib import Path
import argparse,json,importlib.util
p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
r=Path(a.root).resolve()
spec=importlib.util.spec_from_file_location('tra_parser',r/'scripts/analyze_tra_log.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
for matrix in sorted((r/'experiments').glob('P7_ideas_*matrix*')):
    status=matrix/'status.json'; plan=matrix/'plan.json'
    if not status.exists():continue
    statuses=json.loads(status.read_text()); jobs=json.loads(plan.read_text())['jobs']
    print(json.dumps({'matrix':matrix.name,'completed':sum(v['status']=='complete' for v in statuses.values()),
                     'total':len(jobs),'active':[{k:v['status']} for k,v in statuses.items() if v['status']!='complete']}))
for trial in sorted((r/'experiments').glob('P7_ideas_*')):
    for path in trial.glob('metrics_official_*.json'):
        data=json.loads(path.read_text()); log=trial/'official_logs/TRA_log.txt'
        counts=module.parse(log)['counts'] if log.exists() else None
        print(json.dumps({'trial':trial.name,**{k:data[k] for k in ['DET','SEG','TRA']},'AOGM':counts}))
