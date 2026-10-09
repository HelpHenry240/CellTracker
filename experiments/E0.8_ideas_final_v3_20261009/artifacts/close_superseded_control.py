"""结束尚未训练的旧版对照队列；保留计划、已完成 CPU 产物和取消原因。"""
from pathlib import Path
import json
import os
import signal
import time

ROOT = Path('/root/autodl-tmp/CellTracker_rebuild_20261008')
matrix = ROOT / 'experiments/P7_ideas_matched_baseline_matrix_20261009'
pid = int((ROOT / 'logs/ideas_control_20261009.pid').read_text())
status = json.loads((matrix / 'status.json').read_text())
plan = json.loads((matrix / 'plan.json').read_text())
if any(value['status'] == 'running' for value in status.values()):
    raise RuntimeError('对照队列已经在执行任务，需先检查子进程，不能直接结束包装进程')
os.kill(pid, signal.SIGTERM)
reason = 'ideas-v2 background-id motion defect confirmed; superseded by independently retrained ideas-v3 profiles'
for job in plan['jobs']:
    if job['id'] not in status:
        status[job['id']] = {'status': 'cancelled_before_start', 'reason': reason}
pending = matrix / 'status.pending.json'
pending.write_text(json.dumps(status, indent=2))
pending.replace(matrix / 'status.json')
decision = {'pid_terminated': pid, 'reason': reason, 'completed_CPU_outputs_preserved': True,
    'official_v2_result_conclusion': 'screening only', 'replacement': 'P7_ideas_v3_*_matrix_20261009',
    'GPU_release_gate': 'wait for ideas_next_20261009.complete; marker signals queue termination, not completed experiment jobs'}
(matrix / 'cancellation.json').write_text(json.dumps(decision, indent=2))
print(json.dumps(decision), flush=True)
while not (ROOT / 'logs/ideas_next_20261009.complete').exists():
    os.kill(int((ROOT / 'logs/ideas_next_20261009.pid').read_text()), 0)
    time.sleep(10)
# 前端只用该文件判断 GPU 队列已经结束；实际实验状态明确为 cancelled_before_start。
(ROOT / 'logs/ideas_control_20261009.complete').write_text(json.dumps(decision, indent=2))
print('Superseded queue ended; GPU gate released to frozen encoder extraction.', flush=True)
