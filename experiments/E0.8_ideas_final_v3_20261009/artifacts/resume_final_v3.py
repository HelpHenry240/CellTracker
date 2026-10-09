"""读取已存在的三档计划和状态继续执行，不覆盖标定结果、不重建已完成的图。"""
from pathlib import Path
import importlib.util
import json
import os
import time

ROOT = Path('/root/autodl-tmp/CellTracker_rebuild_20261008')
os.chdir(ROOT)
spec = importlib.util.spec_from_file_location('matrix', ROOT / 'paperpipe/scripts/run_ablation_matrix.py')
matrix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(matrix)
while not (ROOT / 'logs/ideas_frontend_features_20261009.complete').exists():
    os.kill(int((ROOT / 'logs/ideas_frontend_prepare_20261009.pid').read_text()), 0)
    time.sleep(10)
for name in ('paper_all_k0', 'conservative_k0', 'conservative_k16'):
    output = ROOT / f'experiments/P7_ideas_v3_{name}_matrix_20261009'
    plan = json.loads((output / 'plan.json').read_text())
    matrix.execute_plan({**plan, 'jobs': plan['jobs'][:2]}, output)
    report = json.loads((output / 'calibration/baseline.json').read_text())
    penalty = 0.001 * report['suggestions']['theta_c']
    print(json.dumps({'profile': name, 'lambda_times_true_cost_p999': penalty}), flush=True)
    if penalty > 1:
        raise RuntimeError('真实边 OT 惩罚过大，需要重新标定 λ，再启动训练')
    matrix.execute_plan(plan, output)
(ROOT / 'logs/ideas_final_v3_20261009.complete').write_text('complete\n')
