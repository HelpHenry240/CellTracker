"""补齐前端再切分 × 孤立节点处理的 2×2 对照，避免依据旧版结果预选后端。"""
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import json
import os
import sys
import time

ROOT = Path('/root/autodl-tmp/CellTracker_rebuild_20261008')
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / 'paperpipe/src'))
from papertrack.config import load_config, override, save_config
spec = importlib.util.spec_from_file_location('matrix', ROOT / 'paperpipe/scripts/run_ablation_matrix.py')
matrix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(matrix)
base = load_config(ROOT / 'experiments/P7_ideas_calibration01_v2_20261009/calibrated.yaml')
base = override(base, ['schema_version=ideas-v3', 'gnn.lambda_ot=0.001'])
output = ROOT / 'experiments/P7_ideas_v3_paper_all_k16_matrix_20261009'
config = ROOT / 'experiments/P7_ideas_v3_paper_all_k16_input_20261009.yaml'
save_config(base, config)
args = SimpleNamespace(config=str(config), out=str(output), ablations=[],
    encoder01='data/interim/encoder_01_resplit_v2.npz', encoder02='data/interim/encoder_02_resplit_v2.npz',
    h5_01='data/interim/Fluo-N3DH-CE_01_resplit_v2.h5', h5_02='data/interim/Fluo-N3DH-CE_02_resplit_v2.h5',
    gt_01='data/interim/Fluo-N3DH-CE_01.h5', gt_02='data/interim/Fluo-N3DH-CE_02.h5',
    exp_prefix='P7_ideas_v3_paper_all_k16', device='cuda', frames=None, seeds=[20261008, 20261009],
    epochs=60, official=True, dataset='Fluo-N3DH-CE', official_gt_root='/root/autodl-tmp/ctc/raw',
    official_tools='/root/EvaluationSoftware/Linux', recalibrate=True, skip_baseline=False)
plan = matrix.make_plan(args)
job = next(job for job in plan['jobs'] if job['id'].endswith('seed20261008_eval02'))
repeat = json.loads(json.dumps(job))
previous = repeat['id']
repeat['id'] = previous + '_repeat'
repeat['command'][repeat['command'].index('--exp-id') + 1] = repeat['id']
repeat['outputs'] = [path.replace(previous, repeat['id']) for path in repeat['outputs']]
plan['jobs'].append(repeat)
plan['noise_protocol'] = 'same checkpoint, configuration and inputs; full seq02 inference/export/official repeat'
(output / 'plan.json').write_text(json.dumps(plan, indent=2))
(output / 'source_release.json').write_text((ROOT / 'SOURCE_VERSION.json').read_text())
while not (ROOT / 'logs/ideas_frontend_features_20261009.complete').exists():
    os.kill(int((ROOT / 'logs/ideas_frontend_prepare_20261009.pid').read_text()), 0)
    time.sleep(10)
# 这里只做 CPU 准备；GPU 任务仍等待前三档队列完成。
matrix.execute_plan({**plan, 'jobs': plan['jobs'][:2]}, output)
report = json.loads((output / 'calibration/baseline.json').read_text())
penalty = 0.001 * report['suggestions']['theta_c']
print(json.dumps({'profile': 'paper_all_k16', 'lambda_times_true_cost_p999': penalty}), flush=True)
if penalty > 1:
    raise RuntimeError('真实边 OT 惩罚过大，需要重新标定 λ')
while not (ROOT / 'logs/ideas_final_v3_20261009.complete').exists():
    os.kill(int((ROOT / 'logs/ideas_final_v3_20261009.pid').read_text()), 0)
    time.sleep(10)
matrix.execute_plan(plan, output)
(ROOT / 'logs/ideas_final_factorial_20261009.complete').write_text('complete\n')
