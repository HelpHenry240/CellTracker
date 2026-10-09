"""最终三档流水线：先准备 CPU 图，再按 GPU 队列顺序训练、评测与重复运行。"""
from pathlib import Path
from types import SimpleNamespace
import importlib.util,json,subprocess,sys,time
ROOT=Path('/root/autodl-tmp/CellTracker_rebuild_20261008')
sys.path.insert(0,str(ROOT/'paperpipe/src'))
from papertrack.config import load_config,override,save_config
spec=importlib.util.spec_from_file_location('matrix',ROOT/'paperpipe/scripts/run_ablation_matrix.py')
matrix=importlib.util.module_from_spec(spec);spec.loader.exec_module(matrix)
base=load_config(ROOT/'experiments/P7_ideas_calibration01_v2_20261009/calibrated.yaml')
base=override(base,['schema_version=ideas-v3','gnn.lambda_ot=0.001'])
plans=[]
for name,resplit,keep_isolated in [('paper_all_k0',False,False),('conservative_k0',False,True),('conservative_k16',True,True)]:
    output=ROOT/'experiments'/f'P7_ideas_v3_{name}_matrix_20261009'
    configuration=override(base,['reconstruct.filter_isolated=false'] if keep_isolated else [])
    config_path=ROOT/'experiments'/f'P7_ideas_v3_{name}_input_20261009.yaml'
    save_config(configuration,config_path)
    suffix='resplit_v2' if resplit else 'rebuild'
    args=SimpleNamespace(config=str(config_path),out=str(output),ablations=[],
        encoder01=f'data/interim/encoder_01_{suffix}.npz',encoder02=f'data/interim/encoder_02_{suffix}.npz',
        h5_01=f'data/interim/Fluo-N3DH-CE_01_{suffix}.h5',h5_02=f'data/interim/Fluo-N3DH-CE_02_{suffix}.h5',
        gt_01='data/interim/Fluo-N3DH-CE_01.h5',gt_02='data/interim/Fluo-N3DH-CE_02.h5',
        exp_prefix=f'P7_ideas_v3_{name}',device='cuda',frames=None,seeds=[20261008,20261009],epochs=60,
        official=True,dataset='Fluo-N3DH-CE',official_gt_root='/root/autodl-tmp/ctc/raw',
        official_tools='/root/EvaluationSoftware/Linux',recalibrate=True,skip_baseline=False)
    plan=matrix.make_plan(args)
    job=next(j for j in plan['jobs'] if j['id'].endswith('seed20261008_eval02'))
    repeat=json.loads(json.dumps(job));old=repeat['id'];repeat['id']=old+'_repeat'
    repeat['command'][repeat['command'].index('--exp-id')+1]=repeat['id']
    repeat['outputs']=[p.replace(old,repeat['id']) for p in repeat['outputs']]
    plan['jobs'].append(repeat)
    plan['noise_protocol']='same configuration, detector H5, encoder NPZ and checkpoint; repeat complete seq02 export and official evaluation'
    (output/'plan.json').write_text(json.dumps(plan,indent=2))
    (output/'source_release.json').write_text((ROOT/'SOURCE_VERSION.json').read_text())
    plans.append((plan,output,resplit))
    if not resplit:
        matrix.execute_plan({**plan,'jobs':plan['jobs'][:2]},output)
        report=json.loads((output/'calibration/baseline.json').read_text())
        penalty=configuration.gnn.lambda_ot*report['suggestions']['theta_c']
        print(json.dumps({'profile':name,'lambda_times_true_cost_p999':penalty}),flush=True)
        if penalty>1:raise RuntimeError('真实边 OT 惩罚过大；先重新选择量纲一致的 λ 再训练')
print('CPU preparation complete; waiting for encoder queue',flush=True)
while not (ROOT/'logs/ideas_frontend_features_20261009.complete').exists():
    pid=int((ROOT/'logs/ideas_frontend_prepare_20261009.pid').read_text())
    try:
        import os
        os.kill(pid,0)
    except ProcessLookupError:
        raise RuntimeError('前端准备队列中断；检查其日志后恢复')
    time.sleep(10)
for plan,output,resplit in plans:
    if resplit:
        matrix.execute_plan({**plan,'jobs':plan['jobs'][:2]},output)
        report=json.loads((output/'calibration/baseline.json').read_text())
        penalty=base.gnn.lambda_ot*report['suggestions']['theta_c']
        print(json.dumps({'profile':'conservative_k16','lambda_times_true_cost_p999':penalty}),flush=True)
        if penalty>1:raise RuntimeError('再切检测的真实边 OT 惩罚过大，请重新标定 λ')
    matrix.execute_plan(plan,output)
(ROOT/'logs/ideas_final_v3_20261009.complete').write_text('complete\n')
