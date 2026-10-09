"""按实际文件、任务状态、设备记录与模型指纹验收正式Oracle实验。"""
from pathlib import Path
import hashlib,json,subprocess

root=Path('/root/CellTracker_nnunet_oracle_encoder_20261009')
release=json.loads((root/'SOURCE_VERSION.json').read_text())
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==sha for name,sha in release['files'].items())
matrix=root/'experiments/P9_nnunet_oracle_encoder_matrix_20261009'
plan=json.loads((matrix/'plan.json').read_text());state=json.loads((matrix/'status.json').read_text())
assert len(plan['jobs'])==9 and all(state[job['id']]['status']=='complete' for job in plan['jobs'])
gpu=[job for job in plan['jobs'] if job['kind'] in {'train','evaluate'}]
assert all(state[job['id']]['gpu_process_observed'] for job in gpu)
for flag in ('oracle_encoder_pipeline.complete','oracle_postprocess.complete'):
    assert (root/'logs'/flag).read_text().strip()=='complete'
frozen=json.loads((root/'logs/oracle_frozen_model_identity.json').read_text())
assert frozen['unchanged'] and frozen['before']==frozen['after']
models=json.loads((root/'oracle_inference/manifest.json').read_text())
assert len(models)==2 and all(row['tensor_identity_verified'] and row['bytes']>0 for row in models)
active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True)
assert not active.strip(), '仍有GPU任务，不能宣告空闲'
receipt={'completed_jobs':9,'official_evaluations':5,'gpu_jobs_observed':len(gpu),
         'source_files_verified':len(release['files']),'source_commit':release['git_commit'],
         'inference_models_tensor_verified':2,'unet_model_unchanged':True,'GPU_idle':True}
out=root/'logs/oracle_final_acceptance.json'
if out.exists():raise FileExistsError(out)
out.write_text(json.dumps(receipt,indent=2))
(root/'logs/oracle_gpu_final_idle.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
print(json.dumps(receipt))
