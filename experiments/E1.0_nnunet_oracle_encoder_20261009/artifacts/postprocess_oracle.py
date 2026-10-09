"""等待正式矩阵完整结束后汇总、导出小权重和打包；失败时保留原文件并退出。"""
from pathlib import Path
import json
import subprocess
import sys
import tarfile
import time

root=Path('/root/CellTracker_nnunet_oracle_encoder_20261009')
helper=root/'experiments/E1.0_nnunet_oracle_encoder_20261009/artifacts'
started=time.monotonic()
while not (root/'logs/oracle_encoder_pipeline.complete').is_file():
    pid=int((root/'logs/oracle_encoder_pipeline.pid').read_text())
    if not Path(f'/proc/{pid}').exists():
        raise RuntimeError('正式矩阵未完成且主进程已退出；检查日志后恢复')
    if time.monotonic()-started>3600:
        raise TimeoutError('等待完整正式矩阵超时')
    time.sleep(10)
subprocess.run([sys.executable,str(helper/'summarize_oracle.py'),'--root',str(root),
                '--repository',str(root),'--out',str(root/'logs/oracle_metrics_final.json')],check=True)
models=root/'oracle_inference'
subprocess.run([sys.executable,str(helper/'export_inference_weights.py'),'--root',str(root),
                '--repository',str(root),'--out',str(models)],check=True)
archive=Path('/root/mask_oracle_inference_20261009.tar.gz')
with tarfile.open(archive,'w:gz') as handle:
    for path in sorted(models.rglob('*')):
        if path.is_file():handle.add(path,arcname=str(path.relative_to(models)),recursive=False)
subprocess.run([sys.executable,str(helper/'collect_oracle.py'),'--root',str(root),
                '--out','/root/mask_oracle_evidence_20261009.tar.gz'],check=True)
(root/'logs/oracle_postprocess.complete').write_text('complete\n')
print(json.dumps({'postprocess_complete':True,'models_tar_bytes':archive.stat().st_size}))
