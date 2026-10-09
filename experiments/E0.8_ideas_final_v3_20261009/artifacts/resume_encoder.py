"""从已完成的实例 H5 续跑冻结特征，直接等待旧版筛查结束，不重做 CPU 前端。"""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import time

ROOT = Path('/root/autodl-tmp/CellTracker_rebuild_20261008')
PYTHON = '/root/autodl-tmp/nnunet/venv/bin/python'
MODEL = Path('/root/autodl-tmp/nnunet/nnUNet_results/Dataset501_CellTrackerCE/nnUNetTrainer__nnUNetPlans__3d_fullres')
IMAGES = '/root/autodl-tmp/nnunet/nnUNet_raw/Dataset501_CellTrackerCE/imagesTs_eval'
os.chdir(ROOT)
os.environ['nnUNet_compile'] = 'false'


def model_hash():
    digest = hashlib.sha256()
    with (MODEL / 'fold_0/checkpoint_best.pth').open('rb') as file:
        for block in iter(lambda: file.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


before = model_hash()
while not (ROOT / 'logs/ideas_next_20261009.complete').exists():
    os.kill(int((ROOT / 'logs/ideas_next_20261009.pid').read_text()), 0)
    time.sleep(10)
for seq in ('01', '02'):
    output = ROOT / f'data/interim/encoder_{seq}_resplit_v2.npz'
    if output.exists():
        raise RuntimeError('已有完整特征文件，检查后另建恢复计划，不能静默覆盖')
    probe = subprocess.run(['nvidia-smi'], check=True, capture_output=True, text=True)
    (ROOT / f'logs/frontend_resume_{seq}_gpu_before.txt').write_text(probe.stdout)
    occupied = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    if occupied:
        raise RuntimeError(f'GPU 已被进程占用：{occupied}')
    process = subprocess.Popen([PYTHON, '-u', 'scripts/nnunet/export_encoder_features.py',
        '--h5', f'data/interim/Fluo-N3DH-CE_{seq}_resplit_v2.h5', '--images', IMAGES,
        '--model', str(MODEL), '--seq', seq, '--out', str(output), '--device', 'cuda'])
    (ROOT / f'logs/frontend_resume_{seq}.pid').write_text(str(process.pid))
    observed = False
    while process.poll() is None:
        active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'], text=True)
        if any(row.strip().startswith(f'{process.pid},') for row in active.splitlines()):
            observed = True
            with (ROOT / f'logs/frontend_resume_{seq}_gpu_active.txt').open('a') as log:
                log.write(active)
        time.sleep(2)
    if process.returncode or not observed or not output.is_file() or not output.stat().st_size:
        raise RuntimeError(f'{seq} 冻结 encoder 输出或 GPU 生效校验失败')
    (ROOT / f'logs/frontend_resume_{seq}_gpu_after.txt').write_text(subprocess.check_output(['nvidia-smi'], text=True))
after = model_hash()
if before != after:
    raise RuntimeError('冻结前向后 nnU-Net 权重发生变化')
(ROOT / 'logs/ideas_frozen_model_identity_20261009.json').write_text(json.dumps({'before_sha256': before,
    'after_sha256': after, 'unchanged': True, 'unet_retrained': False}, indent=2))
(ROOT / 'logs/ideas_frontend_features_20261009.complete').write_text('complete\n')
