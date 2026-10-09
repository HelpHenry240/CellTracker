#!/bin/bash
set -euo pipefail
cd /root/autodl-tmp/CellTracker_rebuild_20261008
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PY=/root/autodl-tmp/nnunet/venv/bin/python
CFG=experiments/P7_ideas_calibration01_v2_20261009/calibrated.yaml
nvidia-smi > logs/next_gpu_before_20261009.txt
"$PY" - <<'PY'
import json
from pathlib import Path
original=json.loads(Path('experiments/P7_ideas_baseline_matrix_20261009/plan.json').read_text())
job=original['jobs'][-1]
old=job['id']; new=old+'_retry1'
job['id']=new; job['depends']=[]
command=job['command']; command[command.index('--exp-id')+1]=new
job['outputs']=[p.replace(old,new) for p in job['outputs']]
plan={**original,'jobs':[job],'recovery_of':old,'reason':'disk full during uncompressed export; completed results retained'}
out=Path('experiments/P7_ideas_baseline_recovery_20261009'); out.mkdir()
(out/'plan.json').write_text(json.dumps(plan,indent=2))
PY
COMMON=(--config "$CFG" --h5-01 data/interim/Fluo-N3DH-CE_01_rebuild.h5 --h5-02 data/interim/Fluo-N3DH-CE_02_rebuild.h5
 --gt-01 data/interim/Fluo-N3DH-CE_01.h5 --gt-02 data/interim/Fluo-N3DH-CE_02.h5
 --encoder01 data/interim/encoder_01_rebuild.npz --encoder02 data/interim/encoder_02_rebuild.npz
 --seeds 20261008 20261009 --epochs 60 --device cuda --official --execute)
"$PY" -u paperpipe/scripts/run_ablation_matrix.py "${COMMON[@]}" \
 --out experiments/P7_ideas_baseline_recovery_20261009 --resume
"$PY" -u paperpipe/scripts/run_ablation_matrix.py "${COMMON[@]}" \
 --out experiments/P7_ideas_ablation_matrix_20261009 --exp-prefix P7_ideas_ablate \
 --ablations isolated fgw multiscale motion tracklet ot_loss --skip-baseline --recalibrate
nvidia-smi > logs/next_gpu_after_20261009.txt
printf 'complete\n' > logs/ideas_next_20261009.complete
