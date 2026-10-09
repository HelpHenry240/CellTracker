#!/bin/bash
set -euo pipefail
cd /root/autodl-tmp/CellTracker_rebuild_20261008
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
PY=/root/autodl-tmp/nnunet/venv/bin/python
OUT=experiments/P7_ideas_matched_baseline_matrix_20261009
ARGS=(--config experiments/P7_ideas_calibration01_v2_20261009/calibrated.yaml
 --h5-01 data/interim/Fluo-N3DH-CE_01_rebuild.h5 --h5-02 data/interim/Fluo-N3DH-CE_02_rebuild.h5
 --gt-01 data/interim/Fluo-N3DH-CE_01.h5 --gt-02 data/interim/Fluo-N3DH-CE_02.h5
 --encoder01 data/interim/encoder_01_rebuild.npz --encoder02 data/interim/encoder_02_rebuild.npz
 --out "$OUT" --exp-prefix P7_ideas_control --ablations --seeds 20261008 20261009
 --epochs 60 --device cuda --official --recalibrate)
"$PY" -u paperpipe/scripts/run_ablation_matrix.py "${ARGS[@]}"
cp SOURCE_VERSION.json "$OUT/source_release.json"
"$PY" -u - <<'PY'
import importlib.util,json
from pathlib import Path
root=Path('experiments/P7_ideas_matched_baseline_matrix_20261009').resolve()
plan=json.loads((root/'plan.json').read_text())
original=next(j for j in plan['jobs'] if j['id']=='P7_ideas_control_baseline_seed20261008_eval02')
noise=json.loads(json.dumps(original)); old=noise['id']; noise['id']=old+'_repeat'
noise['command'][noise['command'].index('--exp-id')+1]=noise['id']
noise['outputs']=[p.replace(old,noise['id']) for p in noise['outputs']]
plan['jobs'].append(noise)
plan['noise_protocol']='identical config, checkpoint and inputs; independent complete seq02 export and official rerun'
(root/'plan.json').write_text(json.dumps(plan,indent=2))
spec=importlib.util.spec_from_file_location('matrix',Path('paperpipe/scripts/run_ablation_matrix.py').resolve())
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.execute_plan({**plan,'jobs':plan['jobs'][:2]},root)
PY
printf 'prepared\n' > logs/ideas_control_20261009.prepared
while ! test -f logs/ideas_next_20261009.complete; do
 if ! kill -0 "$(cat logs/ideas_next_20261009.pid)" 2>/dev/null; then
  echo 'Ablation queue stopped; inspect its log before GPU continuation'; exit 1
 fi
 sleep 10
done
"$PY" -u paperpipe/scripts/run_ablation_matrix.py "${ARGS[@]}" --resume --execute
printf 'complete\n' > logs/ideas_control_20261009.complete
