import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path.cwd()/'paperpipe/src'))
import papertrack
from papertrack.config import load_config,override,save_config
from papertrack.runtime import run_pipeline,load_gt_parent
from papertrack.runtime.calibration import calibrate_run
from papertrack.reconstruction import export_ctc
from papertrack.runtime.validate import validate_ctc_dir
from papertrack.gnn.train import TrainConfig,train
from celltracker.experiment import Experiment
exp=Experiment('E0.5_ideas_rebuild_local_20261009','本地双序列小样本全链路验证；云端暂停',
    params={'frames':'120:125','detection_source':'nnunet_pred','seed':20261009,
            'appearance':'intensity engineering smoke','official':'paused'})
cfg=override(load_config('paperpipe/configs/paper_e2e_ce.yaml'),['node.f_source=intensity',
    'node.encoder_feat_path=null','coupling.fgw_outer=3','coupling.sinkhorn_iters=200',
    'multiscale.n_rounds=1','gnn.hidden=16','gnn.layers=3','seed=20261009'])
save_config(cfg,exp.dir/'pipeline_config.yaml')
metrics={}
for seq in ['01','02']:
 h5=f'data/interim/rebuild_local_smoke_{seq}_20261009.h5'
 gt=f'data/interim/Fluo-N3DH-CE_{seq}.h5'
 graphs=Path(f'data/interim/ideas_rebuild_local_graphs_{seq}_20261009')
 run=run_pipeline(h5,cfg,gt_h5=gt,dump_graphs=graphs,
     cache_dir=f'data/interim/ideas_rebuild_local_cache_{seq}_20261009',
     artifacts_dir=exp.artifact_dir(f'pipeline_{seq}'))
 res=exp.artifact_dir('ot_submission')/f'{seq}_RES'
 diagnostic=export_ctc(run,h5,res,seq=seq,gt_h5=gt)
 check=validate_ctc_dir(res,expected_frames=run.dets.t_range)
 if not check['ok']: raise ValueError(check)
 suggested,calibration=calibrate_run(run,load_gt_parent(gt))
 (exp.dir/f'calibration_smoke_{seq}.json').write_text(json.dumps(calibration,ensure_ascii=False,indent=2))
 metrics[seq]={'OT_format':check,'pipeline':run.info,'local_diagnostic':diagnostic}
 if seq=='01':
  training=train(graphs,TrainConfig(epochs=2,hidden=16,layers=3,lambda_ot=cfg.gnn.lambda_ot,
      val_fraction=0,batch_pairs=1,seed=cfg.seed,out_dir=str(exp.artifact_dir('model'))))
  metrics['training_smoke']={k:v for k,v in training.items() if k!='history'}
 predicted=run_pipeline(h5,cfg,ckpt=exp.dir/'artifacts/model/best.pt',
     cache_dir=f'data/interim/ideas_rebuild_local_cache_{seq}_20261009',verbose=True)
 res=exp.artifact_dir('gnn_submission')/f'{seq}_RES'
 export_ctc(predicted,h5,res,seq=seq,gt_h5=gt)
 check=validate_ctc_dir(res,expected_frames=predicted.dets.t_range)
 if not check['ok']: raise ValueError(check)
 metrics[seq]['GNN_format']=check
metrics.update(status='local_smoke_passed_cloud_paused',cloud_tasks='全量 GNN / 消融 / 官方指标暂停',
    limitation='6 帧强度外观接口冒烟；不是完整序列或模型精度结论')
exp.save_metrics(metrics)
(exp.dir/'notes.md').write_text('seq01+seq02 各 120–125 帧；真实预测实例。\n'
 '全部追踪模块开启；外观使用 intensity 工程近似，不宣称是 encoder 完整性能验证。\n'
 'GNN 仅两轮训练，无验证集，作用是验证接口；训练 seq01，推理两序列。\n'
 '官方指标和全量训练因用户暂停云服务器而未执行。\n')
svg='<svg xmlns="http://www.w3.org/2000/svg" width="850" height="220"><rect width="100%" height="100%" fill="white"/>'
for idx,seq in enumerate(['01','02']):
 seconds=metrics[seq]['pipeline']['total_seconds']
 svg+=f'<text x="20" y="{40+idx*45}" font-size="18">CE {seq}: OT/GNN format PASS; 6 frames; build {seconds:.2f}s</text>'
(exp.dir/'figures/local_smoke.svg').write_text(svg+'<text x="20" y="160" font-size="18">Official evaluation: paused</text></svg>')
exp.finish(summary='本地双序列小样本链路通过；云端全量实验暂停')
