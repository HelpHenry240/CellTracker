#!/usr/bin/env python3
"""双序列、多种子消融执行器；计划生成不启动训练、不访问云服务器。

--execute 在当前机器顺序执行计划。完整任务在云端 nohup 后台启动；本地
只运行小样本 smoke。每个变体/种子独立训练，官方评测失败会停止队列。
完成状态只在输出校验后写入，--resume 不重跑已完成任务。
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
PKG=Path(__file__).resolve().parents[1]
ROOT=PKG.parent
sys.path.insert(0,str(PKG/'src'))
import papertrack  # noqa: E402,F401
from papertrack.config import load_config,override,save_config
from papertrack.runtime.ablation import variant,module_names


def make_plan(args):
    base=load_config(args.config)
    output=Path(args.out).resolve()
    output.mkdir(parents=True,exist_ok=True)
    if (output/'plan.json').exists():
        raise FileExistsError('计划已存在；恢复请使用 --resume，或另建目录')
    config_dir=output/'configs'; config_dir.mkdir()
    jobs=[]; descriptions=[]
    names=[*args.ablations] if getattr(args,'skip_baseline',False) else ['baseline',*args.ablations]
    for name in names:
        cfg,description=(base,{'module':'baseline','direction':'baseline'}) if name=='baseline' else variant(base,name)
        descriptions.append(description)
        seq_config={}
        for seq,encoder in [('01',args.encoder01),('02',args.encoder02)]:
            sequence=cfg
            if cfg.node.f_source=='encoder_npz':
                if not encoder:
                    raise ValueError(f'{seq} 序列缺少 encoder 路径')
                sequence=override(cfg,[f'node.encoder_feat_path={encoder}'])
            seq_config[seq]=config_dir/f'{name}_{seq}.yaml'
            save_config(sequence,seq_config[seq])
        calibration_id=None
        if getattr(args,'recalibrate',False) and cfg.coupling.enabled:
            calibration_id=f'{args.exp_prefix}_{name}_calibration01'
            calibrated=config_dir/f'{name}_calibrated.yaml'
            report=output/'calibration'/f'{name}.json'
            command=[sys.executable,str(PKG/'scripts/calibrate_params.py'),'--h5',args.h5_01,
                '--gt-h5',args.gt_01,'--config',str(seq_config['01']),'--out',str(report),
                '--apply-out',str(calibrated),'--cache-dir',str(output/'calibration_cache'/name)]
            if args.frames:
                command+=['--frames',args.frames]
            jobs.append({'id':calibration_id,'command':command,'outputs':[str(report),str(calibrated)],
                         'kind':'calibrate','depends':[]})
            seq_config={'01':calibrated,'02':calibrated}
            description['threshold_protocol']='recalibrated_on_seq01_only'
        graphs=output/'graphs'/name
        cache=output/'cache'/name
        def pipeline_command(seq,identifier):
            h5,gt=(args.h5_01,args.gt_01) if seq=='01' else (args.h5_02,args.gt_02)
            command=[sys.executable,str(PKG/'scripts/run_paper_pipeline.py'),'--h5',h5,'--gt-h5',gt,
                '--seq',seq,'--exp-id',identifier,'--config',str(seq_config[seq]),
                '--cache-dir',str(cache/seq),'--device',args.device]
            if args.frames:
                command+=['--frames',args.frames]
            if cfg.node.f_source=='encoder_npz':
                command+=['--set',f'node.encoder_feat_path={args.encoder01 if seq=="01" else args.encoder02}']
            return command
        if cfg.gnn.enabled:
            build=f'{args.exp_prefix}_{name}_build01'
            command=pipeline_command('01',build)+['--dump-graphs',str(graphs),'--build-only']
            jobs.append({'id':build,'command':command,'outputs':[str(graphs/'manifest.json')],
                         'kind':'build','depends':[calibration_id] if calibration_id else []})
        for seed in args.seeds:
            train_id=f'{args.exp_prefix}_{name}_seed{seed}_train'
            model=output/'models'/name/f'seed{seed}'
            if cfg.gnn.enabled:
                command=[sys.executable,str(PKG/'scripts/train_paper_gnn.py'),'--graphs',str(graphs),
                    '--out',str(model),'--config',str(seq_config['01']),'--epochs',str(args.epochs),
                    '--seed',str(seed),'--device',args.device,'--exp-id',train_id]
                jobs.append({'id':train_id,'command':command,'outputs':[str(model/'best.pt'),str(model/'train_summary.json')],
                             'kind':'train','depends':[build]})
            for seq in ['01','02']:
                identifier=f'{args.exp_prefix}_{name}_seed{seed}_eval{seq}'
                command=pipeline_command(seq,identifier)
                if cfg.gnn.enabled:
                    command+=['--ckpt',str(model/'best.pt')]
                expected=ROOT/'experiments'/identifier/'metrics.json'
                if args.official:
                    gt_dir=Path(args.official_gt_root)/args.dataset/f'{seq}_GT'
                    command+=['--official','--official-tools',args.official_tools,'--official-gt-dir',str(gt_dir)]
                    expected=ROOT/'experiments'/identifier/f'metrics_official_{seq}.json'
                jobs.append({'id':identifier,'command':command,'outputs':[str(expected)],
                             'kind':'evaluate','depends':[train_id] if cfg.gnn.enabled else ([calibration_id] if calibration_id else [])})
    plan={'schema':'ideas-ablation-v1','dataset':args.dataset,'protocol':'seq01 train / seq02 held out',
          'seeds':args.seeds,'variants':descriptions,'official':args.official,'jobs':jobs,
          'cloud_note':'云端长任务须 nohup 后台；启动前/训练中/结束后检查 nvidia-smi',
          'performance_conclusion':'需双序列官方 DET/SEG/TRA 与当前设置重复实验噪声地板'}
    (output/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2))
    return plan


def execute_plan(plan,root):
    status_path=root/'status.json'
    status=json.loads(status_path.read_text()) if status_path.exists() else {}
    logs=root/'logs'; logs.mkdir(exist_ok=True)
    def persist():
        pending=status_path.with_suffix('.pending.json')
        pending.write_text(json.dumps(status,ensure_ascii=False,indent=2)); pending.replace(status_path)
    for job in plan['jobs']:
        identifier=job['id']
        if status.get(identifier,{}).get('status')=='complete':
            if not all(Path(path).is_file() and Path(path).stat().st_size for path in job['outputs']):
                raise RuntimeError(f'已完成任务的产物缺失：{identifier}')
            continue
        if any(status.get(parent,{}).get('status')!='complete' for parent in job['depends']):
            raise RuntimeError(f'前置任务未完成：{identifier}')
        command=list(job['command'])
        if job['kind']=='build' and status.get(identifier,{}).get('status') in {'running','failed'}:
            command+=['--resume-graphs']
        if job['kind']=='train' and (Path(command[command.index('--out')+1])/'last.pt').exists():
            command+=['--resume']
        status[identifier]={'status':'running','command':command,'log':str(logs/f'{identifier}.log')}
        persist()
        print(f'[{identifier}] start',flush=True)
        uses_gpu=job['kind'] in {'train','evaluate'} and '--device' in command and command[command.index('--device')+1].startswith('cuda')
        if uses_gpu:
            before=subprocess.run(['nvidia-smi'],capture_output=True,text=True,check=True)
            (logs/f'{identifier}_gpu_before.txt').write_text(before.stdout)
        with (logs/f'{identifier}.log').open('a') as stream:
            process=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            status[identifier]['pid']=process.pid; persist()
            observed=False
            while process.poll() is None:
                if uses_gpu:
                    probe=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory',
                        '--format=csv,noheader'],capture_output=True,text=True)
                    if any(row.strip().startswith(f'{process.pid},') for row in probe.stdout.splitlines()):
                        observed=True
                        (logs/f'{identifier}_gpu_active.txt').write_text(probe.stdout)
                time.sleep(1)
            code=process.returncode
        if uses_gpu:
            after=subprocess.run(['nvidia-smi'],capture_output=True,text=True)
            (logs/f'{identifier}_gpu_after.txt').write_text(after.stdout)
            status[identifier]['gpu_process_observed']=observed
            if not observed:
                code=code or 1
        ready=all(Path(path).is_file() and Path(path).stat().st_size for path in job['outputs'])
        status[identifier].update(status='complete' if code==0 and ready else 'failed',returncode=code)
        persist()
        if status[identifier]['status']!='complete':
            raise RuntimeError(f'{identifier} 失败；查看 {logs/identifier}.log 后再恢复')
        print(f'[{identifier}] complete',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--h5-01',required=True)
    parser.add_argument('--h5-02',required=True)
    parser.add_argument('--gt-01',required=True)
    parser.add_argument('--gt-02',required=True)
    parser.add_argument('--encoder01')
    parser.add_argument('--encoder02')
    parser.add_argument('--out',required=True)
    parser.add_argument('--exp-prefix',default='P7_ideas_rebuild')
    parser.add_argument('--dataset',default='Fluo-N3DH-CE')
    parser.add_argument('--ablations',nargs='*',default=['fgw','motion','multiscale','tracklet','gnn'])
    parser.add_argument('--seeds',nargs='+',type=int,default=[20261008,20261009])
    parser.add_argument('--epochs',type=int,default=60)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--frames')
    parser.add_argument('--official',action='store_true')
    parser.add_argument('--official-tools',default='/root/EvaluationSoftware/Linux')
    parser.add_argument('--official-gt-root',default='/root/autodl-tmp/ctc/raw')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--recalibrate',action='store_true',help='每个变体在 seq01 重新标定绝对质量/代价阈值')
    parser.add_argument('--skip-baseline',action='store_true',help='仅用于已有同配置双序列 baseline 证据的后续矩阵')
    args=parser.parse_args()
    if args.official and args.frames:
        parser.error('官方结论必须使用完整序列')
    if len(set(args.seeds))!=len(args.seeds) or len(set(args.ablations))!=len(args.ablations):
        parser.error('种子和消融项不能重复')
    if not set(args.ablations)<=set(module_names()):
        parser.error('存在未知消融项')
    if args.execute and args.device=='cpu' and not args.frames:
        parser.error('本地 CPU 执行仅限指定 --frames 的小样本；完整队列请在云端执行')
    root=Path(args.out).resolve()
    plan=json.loads((root/'plan.json').read_text()) if args.resume else make_plan(args)
    print(f'计划：{root/"plan.json"}；{len(plan["jobs"])} 个任务',flush=True)
    if args.execute:
        execute_plan(plan,root)

if __name__=='__main__':
    main()
