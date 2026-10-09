#!/usr/bin/env python3
"""ideas pipeline 主入口：完整推理、监督建图、消融配置和 CTC 导出。"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
PKG = Path(__file__).resolve().parents[1]
ROOT = PKG
sys.path.insert(0,str(PKG/'src'))
import papertrack  # noqa: E402,F401
from celltracker.experiment import Experiment
from papertrack.config import load_config,override,save_config
from papertrack.reconstruction import export_ctc
from papertrack.runtime import run_pipeline
from papertrack.runtime.ablation import describe,variant
from papertrack.runtime.validate import validate_ctc_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--h5')
    parser.add_argument('--gt-h5')
    parser.add_argument('--seq')
    parser.add_argument('--dataset',default='Fluo-N3DH-CE')
    parser.add_argument('--exp-id')
    parser.add_argument('--config',default=str(PKG/'configs/paper_default.yaml'))
    parser.add_argument('--set',nargs='*',default=[])
    parser.add_argument('--ablate',nargs='*',default=[])
    parser.add_argument('--list-modules',action='store_true')
    parser.add_argument('--ckpt')
    parser.add_argument('--frames',help='小样本的闭区间，例如 120:130')
    parser.add_argument('--dump-graphs')
    parser.add_argument('--build-only',action='store_true')
    parser.add_argument('--resume-graphs',action='store_true',help='检查已有图后补齐中断的建图任务')
    parser.add_argument('--cache-dir',help='带配置与输入指纹的帧对缓存')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--official',action='store_true')
    parser.add_argument('--official-tools',help='在当前机器调用已有官方二进制的目录')
    parser.add_argument('--official-gt-dir',help='当前机器的 <seq>_GT 目录')
    parser.add_argument('--cloud-gt-root')
    parser.add_argument('--gt-seg-dir')
    parser.add_argument('--no-validate',action='store_true',help='仅非官方调试可跳过校验')
    args = parser.parse_args()
    cfg = override(load_config(args.config),args.set)
    changes = []
    for name in args.ablate:
        cfg,change = variant(cfg,name)
        changes.append(change)
    if args.list_modules:
        print(json.dumps(describe(cfg),ensure_ascii=False,indent=2))
        return
    if not all([args.h5,args.seq,args.exp_id]):
        parser.error('--h5、--seq、--exp-id 必须提供')
    if args.build_only and not args.dump_graphs:
        parser.error('--build-only 需要 --dump-graphs')
    if args.official and (args.no_validate or args.frames or args.build_only):
        parser.error('官方评测必须执行完整序列导出和格式校验')
    if args.official_tools and not args.official_gt_dir:
        parser.error('--official-tools 需要 --official-gt-dir')
    frames = None
    if args.frames:
        low,separator,high = args.frames.partition(':')
        if not separator:
            parser.error('--frames 格式应为 start:end')
        frames = list(range(int(low),int(high)+1))
    exp = Experiment(args.exp_id,'ideas pipeline 重建链路',root=ROOT,resume=args.resume_graphs,
                     params={**vars(args),'seed':cfg.seed,'ablations':changes})
    save_config(cfg,exp.dir/'pipeline_config.yaml')
    try:
        run = run_pipeline(args.h5,cfg,gt_h5=args.gt_h5,ckpt=args.ckpt,frames=frames,
            artifacts_dir=exp.artifact_dir('pipeline'),dump_graphs=args.dump_graphs,
            device=args.device,cache_dir=args.cache_dir,resume_graphs=args.resume_graphs)
        if args.build_only:
            exp.save_metrics({'status':'graphs_built','pipeline':run.info})
        else:
            res_dir = exp.artifact_dir('submission')/f'{args.seq}_RES'
            seg_dir = args.gt_seg_dir or ROOT/'data/raw'/args.dataset/f'{args.seq}_GT/SEG'
            stats = export_ctc(run,args.h5,res_dir,seq=args.seq,gt_h5=args.gt_h5,gt_seg_dir=seg_dir)
            if not args.no_validate:
                check = validate_ctc_dir(res_dir,expected_frames=run.dets.t_range)
                stats['format_validation'] = check
                if not check['ok']:
                    raise ValueError(f'CTC 格式校验失败：{check["errors"]}')
            exp.save_metrics(stats)
            if args.official:
                official_path = exp.dir/f'metrics_official_{args.seq}.json'
                if args.official_tools:
                    from papertrack.runtime.official import evaluate_official
                    official = evaluate_official(res_dir,args.official_gt_dir,args.official_tools,args.seq,official_path)
                else:
                    command = [os.environ.get('CT_CLOUD_PYTHON',sys.executable),str(ROOT/'scripts/cloud_eval.py'),
                        '--res-dir',str(res_dir),'--dataset',args.dataset,'--seq',args.seq,'--out',str(official_path)]
                    if args.cloud_gt_root:
                        command += ['--cloud-gt-root',args.cloud_gt_root]
                    process = subprocess.run(command,capture_output=True,text=True)
                    exp.log(process.stdout[-3000:])
                    if process.returncode:
                        raise RuntimeError(f'云端官方评测失败：{process.stderr[-2000:]}')
                    official = json.loads(official_path.read_text())
                if any(official.get(key) is None for key in ['DET','SEG','TRA']):
                    raise RuntimeError('官方结果缺少 DET/SEG/TRA，不能标记成功')
        _write_overview(exp,run.info,changes)
        exp.finish(summary=f'tracks={run.result.n_tracks()} decision={run.info["decision"]}')
    except Exception as error:
        exp.finish(status='failed',summary=str(error))
        raise


def _write_overview(exp,info,changes):
    lines = [f'{key}: {value:.3f}s' for key,value in info['stage_seconds'].items()]
    svg = '<svg xmlns="http://www.w3.org/2000/svg" width="700" height="300"><rect width="100%" height="100%" fill="white"/>'
    for index,line in enumerate(lines):
        svg += f'<text x="20" y="{30+index*25}" font-family="monospace" font-size="15">{line}</text>'
    (exp.dir/'figures/pipeline_stages.svg').write_text(svg+'</svg>')
    (exp.dir/'notes.md').write_text('本实验用于验证 ideas pipeline。\n\n'
        f'决策路径：{info["decision"]}；检测来源：{info["detection_source"]}。\n'
        f'消融：{json.dumps(changes,ensure_ascii=False)}。\n'
        '本地诊断只用于归因；方法效果须结合双序列官方指标和当前设置的噪声地板。\n')

if __name__ == '__main__':
    main()
