#!/usr/bin/env python3
"""跑完整上游后标定绝对 Γ 和生死阈值，保存物理分布与候选损失。"""
import argparse
import json
import sys
from pathlib import Path
PKG=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PKG/'src'))
import papertrack  # noqa: E402,F401
from papertrack.config import load_config,override,save_config
from papertrack.runtime import run_pipeline,load_gt_parent
from papertrack.runtime.calibration import calibrate_run


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--h5',required=True)
    parser.add_argument('--gt-h5',required=True)
    parser.add_argument('--config',default=str(PKG/'configs/paper_default.yaml'))
    parser.add_argument('--set',nargs='*',default=[])
    parser.add_argument('--out',required=True)
    parser.add_argument('--apply-out',required=True,help='输出建议部署配置；官方验证前不能称为最优')
    parser.add_argument('--cache-dir')
    parser.add_argument('--frames',help='仅小样本调试使用，例如 100:110')
    args=parser.parse_args()
    output=Path(args.out); effective=Path(args.apply_out)
    if output.exists() or effective.exists():
        raise FileExistsError('标定结果只增不改，请换新输出路径')
    cfg=override(load_config(args.config),args.set)
    frames=None
    if args.frames:
        a,b=map(int,args.frames.split(':')); frames=list(range(a,b+1))
    run=run_pipeline(args.h5,cfg,gt_h5=args.gt_h5,frames=frames,cache_dir=args.cache_dir)
    suggested,report=calibrate_run(run,load_gt_parent(args.gt_h5))
    print(json.dumps(report,ensure_ascii=False,indent=2))
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    save_config(suggested,effective)

if __name__=='__main__':
    main()
