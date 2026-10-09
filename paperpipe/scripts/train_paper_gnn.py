#!/usr/bin/env python3
"""训练式(30)–(35) 的 GNN；从 pipeline 配置读取结构、损失和随机种子。"""
from __future__ import annotations
import argparse
import json
import sys
import hashlib
from pathlib import Path
PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PKG/'src'))
import papertrack  # noqa: E402,F401
from papertrack.config import load_config
from papertrack.gnn.train import TrainConfig,train
from papertrack.runtime.contracts import file_hash
from celltracker.experiment import Experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graphs',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--config',default=None)
    parser.add_argument('--epochs',type=int,default=60)
    parser.add_argument('--batch-pairs',type=int,default=8)
    parser.add_argument('--hidden',type=int,default=None)
    parser.add_argument('--layers',type=int,default=None)
    parser.add_argument('--lr',type=float,default=1e-3)
    parser.add_argument('--lambda-ot',type=float,default=None)
    parser.add_argument('--residual',action='store_true',default=None)
    parser.add_argument('--class-weighted',action='store_true',default=None)
    parser.add_argument('--seed',type=int,default=None)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--val-fraction',type=float,default=0.2)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--exp-id',default=None,help='训练阶段八件套编号；默认由输出路径生成')
    args = parser.parse_args()
    pipeline = load_config(args.config)
    if not pipeline.gnn.enabled:
        raise ValueError('GNN 关闭的配置不需要训练')
    gnn = pipeline.gnn
    config = TrainConfig(epochs=args.epochs,batch_pairs=args.batch_pairs,lr=args.lr,
        hidden=args.hidden if args.hidden is not None else gnn.hidden,
        layers=args.layers if args.layers is not None else gnn.layers,
        dropout=gnn.dropout,lambda_ot=args.lambda_ot if args.lambda_ot is not None else gnn.lambda_ot,
        residual=args.residual if args.residual is not None else gnn.residual,
        class_weighted_ce=args.class_weighted if args.class_weighted is not None else gnn.class_weighted_ce,
        seed=args.seed if args.seed is not None else pipeline.seed,device=args.device,
        val_fraction=args.val_fraction,out_dir=args.out,resume=args.resume)
    identity=hashlib.sha256(str(Path(args.out).resolve()).encode()).hexdigest()[:8]
    exp=Experiment(args.exp_id or f'GNN_{Path(args.out).name}_{identity}',
        'ideas GNN 独立训练',root=PKG.parent,resume=args.resume,params={**vars(args),'seed':config.seed})
    try:
        result = train(args.graphs,config)
    except Exception as error:
        exp.finish(status='failed',summary=str(error))
        raise
    summary = {key:value for key,value in result.items() if key != 'history'}
    summary['checkpoints'] = {name:{'bytes':(Path(args.out)/name).stat().st_size,
        'sha256':file_hash(Path(args.out)/name)} for name in ['best.pt','last.pt']}
    (Path(args.out)/'train_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    exp.save_metrics(summary)
    import shutil
    target=exp.artifact_dir('model')/'best.pt'
    if target.resolve()!=(Path(args.out)/'best.pt').resolve():
        shutil.copy2(Path(args.out)/'best.pt',target)
    (exp.dir/'notes.md').write_text('训练检测与推理来源必须同分布，权重保存输入契约。\n'
        f'训练图：{args.graphs}；模型：{args.out}；种子：{config.seed}。\n'
        '验证 F1 仅用于选择检查点，方法性能需双序列官方指标。\n')
    history=result['history']; points=' '.join(f'{20+i*560/max(len(history)-1,1):.2f},{180-160*row["f1"]:.2f}'
                                           for i,row in enumerate(history))
    (exp.dir/'figures/validation_f1.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg" width="600" height="200">'
        '<rect width="100%" height="100%" fill="white"/><polyline fill="none" stroke="blue" points="'+points+'"/></svg>')
    exp.finish(summary=f'best_f1={summary["best_f1"]:.5f}')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
