"""绘制官方指标的双种子范围、AOGM归属和检测覆盖；不把范围当置信区间。"""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def save(fig,out,name):
    fig.savefig(out/f'{name}.svg',bbox_inches='tight');fig.savefig(out/f'{name}.png',dpi=200,bbox_inches='tight');plt.close(fig)


def plot(summary,audit,output):
    result=json.loads(Path(summary).read_text());inputs=json.loads(Path(audit).read_text());out=Path(output);out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,key in zip(axes,('DET','SEG','TRA')):
        for i,seq in enumerate(('01','02')):
            scores=np.array([r[key] for r in result['runs'] if r['seq']==seq and not r['repeat']]);mean=scores.mean()
            ax.errorbar(i,mean,yerr=[[mean-scores.min()],[scores.max()-mean]],fmt='o',color='#167988',capsize=5)
            ax.scatter(i+np.array([-.07,.07]),scores,s=24,color=['#566479','#b17b3d'])
            ax.annotate(f'{mean:.6f}',(i,mean),xytext=(0,12),textcoords='offset points',ha='center')
        ax.set_title('Official '+key);ax.set_xticks([0,1],['Seq01','Seq02']);ax.grid(axis='y',alpha=.2);ax.ticklabel_format(axis='y',useOffset=False)
    fig.suptitle('nnU-Net mask + GT seed Oracle + encoder / two independent GNN seeds');fig.tight_layout();save(fig,out,'official_metrics')
    fig,ax=plt.subplots(figsize=(8,4));rows=[r for r in result['runs'] if not r['repeat']];bottom=np.zeros(4)
    for key,weight in {'NS':5,'FN':10,'FP':1,'ED':1,'EA':1.5,'EC':1}.items():
        values=np.array([r['AOGM'][key]*weight for r in rows]);ax.bar(range(4),values,bottom=bottom,label=key);bottom+=values
    ax.set_xticks(range(4),[f"Seq{r['seq']} / {r['seed']}" for r in rows],rotation=12)
    ax.set_ylabel('Weighted official edit counts');ax.set_title('Official AOGM attribution');ax.legend(ncol=3,frameon=False);fig.tight_layout();save(fig,out,'official_AOGM')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,seq in zip(axes,('01','02')):
        rows=inputs['sequences'][seq]['frames'];ts=[r['frame'] for r in rows]
        for key,label,color in [('gt_nodes','GT marker identities','#444444'),('nodes','Oracle predicted instances','#247d84'),('matched_gt_nodes','Covered GT identities','#b47b3b')]:
            ax.plot(ts,[r[key] for r in rows],label=label,color=color)
        ax.set_title(f'Seq{seq} / raw foreground preserved');ax.set_xlabel('Frame');ax.grid(alpha=.2)
    axes[0].set_ylabel('Instances or identities');axes[1].legend(frameon=False);fig.tight_layout();save(fig,out,'oracle_identity_coverage')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--summary',required=True);p.add_argument('--audit',required=True);p.add_argument('--out',required=True);a=p.parse_args();plot(a.summary,a.audit,a.out)
