"""绘制两个独立种子的官方结果、误差归属与候选安全漏斗；不比较不同特征。"""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def save(fig,out,name):
    fig.savefig(out/f'{name}.svg',bbox_inches='tight')
    fig.savefig(out/f'{name}.png',dpi=200,bbox_inches='tight');plt.close(fig)


def plot(summary,safety,output):
    result=json.loads(Path(summary).read_text());candidate=json.loads(Path(safety).read_text())
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(7.5,4))
    for i,seq in enumerate(('01','02')):
        rows=[row for row in result['runs'] if row['seq']==seq and not row['repeat']]
        scores=np.array([row['TRA'] for row in rows]);mean=scores.mean()
        ax.errorbar(i,mean,yerr=[[mean-scores.min()],[scores.max()-mean]],fmt='o',capsize=6,color='#157b88')
        ax.scatter(i+np.array([-.07,.07]),scores,s=32,color=['#4e5b78','#b0672f'])
        ax.text(i, scores.max()+.00012,f'{mean:.7f}',ha='center')
    ax.set_xticks([0,1],['Sequence 01 / train source','Sequence 02 / GNN holdout'])
    ax.set_ylabel('Official TRA');ax.grid(axis='y',alpha=.2);ax.ticklabel_format(axis='y',useOffset=False)
    ax.set_title('GT markers + frozen encoder / two independently trained seeds')
    fig.tight_layout();save(fig,out,'oracle_official_TRA')
    fig,ax=plt.subplots(figsize=(7.5,4))
    bottom=np.zeros(4)
    rows=[row for row in result['runs'] if not row['repeat']]
    for key,weight in {'NS':5,'FN':10,'FP':1,'ED':1,'EA':1.5,'EC':1}.items():
        values=np.array([row['AOGM'][key]*weight for row in rows]);ax.bar(range(4),values,bottom=bottom,label=key);bottom+=values
    ax.set_xticks(range(4),[f"Seq{row['seq']} / {row['seed']}" for row in rows],rotation=15)
    ax.set_ylabel('Weighted official edit counts');ax.legend(ncol=3,frameon=False)
    ax.set_title('AOGM attribution, same GT-marker detection input');fig.tight_layout();save(fig,out,'oracle_AOGM')
    fig,ax=plt.subplots(figsize=(7.5,4))
    ks=[int(k) for k in candidate['counts']]
    for name,label in [('true_recall','All true edges'),('division_recall','Division edges')]:
        ax.plot(ks,[candidate['counts'][str(k)][name] for k in ks],marker='o',label=label)
    ax.axvline(candidate['selected_topk'],color='#536878',linestyle='--',label='Existing default k=3, frozen before training')
    ax.axhline(.99,color='gray',linestyle=':');ax.set_xlabel('Existing candidate top-k fallback');ax.set_ylabel('Retained true edges / all true edges')
    ax.set_title('Seq01 candidate safety only / no official-score parameter search');ax.legend(frameon=False)
    fig.tight_layout();save(fig,out,'oracle_candidate_safety')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--summary',required=True);p.add_argument('--safety',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();plot(a.summary,a.safety,a.out)
