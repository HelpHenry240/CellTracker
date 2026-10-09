#!/usr/bin/env python3
"""流式再切预测实例，重新计算图像特征与完整 GT 身份，不改动原输入。

ideas.pdf §2.0.1 将欠分割描述为“多真实细胞黏连为单一大 mask”。此处复用
已有工程补充 refine_oversized_instances（C5.0e），不是论文新增的分割机制：
体积超过同帧中位数 k 倍时，用 0.5×原 h_frac 在局部包围盒中再切。
--k 0 关闭再切，是同来源前端对照。已有标签之外的背景不重新预测。
"""
from pathlib import Path
import argparse,hashlib,json,sys,time
import h5py
import numpy as np
import tifffile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from celltracker.data.ctc import object_table_from_labels
from celltracker.detect.instances import InstanceSplitConfig,refine_oversized_instances
from celltracker.detect.labels import match_gt_labels


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def resplit(input_path,gt_path,img_dir,out_path,k,resume=False):
    out=Path(out_path); pending=out.with_suffix('.pending.h5')
    if out.exists():raise FileExistsError(out)
    if k<0:raise ValueError('k 必须非负；0 表示关闭')
    out.parent.mkdir(parents=True,exist_ok=True)
    identity=json.dumps({'input_sha256':sha256(input_path),'gt_sha256':sha256(gt_path),
                         'k':k,'img_dir':str(Path(img_dir).resolve()),'version':'resplit-background-v2'},sort_keys=True)
    if pending.exists() and not resume:raise FileExistsError('已有未完成输出，请检查后使用 --resume')
    started=time.monotonic()
    with h5py.File(input_path,'r') as src,h5py.File(gt_path,'r') as gt,h5py.File(pending,'a' if resume else 'x') as dst:
        required=['spacing_zyx','h_frac','min_volume','gaussian_sigma']
        if any(key not in src.attrs for key in required):raise ValueError('输入缺少实例拆分标定参数')
        if float(src.attrs.get('resplit_k',0.))!=0:
            raise ValueError('输入已经再切过；关闭对照必须从未再切的同一输入生成')
        if 'resplit_identity' in dst.attrs and dst.attrs['resplit_identity']!=identity:
            raise ValueError('恢复输入或配置不一致')
        for key,value in src.attrs.items():dst.attrs[key]=value
        dst.attrs.update(resplit_identity=identity,resplit_k=float(k),
            resplit_version='background_zero_v2',foreground_source='existing_prediction_instances',
            label_mapping='ctc_full_marker_majority_v2',detection_source='nnunet_pred')
        groups=dst.require_group('frames')
        for name in src:
            if name not in {'frames','tracks'} and name not in dst:src.copy(src[name],dst,name=name)
        cfg=InstanceSplitConfig(spacing_zyx=tuple(src.attrs['spacing_zyx']),h_frac=float(src.attrs['h_frac']),
             min_volume=int(src.attrs['min_volume']),min_distance=int(src.attrs.get('min_distance',3)),
             gaussian_sigma=float(src.attrs['gaussian_sigma']),use_watershed=bool(src.attrs.get('watershed',True)))
        rows=[]
        for index,key in enumerate(sorted(src['frames'])):
            if key in groups:
                if not groups[key].attrs.get('complete',False):raise ValueError(f'帧 {key} 未完整落盘，请另建实验编号')
                rows.append(json.loads(groups[key].attrs['summary']));continue
            old=np.asarray(src[f'frames/{key}/labels'])
            labels=refine_oversized_instances(old>0,old,cfg,k=k) if k>0 else old
            image=tifffile.imread(Path(img_dir)/f't{int(key):03d}.tif')
            if image.shape!=labels.shape:raise ValueError('原图与实例形状不一致')
            table=object_table_from_labels(labels,image)
            primary,all_ids=match_gt_labels(labels,np.asarray(gt[f'frames/{key}/labels']),table['label'])
            table.update(gt_label=primary,gt_ids=all_ids)
            group=groups.create_group(key)
            group.create_dataset('labels',data=labels.astype('int32'),compression='gzip',compression_opts=1)
            for name,value in table.items():group.create_dataset(name,data=value)
            row={'frame':int(key),'before':len(src[f'frames/{key}/label']),'after':len(primary),
                 'foreground_removed':int(np.sum((old>0)&(labels==0))),
                 'multi_identity':int(np.sum((all_ids>0).sum(1)>1))}
            group.attrs['summary']=json.dumps(row);group.attrs['complete']=True;dst.flush();rows.append(row)
            if index%10==0:print(json.dumps({**row,'done':index+1,'total':len(src['frames']),'elapsed_s':round(time.monotonic()-started,1)}),flush=True)
            del old,labels,image,table,primary,all_ids
    pending.replace(out)
    report={'input':str(input_path),'output':str(out),'k':k,'identity':json.loads(identity),'frames':rows,
            'output_sha256':sha256(out),'elapsed_s':round(time.monotonic()-started,2)}
    out.with_suffix('.resplit.json').write_text(json.dumps(report,indent=2))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True);p.add_argument('--gt-h5',required=True)
    p.add_argument('--img-dir',required=True);p.add_argument('--out',required=True)
    p.add_argument('--k',type=float,required=True);p.add_argument('--resume',action='store_true')
    a=p.parse_args();resplit(a.input,a.gt_h5,a.img_dir,a.out,a.k,a.resume)

if __name__=='__main__':main()
