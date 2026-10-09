"""nnU-Net预测前景的GT种子实例Oracle：冻结encoder、独立GNN训练与双序列官方评测。

追踪方法仍为 ideas.pdf 式(25)–(35)，不改变网络或损失；质量取预测实例体积。
逐帧核验前景和GT身份。任务可恢复，原始检测、模型与历史实验均保留。
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time


def digest(path):
    value = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix('.pending.json')
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    pending.replace(path)


def prepare_oracle(root, original, predictions, raw_images, gt_root, model_sha):
    import h5py
    for seq,expected in [('01',195),('02',190)]:
        truth=root/f'data/interim/Fluo-N3DH-CE_{seq}.h5'
        truth.parent.mkdir(parents=True,exist_ok=True)
        if not truth.exists():truth.symlink_to(original/truth.name)
        output=root/f'data/interim/Fluo-N3DH-CE_{seq}_nnunet_oracle.h5'
        if not output.exists():
            partial=output.with_name(output.stem+'.partial.h5')
            command=[sys.executable,str(root/'scripts/predict_to_h5.py'),'--pred-dir',str(predictions),
                '--img-root',str(raw_images),'--seq',seq,'--out',str(partial),'--gt-root',str(gt_root),
                '--oracle-markers','--unseeded-policy','keep_component','--min-volume','1',
                '--h-frac','.1','--gaussian-sigma','0','--spacing-zyx','1,.09,.09',
                '--report',str(root/f'logs/oracle_frontend_{seq}.json')]
            atomic_json(root/f'logs/oracle_frontend_{seq}_command.json',{'command':command})
            with (root/f'logs/oracle_frontend_{seq}.log').open('a') as stream:
                subprocess.run(command,cwd=root,stdout=stream,stderr=subprocess.STDOUT,check=True)
            with h5py.File(partial,'r+') as h5:
                if len(h5['frames'])!=expected:raise RuntimeError('Oracle检测帧数不完整')
                h5.attrs['foreground_source']='nnunet_semantic_prediction'
                h5.attrs['frozen_model_sha256']=model_sha
            partial.replace(output)
        with h5py.File(output,'r') as h5:
            if (h5.attrs['frozen_model_sha256']!=model_sha or
                h5.attrs['oracle_seed_version']!='preserve_label_ids_v2' or
                h5.attrs['unseeded_policy']!='keep_component' or h5.attrs['min_volume']!=1):
                raise RuntimeError('已存在Oracle实例输入来源不匹配')
        print(json.dumps({'Oracle_instances_prepared':seq,'bytes':output.stat().st_size}),flush=True)


def audit_oracle(root,predictions,gt_root):
    import h5py
    import numpy as np
    import SimpleITK as sitk
    import tifffile
    from scipy import ndimage
    out=root/'logs/oracle_mask_input_audit.json'
    if out.exists():
        saved=json.loads(out.read_text())
        for seq,row in saved['sequences'].items():
            if digest(root/f'data/interim/Fluo-N3DH-CE_{seq}_nnunet_oracle.h5')!=row['source_sha256']:
                raise RuntimeError('Oracle实例输入在恢复期间改变')
        return saved
    result={'protocol':'strict preserve nnU-Net foreground; GT seeds affect instance ownership only','sequences':{}}
    for seq,expected in [('01',195),('02',190)]:
        path=root/f'data/interim/Fluo-N3DH-CE_{seq}_nnunet_oracle.h5'
        rows,distances,volumes,manifest=[],[],[],{}
        totals={key:0 for key in ('gt_nodes','nodes','matched_gt_nodes','unmatched_nodes','multi_identity_nodes',
            'legacy_merged_seed_identity_excess','legacy_fragmented_seed_identity_excess','visible_seed_ids')}
        with h5py.File(path,'r') as h5,h5py.File(root/f'data/interim/Fluo-N3DH-CE_{seq}.h5','r') as truth:
            parent={int(row['label']):int(row['parent']) for row in truth['tracks'][:]}
            keys=sorted(h5['frames'])
            if [int(key) for key in keys]!=list(range(expected)):raise RuntimeError('Oracle检测帧缺失')
            previous={}
            for key in keys:
                t=int(key);frame=h5['frames'][key]
                pred=predictions/f'CE{seq}_f{t:03d}.nii.gz'
                image=sitk.ReadImage(str(pred));binary=sitk.GetArrayFromImage(image)>0
                spacing=np.array(list(reversed(image.GetSpacing())))
                if not np.allclose(spacing,[1.,.09,.09]):raise RuntimeError('预测掩码物理间距不一致')
                labels=frame['labels'][:]
                if not np.array_equal(labels>0,binary):raise RuntimeError('Oracle实例化改变了nnU-Net预测前景')
                gt=tifffile.imread(gt_root/f'{seq}_GT/TRA/man_track{t:03d}.tif')
                old,count=ndimage.label((gt>0)&binary)
                present=gt[old>0].astype(np.int64);components=old[old>0].astype(np.int64)
                base=int(gt.max())+1
                pairs=np.unique(components*base+present)
                c,g=pairs//base,pairs%base
                merged=int(np.maximum(np.bincount(c)-1,0).sum())
                fragmented=int(np.maximum(np.bincount(g)-1,0).sum())
                identities=frame['gt_ids'][:];n_ids=(identities>0).sum(axis=1)
                visible=set(identities[identities>0].tolist())
                target=set(truth['frames'][key]['label'][:].tolist())-{0}
                row={'frame':t,'gt_nodes':len(target),'nodes':len(identities),
                    'matched_gt_nodes':len(visible & target),'unmatched_nodes':int((n_ids==0).sum()),
                    'multi_identity_nodes':int((n_ids>1).sum()),'foreground_exact':True,
                    'foreground_voxels':int(binary.sum()),'visible_seed_ids':int(np.unique(present).size),
                    'legacy_merged_seed_identity_excess':merged,'legacy_fragmented_seed_identity_excess':fragmented}
                if row['multi_identity_nodes']:raise RuntimeError('身份保留Oracle仍有多GT归属，先核验输入')
                for name in totals:totals[name]+=row[name]
                volumes.extend(frame['volume'][:].tolist())
                current={int(label):point for ids,point in zip(identities,frame['centroid'][:]) for label in ids if label>0}
                for label,point in current.items():
                    predecessor=label if label in previous else parent.get(label,0)
                    if predecessor in previous:distances.append(float(np.linalg.norm((point-previous[predecessor])*spacing)))
                previous=current;rows.append(row);manifest[pred.name]=digest(pred)
                del binary,labels,gt,old,image
        spacing=np.array([1.,.09,.09])
        totals['missing_gt_nodes']=totals['gt_nodes']-totals['matched_gt_nodes']
        totals['GT_identity_coverage']=totals['matched_gt_nodes']/totals['gt_nodes']
        row={'n_frames':expected,'totals':totals,'frames':rows,'source_sha256':digest(path),
             'prediction_source':str(predictions),'prediction_files_sha256':manifest,
             'spacing_zyx_um':spacing.tolist(),'true_pairs':len(distances),
             'true_displacement_um':dict(zip(('p50','p95','p99','p99.9','max'),np.percentile(distances,[50,95,99,99.9,100]).tolist())),
             'instance_volume_um3':dict(zip(('min','p50','p95','max'),(np.percentile(volumes,[0,50,95,100])*spacing.prod()).tolist()))}
        result['sequences'][seq]=row
        print(json.dumps({'mask_and_identity_audited':seq,'totals':totals,'displacement_um':row['true_displacement_um']}),flush=True)
    atomic_json(out,result)
    return result


def export_encoder(root, seq, images, model):
    import h5py
    import numpy as np
    output = root / f'data/interim/encoder_{seq}_nnunet_oracle_20261009.npz'
    h5 = root / f'data/interim/Fluo-N3DH-CE_{seq}_nnunet_oracle.h5'
    identity = {'model_sha256': digest(model / 'fold_0/checkpoint_best.pth'),
                'detection_sha256': digest(h5), 'stage': 2, 'seq': seq}
    logs = root / 'logs'
    if not output.exists():
        before = subprocess.check_output(['nvidia-smi'], text=True)
        (logs / f'oracle_encoder_{seq}_gpu_before.txt').write_text(before)
        command = [sys.executable, str(root / 'scripts/nnunet/export_encoder_features.py'),
            '--h5', str(h5), '--images', str(images), '--model', str(model),
            '--seq', seq, '--out', str(output), '--stage', '2', '--device', 'cuda']
        with (logs / f'oracle_encoder_{seq}.log').open('a') as stream:
            process = subprocess.Popen(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT)
            atomic_json(logs / f'oracle_encoder_{seq}_process.json', {'pid': process.pid, 'command': command})
            observed = False
            while process.poll() is None:
                probe = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                                                 '--format=csv,noheader'], text=True)
                if any(line.strip().startswith(f'{process.pid},') for line in probe.splitlines()):
                    observed = True
                    (logs / f'oracle_encoder_{seq}_gpu_active.txt').write_text(probe)
                time.sleep(1)
        (logs / f'oracle_encoder_{seq}_gpu_after.txt').write_text(subprocess.check_output(['nvidia-smi'], text=True))
        if process.returncode or not observed:
            raise RuntimeError(f'encoder {seq} 失败或没有观察到 GPU 占用')
    metadata = json.loads(output.with_suffix('.json').read_text())
    if any(metadata.get(key) != value for key, value in identity.items()):
        raise RuntimeError('encoder 侧车来源不匹配')
    with h5py.File(h5) as detections, np.load(output) as features:
        for key in detections['frames']:
            t = int(key)
            if not np.array_equal(features[f'label_{t}'], detections['frames'][key]['label'][:]):
                raise RuntimeError('encoder 实例身份错位')
            if features[f'frame_{t}'].shape != (len(detections['frames'][key]['label']), 128):
                raise RuntimeError('encoder 特征维度或帧覆盖错误')
            if not np.isfinite(features[f'frame_{t}']).all():
                raise RuntimeError('encoder 特征非有限')
    print(json.dumps({'encoder_complete': seq, 'bytes': output.stat().st_size}), flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--original-interim', required=True)
    parser.add_argument('--predictions',default='/root/autodl-tmp/nnunet/preds_eval')
    parser.add_argument('--raw-images',default='/root/autodl-tmp/ctc/Fluo-N3DH-CE')
    parser.add_argument('--gt-root', default='/root/autodl-tmp/ctc/raw/Fluo-N3DH-CE')
    parser.add_argument('--images', default='/root/autodl-tmp/nnunet/nnUNet_raw/Dataset501_CellTrackerCE/imagesTs_eval')
    parser.add_argument('--model', default='/root/autodl-tmp/nnunet/nnUNet_results/Dataset501_CellTrackerCE/nnUNetTrainer__nnUNetPlans__3d_fullres')
    args = parser.parse_args()
    root = Path(args.root).resolve()
    sys.path.insert(0, str(root / 'paperpipe/src'))
    from papertrack.config import load_config
    spec = importlib.util.spec_from_file_location('matrix', root / 'paperpipe/scripts/run_ablation_matrix.py')
    matrix = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(matrix)
    (root / 'logs').mkdir(exist_ok=True)
    model = Path(args.model)
    frozen_before = digest(model / 'fold_0/checkpoint_best.pth')
    prepare_oracle(root,Path(args.original_interim),Path(args.predictions),Path(args.raw_images),
                   Path(args.gt_root),frozen_before)
    audit=audit_oracle(root,Path(args.predictions),Path(args.gt_root))
    for seq in ('01', '02'):
        export_encoder(root, seq, Path(args.images), model)
    from papertrack.config import save_config
    base=load_config(root / 'experiments/E1.0_nnunet_oracle_encoder_20261009/pipeline_input.yaml')
    # 只按seq01真实Oracle质心位移标定物理门限，向上取0.1µm以覆盖边界舍入。
    import math
    maximum=audit['sequences']['01']['true_displacement_um']['max']
    base.coupling.r_max=max(6.4,math.ceil(maximum*10)/10)
    configuration=root/'logs/pipeline_preflight.yaml'
    if not configuration.exists(): save_config(base,configuration)
    output = root / 'experiments/P9_nnunet_oracle_encoder_matrix_20261009'
    if (output / 'plan.json').exists():
        plan = json.loads((output / 'plan.json').read_text())
    else:
        arguments = SimpleNamespace(config=str(configuration), out=str(output), ablations=[],
            encoder01='data/interim/encoder_01_nnunet_oracle_20261009.npz',
            encoder02='data/interim/encoder_02_nnunet_oracle_20261009.npz',
            h5_01='data/interim/Fluo-N3DH-CE_01_nnunet_oracle.h5', h5_02='data/interim/Fluo-N3DH-CE_02_nnunet_oracle.h5',
            gt_01='data/interim/Fluo-N3DH-CE_01.h5', gt_02='data/interim/Fluo-N3DH-CE_02.h5',
            exp_prefix='P9_nnunet_oracle_encoder', device='cuda', frames=None,
            seeds=[20261008, 20261009], epochs=60, official=True, dataset='Fluo-N3DH-CE',
            official_gt_root='/root/autodl-tmp/ctc/raw', official_tools='/root/EvaluationSoftware/Linux',
            recalibrate=True, skip_baseline=False)
        plan = matrix.make_plan(arguments)
        source = next(job for job in plan['jobs'] if job['id'].endswith('seed20261008_eval02'))
        repeat = json.loads(json.dumps(source))
        previous = repeat['id']
        repeat['id'] += '_repeat'
        repeat['command'][repeat['command'].index('--exp-id') + 1] = repeat['id']
        repeat['outputs'] = [value.replace(previous, repeat['id']) for value in repeat['outputs']]
        plan['jobs'].append(repeat)
        plan['noise_protocol'] = 'same seed20261008 checkpoint, complete seq02 inference/export/official rerun'
        plan['Oracle_definition'] = 'nnU-Net prediction foreground preserved exactly; GT label identity seeds; frozen encoder at Oracle-instance centroids'
        atomic_json(output / 'plan.json', plan)
        (output / 'source_release.json').write_text((root / 'SOURCE_VERSION.json').read_text())
    matrix.execute_plan({**plan, 'jobs': plan['jobs'][:2]}, output)
    report = json.loads((output / 'calibration/baseline.json').read_text())
    config = load_config(output / 'configs/baseline_calibrated.yaml')
    penalty = config.gnn.lambda_ot * report['distributions']['true_cost']['p99.9']
    print(json.dumps({'lambda_times_true_cost_p999': penalty,
                      'candidate_recall': report['after_calibration']['candidate_true_recall'],
                      'division_recall': report['after_calibration']['division_pair_recall']}), flush=True)
    if (penalty > 1 or report['after_calibration']['candidate_true_recall'] < .99
            or report['after_calibration']['division_pair_recall'] < .99):
        raise RuntimeError('训练前尺度/候选审计未通过，检查标定报告后再恢复')
    # 部署阈值会改变图里的边；训练判据使用实际训练图，不用标定前的图替代。
    import numpy as np
    from papertrack.graph.build import EDGE_COST
    costs, positive, division, bridge, all_positive, bridge_positive = [], 0, 0, 0, 0, 0
    graphs = sorted((output / 'graphs/baseline').glob('pair_*.npz'))
    for path in graphs:
        with np.load(path) as graph:
            true = graph['label'] > 0
            adjacent = graph['cand_gap'] == 1
            positive += int((true & adjacent).sum())
            all_positive += int(true.sum())
            bridge_positive += int((true & ~adjacent).sum())
            division += int(graph['cand_label_div'][adjacent].sum())
            bridge += int((graph['cand_gap'] > 1).sum())
            costs.append(graph['cand_feat'][true, EDGE_COST])
    actual_cost = np.concatenate(costs)
    if len(graphs) != 194 or len(actual_cost) == 0:
        raise RuntimeError('实际训练图不完整或没有真实关联边')
    graph_audit = {'graphs': len(graphs), 'positive_edges': positive,
        'division_positive': division, 'bridge_candidates': bridge,
        'all_positive_edges': all_positive, 'bridge_positive_edges': bridge_positive,
        'true_edges_total': report['after_calibration']['true_detection_pairs'],
        'division_edges_total': report['after_calibration']['division_pairs'],
        'true_recall': positive / report['after_calibration']['true_detection_pairs'],
        'division_recall': division / report['after_calibration']['division_pairs'],
        'positive_C_percentiles': dict(zip(('p50', 'p95', 'p99', 'p99.9', 'max'),
            np.percentile(actual_cost, [50, 95, 99, 99.9, 100]).tolist())),
        'lambda_OT': config.gnn.lambda_ot,
        'input_source': config.detection_source,
        'features': config.node.f_source,
        'GT_role': 'seed-label identities and edge labels; node vectors have no track identity column'}
    atomic_json(output / 'training_graph_audit.json', graph_audit)
    # ideas.pdf式(26)及R5/R14：不可逆筛选须分别审计移动/分裂分母，
    # 避免数量多的移动边掩盖分裂事件损失；沿用已有top-k工程保底。
    if graph_audit['true_recall'] < .99 or graph_audit['division_recall'] < .99:
        raise RuntimeError('部署训练图真实关联或分裂边覆盖不足99%，禁止开始训练')
    if config.gnn.lambda_ot * graph_audit['positive_C_percentiles']['p99.9'] > 1:
        raise RuntimeError('实际训练图的真实边 OT 损失尺度过大')
    matrix.execute_plan(plan, output)
    frozen_after = digest(model / 'fold_0/checkpoint_best.pth')
    if frozen_after != frozen_before:
        raise RuntimeError('nnU-Net 冻结模型在任务期间发生变化')
    atomic_json(root / 'logs/oracle_frozen_model_identity.json',
                {'before': frozen_before, 'after': frozen_after, 'unchanged': True})
    (root / 'logs/oracle_gpu_idle.txt').write_text(subprocess.check_output(['nvidia-smi'], text=True))
    (root / 'logs/oracle_encoder_pipeline.complete').write_text('complete\n')
    print('nnU-Net mask + GT seed Oracle + encoder: complete', flush=True)


if __name__ == '__main__':
    main()
