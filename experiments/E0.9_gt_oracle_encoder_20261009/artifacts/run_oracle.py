"""GT marker 上界实验：审计输入、冻结 encoder 前向、独立训练与双序列官方评测。

方法仍为 ideas.pdf 式(25)–(35)，不改变网络或损失；标记体积适配沿用决策0001。
逐帧验证 GT 输入，全部任务可恢复。原始检测、模型与历史实验均保留。
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


def audit_inputs(root, original, gt_root):
    import h5py
    import numpy as np
    import tifffile
    output = root / 'logs/oracle_input_audit.json'
    if output.exists():
        saved = json.loads(output.read_text())
        for seq, row in saved['sequences'].items():
            if digest(original / f'Fluo-N3DH-CE_{seq}.h5') != row['source_sha256']:
                raise RuntimeError('GT 原始输入在恢复期间发生变化')
        return saved
    result = {'sequences': {}, 'protocol': 'GT TRA marker pixels, no predicted segmentation input'}
    spacing = np.array([1., .09, .09])
    for seq, expected in [('01', 195), ('02', 190)]:
        source = original / f'Fluo-N3DH-CE_{seq}.h5'
        destination = root / 'data/interim' / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            destination.symlink_to(source)
        if destination.resolve() != source.resolve():
            raise RuntimeError('检测入口没有指向声明的 GT 原件')
        distances, volumes, frames = [], [], []
        with h5py.File(source, 'r') as handle:
            keys = sorted(handle['frames'])
            if [int(key) for key in keys] != list(range(expected)):
                raise RuntimeError('GT 检测帧不完整')
            tracks = handle['tracks'][:]
            parent = {int(row['label']): int(row['parent']) for row in tracks}
            previous = None
            for key in keys:
                frame = handle['frames'][key]
                marker = tifffile.imread(gt_root / f'{seq}_GT/TRA/man_track{int(key):03d}.tif')
                labels = frame['labels'][:]
                if not np.array_equal(marker, labels):
                    raise RuntimeError(f'{seq}/{key}: H5 与官方 GT marker 像素不同')
                identifiers = frame['label'][:]
                if not np.array_equal(np.unique(labels[labels > 0]), identifiers):
                    raise RuntimeError('GT 实例表与体标签身份不一致')
                current = {int(label): (point, size) for label, point, size in
                           zip(identifiers, frame['centroid'][:], frame['volume'][:])}
                if not set(current).issubset(parent):
                    raise RuntimeError('GT marker 身份没有对应谱系')
                if previous is not None:
                    for label, (point, size) in current.items():
                        predecessor = label if label in previous else parent.get(label, 0)
                        if predecessor in previous:
                            distances.append(float(np.linalg.norm((point - previous[predecessor][0]) * spacing)))
                volumes.extend(float(value[1]) for value in current.values())
                frames.append({'frame': int(key), 'nodes': len(current), 'pixels_equal_official_GT': True})
                previous = current
                del labels, marker
        result['sequences'][seq] = {'frames': frames, 'n_frames': expected,
            'nodes': sum(row['nodes'] for row in frames), 'tracks': len(tracks),
            'source': str(source), 'source_sha256': digest(source),
            'spacing_zyx_um': spacing.tolist(), 'true_pairs': len(distances),
            'displacement_um_percentiles': dict(zip(('p50', 'p95', 'p99', 'p99.9', 'max'),
                np.percentile(distances, [50, 95, 99, 99.9, 100]).tolist())),
            'marker_volume_vox_percentiles': np.percentile(volumes, [0, 50, 100]).tolist(),
            'marker_volume_um3_percentiles': (np.percentile(volumes, [0, 50, 100]) * spacing.prod()).tolist()}
        print(json.dumps({'input_audited': seq, 'frames': expected, 'nodes': result['sequences'][seq]['nodes']}), flush=True)
    atomic_json(output, result)
    return result


def export_encoder(root, seq, images, model):
    import h5py
    import numpy as np
    output = root / f'data/interim/encoder_{seq}_gt_marker_20261009.npz'
    h5 = root / f'data/interim/Fluo-N3DH-CE_{seq}.h5'
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
    audit_inputs(root, Path(args.original_interim), Path(args.gt_root))
    model = Path(args.model)
    frozen_before = digest(model / 'fold_0/checkpoint_best.pth')
    for seq in ('01', '02'):
        export_encoder(root, seq, Path(args.images), model)
    configuration = root / 'experiments/E0.9_gt_oracle_encoder_20261009/pipeline_input.yaml'
    output = root / 'experiments/P8_gt_marker_encoder_matrix_20261009'
    if (output / 'plan.json').exists():
        plan = json.loads((output / 'plan.json').read_text())
    else:
        arguments = SimpleNamespace(config=str(configuration), out=str(output), ablations=[],
            encoder01='data/interim/encoder_01_gt_marker_20261009.npz',
            encoder02='data/interim/encoder_02_gt_marker_20261009.npz',
            h5_01='data/interim/Fluo-N3DH-CE_01.h5', h5_02='data/interim/Fluo-N3DH-CE_02.h5',
            gt_01='data/interim/Fluo-N3DH-CE_01.h5', gt_02='data/interim/Fluo-N3DH-CE_02.h5',
            exp_prefix='P8_gt_marker_encoder', device='cuda', frames=None,
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
        plan['Oracle_definition'] = 'GT TRA marker positions and masks; frozen encoder sampled from original images'
        atomic_json(output / 'plan.json', plan)
        (output / 'source_release.json').write_text((root / 'SOURCE_VERSION.json').read_text())
    matrix.execute_plan({**plan, 'jobs': plan['jobs'][:2]}, output)
    report = json.loads((output / 'calibration/baseline.json').read_text())
    config = load_config(output / 'configs/baseline_calibrated.yaml')
    penalty = config.gnn.lambda_ot * report['distributions']['true_cost']['p99.9']
    print(json.dumps({'lambda_times_true_cost_p999': penalty,
                      'candidate_recall': report['after_calibration']['candidate_true_recall']}), flush=True)
    if penalty > 1 or report['after_calibration']['candidate_true_recall'] < .99:
        raise RuntimeError('训练前尺度/候选审计未通过，检查标定报告后再恢复')
    # 部署阈值会改变图里的边；训练判据使用实际训练图，不用标定前的图替代。
    import numpy as np
    from papertrack.graph.build import EDGE_COST
    costs, positive, division, bridge = [], 0, 0, 0
    graphs = sorted((output / 'graphs/baseline').glob('pair_*.npz'))
    for path in graphs:
        with np.load(path) as graph:
            true = graph['label'] > 0
            positive += int(true.sum())
            division += int(graph['cand_label_div'].sum())
            bridge += int((graph['cand_gap'] > 1).sum())
            costs.append(graph['cand_feat'][true, EDGE_COST])
    actual_cost = np.concatenate(costs)
    if len(graphs) != 194 or len(actual_cost) == 0:
        raise RuntimeError('实际训练图不完整或没有真实关联边')
    graph_audit = {'graphs': len(graphs), 'positive_edges': positive,
        'division_positive': division, 'bridge_candidates': bridge,
        'positive_C_percentiles': dict(zip(('p50', 'p95', 'p99', 'p99.9', 'max'),
            np.percentile(actual_cost, [50, 95, 99, 99.9, 100]).tolist())),
        'lambda_OT': config.gnn.lambda_ot,
        'input_source': config.detection_source,
        'features': config.node.f_source,
        'train_GT_role': 'edge labels; node vectors have no track identity column'}
    atomic_json(output / 'training_graph_audit.json', graph_audit)
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
    print('GT marker Oracle + encoder: complete', flush=True)


if __name__ == '__main__':
    main()
