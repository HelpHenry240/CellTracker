#!/usr/bin/env python3
"""整合两套源码位置；算法实现保持原样，调整导入、入口和配置路径。"""
from pathlib import Path
import json
import shutil

ROOT = Path(__file__).resolve().parents[3]
STAGE = Path(__file__).resolve().parents[1]


def edit(path, replacements):
    file = ROOT / path
    content = file.read_text()
    for old, new in replacements:
        assert old in content, f'{path}: 找不到待替换文本 {old!r}'
        content = content.replace(old, new)
    file.write_text(content)


def main():
    catalog = json.loads((ROOT / 'experiments/archives/catalog.json').read_text())
    assert catalog[0]['archive'].endswith('00_source_and_documents.tar.gz'), '必须先完成源码归档'
    # 两份公共计算代码完全相同，三处包初始化差异在下方用兼容导出解决。
    excluded = {'detect/__init__.py', 'gnn/__init__.py', 'pipeline/__init__.py'}
    for file in (ROOT / 'paperpipe/src/vendor/celltracker').rglob('*.py'):
        rel = file.relative_to(ROOT / 'paperpipe/src/vendor/celltracker')
        if str(rel) not in excluded:
            assert file.read_bytes() == (ROOT / 'src/celltracker' / rel).read_bytes(), rel
    moves = {
        'paperpipe/src/papertrack': 'src/papertrack',
        'paperpipe/scripts/run_paper_pipeline.py': 'scripts/run_pipeline.py',
        'paperpipe/scripts/run_ablation_matrix.py': 'scripts/run_ablation_matrix.py',
        'paperpipe/scripts/train_paper_gnn.py': 'scripts/train_gnn.py',
        'paperpipe/scripts/calibrate_params.py': 'scripts/calibrate_params.py',
        'paperpipe/scripts/check_candidate_coverage.py': 'scripts/check_candidate_coverage.py',
        'paperpipe/scripts/cloud_nnunet_infer.sh': 'scripts/nnunet/cloud_infer.sh',
        'paperpipe/configs/paper_default.yaml': 'configs/paper_default.yaml',
        'paperpipe/configs/ce_calibrated_20261009.yaml': 'configs/ce_calibrated_20261009.yaml',
        'paperpipe/tests/test_papertrack.py': 'tests/test_papertrack.py',
        'paperpipe/tests/test_rebuild_regressions.py': 'tests/test_rebuild_regressions.py',
    }
    for old, new in moves.items():
        shutil.move(str(ROOT / old), str(ROOT / new))
    edits = [('ROOT = PKG.parent', 'ROOT = PKG'), ('ROOT=PKG.parent', 'ROOT=PKG'),
             ('root=PKG.parent', 'root=PKG'), ('run_paper_pipeline.py', 'run_pipeline.py'),
             ('train_paper_gnn.py', 'train_gnn.py'),
             ('ROOT / "paperpipe" / "configs"', 'ROOT / "configs"'),
             ('paperpipe/scripts/check_candidate_coverage.py', 'scripts/check_candidate_coverage.py'),
             ('  —— 触发 _paths（vendor 优先）', '')]
    for name in ['run_pipeline', 'run_ablation_matrix', 'train_gnn', 'calibrate_params', 'check_candidate_coverage']:
        file = ROOT / f'scripts/{name}.py'
        content = file.read_text()
        for old, new in edits:
            content = content.replace(old, new)
        file.write_text(content)
    edit('src/papertrack/gnn/train.py', [('"paperpipe/runs/gnn"', '"data/interim/gnn"')])
    edit('tests/test_rebuild_regressions.py', [("parents[2]/'scripts/nnunet/export_encoder_features.py'",
                                             "parents[1]/'scripts/nnunet/export_encoder_features.py'")])
    edit('pyproject.toml', [('pythonpath = ["src", "paperpipe/src"]', 'pythonpath = ["src"]'),
                          ('testpaths = ["tests", "paperpipe/tests"]', 'testpaths = ["tests"]')])
    obsolete = ['ablate_ot_prior.py', 'check_candidate_recall.py', 'eval_pipeline.py', 'make_comparison.py',
                'pipeline_funnel.py', 'plot_c1c2_evidence.py', 'plot_official_tiers.py', 'plot_resplit_sweep.py',
                'run_gnn.py', 'run_ot.py', 'run_phase12_suite.sh', 'run_sweep.py', 'sweep_gnn_thresholds.py',
                'run_ideas_pipeline.py']
    for name in obsolete:
        (ROOT / 'scripts' / name).unlink()
    (ROOT / 'src/papertrack/_paths.py').unlink()
    (ROOT / 'configs/pipeline_default.yaml').unlink()
    # 未迁出的旧校准、旧权重与 vendor 副本已在 00_source_and_documents 完整归档。
    shutil.rmtree(ROOT / 'paperpipe')
    record = {'moves': moves, 'obsolete_scripts_removed': obsolete,
              'duplicate_source_removed': 'paperpipe/src/vendor/celltracker',
              'restore_archive': 'experiments/archives/00_source_and_documents.tar.gz'}
    (STAGE / 'artifacts/source_migration.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
