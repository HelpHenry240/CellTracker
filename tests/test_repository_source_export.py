"""验证 Git 导出的源码包含数据接口，防止本机完整、克隆后缺模块。"""
from pathlib import Path
import io
import subprocess
import tarfile

import pytest


def test_git_export_contains_data_packages():
    root = Path(__file__).resolve().parents[1]
    probe = subprocess.run(['git', '-C', str(root), 'rev-parse', '--show-toplevel'],
                           capture_output=True, text=True)
    if probe.returncode or Path(probe.stdout.strip()).resolve() != root:
        pytest.skip('源码归档没有 Git 索引；包导入由接口测试验证')
    expected = ('src/celltracker/data/__init__.py', 'src/celltracker/data/ctc.py',
                'src/celltracker/data/build_dataset.py', 'src/celltracker/data/summarize.py',
                'paperpipe/src/vendor/celltracker/data/__init__.py',
                'paperpipe/src/vendor/celltracker/data/ctc.py')
    # 暂存树与提交时的实际内容相同；测试也能在提交前验证新增源码。
    tree = subprocess.check_output(['git', '-C', str(root), 'write-tree'], text=True).strip()
    exported = subprocess.check_output(['git', '-C', str(root), 'archive', tree,
                                       'src/celltracker', 'paperpipe/src/vendor/celltracker'])
    with tarfile.open(fileobj=io.BytesIO(exported)) as archive:
        names = set(archive.getnames())
    assert set(expected).issubset(names), 'Git 导出遗漏数据接口源码'
    for name in expected:
        ignored = subprocess.run(['git', '-C', str(root), 'check-ignore', '--no-index', name],
                                 capture_output=True)
        assert ignored.returncode == 1, f'源码仍受忽略规则屏蔽：{name}'
    raw_data = subprocess.run(['git', '-C', str(root), 'check-ignore', 'data/example.txt'],
                              capture_output=True)
    assert raw_data.returncode == 0, '原始数据目录应继续忽略'
