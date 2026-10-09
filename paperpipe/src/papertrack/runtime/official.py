"""在官方二进制所在机器执行 CTC 评测，保存指标、日志与可执行文件指纹。"""
from __future__ import annotations
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from .contracts import file_hash
from .validate import validate_ctc_dir


def evaluate_official(res_dir,gt_dir,tools_dir,seq,out_path,num_digits=3):
    """只调用已有官方程序；格式错误、程序失败和指标缺失均视为失败。"""
    res_dir,gt_dir,tools_dir = map(lambda p:Path(p).resolve(),(res_dir,gt_dir,tools_dir))
    check = validate_ctc_dir(res_dir,num_digits)
    if not check['ok']:
        raise ValueError(f'CTC 格式不合法：{check["errors"]}')
    if not gt_dir.is_dir():
        raise FileNotFoundError(f'GT 目录不存在：{gt_dir}')
    binaries = {name:tools_dir/f'{name}Measure' for name in ['DET','SEG','TRA']}
    for path in binaries.values():
        if not path.is_file():
            raise FileNotFoundError(f'官方二进制不存在：{path}')
    output = Path(out_path)
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():
        raise FileExistsError(f'官方结果已存在，不能覆盖：{output}')
    log_dir = output.parent/'official_logs'
    log_dir.mkdir(exist_ok=True)
    report = {'seq':seq,'evaluator':'CTC official EvaluationSoftware',
              'format_validation':check,'binary_sha256':{name:file_hash(p) for name,p in binaries.items()}}
    with tempfile.TemporaryDirectory(prefix='ctc_eval_') as temporary:
        root = Path(temporary)
        (root/f'{seq}_GT').symlink_to(gt_dir,target_is_directory=True)
        (root/f'{seq}_RES').symlink_to(res_dir,target_is_directory=True)
        for name,binary in binaries.items():
            command = [str(binary),str(root),seq,str(num_digits)]
            process = subprocess.run(command,capture_output=True,text=True)
            text = process.stdout+'\n'+process.stderr
            (log_dir/f'{name}_stdout.txt').write_text(text)
            match = re.search(rf'{name} measure:\s*([0-9.]+)',text)
            if process.returncode or match is None:
                raise RuntimeError(f'{name} 官方评测失败；详见 {log_dir/name}_stdout.txt')
            value = float(match.group(1))
            if not 0 <= value <= 1:
                raise RuntimeError(f'{name} 官方指标越界：{value}')
            report[name] = value
            log = res_dir/f'{name}_log.txt'
            if log.exists():
                shutil.copy2(log,log_dir/log.name)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    return report
