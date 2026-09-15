#!/usr/bin/env python3
"""在云服务器上用 CTC 官方二进制评测结果目录，并把指标与日志回传。

用法::

    python3 scripts/cloud_eval.py \
        --res-dir experiments/E1.1_baseline/artifacts/submission/01_RES \
        --dataset Fluo-N3DH-CE --seq 01 \
        --out experiments/E1.1_baseline/metrics_official_01.json

云端布局（自动创建）::

    /root/autodl-tmp/celltracker/eval/<tag>/<dataset>/<seq>_GT  -> 软链到数据集 GT
    /root/autodl-tmp/celltracker/eval/<tag>/<dataset>/<seq>_RES -> 上传的结果
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cloud_run import parse_cred  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CLOUD_CTC = "/root/autodl-tmp/ctc/raw"
CLOUD_TOOLS = "/root/EvaluationSoftware/Linux"
CLOUD_EVAL = "/root/autodl-tmp/celltracker/eval"


def pexpect_cmd(argv: list[str], password: str, timeout: int = 1800,
                stdin_data: str | None = None) -> tuple[int, str]:
    import pexpect

    child = pexpect.spawn(argv[0], argv[1:], timeout=timeout, encoding="utf-8",
                          codec_errors="replace")
    sent = False
    try:
        try:
            child.setecho(False)
        except Exception:  # noqa: BLE001
            pass
        while True:
            i = child.expect([r"[Pp]assword:\s*$", pexpect.EOF], timeout=timeout)
            if i == 0:
                if sent:
                    child.close(force=True)
                    return 1, child.before or ""
                child.sendline(password)
                sent = True
                if stdin_data is not None:
                    child.send(stdin_data)
                    child.sendeof()
            else:
                break
        return child.exitstatus or 0, child.before or ""
    finally:
        child.close()


def ssh(argv: list[str], password: str, timeout: int = 1800,
        stdin_data: str | None = None) -> tuple[int, str]:
    host, port, user, _ = parse_cred()
    cmd = ["ssh", "-p", port, "-o", "StrictHostKeyChecking=accept-new",
           "-o", "ServerAliveInterval=30", f"{user}@{host}", *argv]
    return pexpect_cmd(cmd, password, timeout, stdin_data=stdin_data)


def scp(local: str, remote: str, password: str, timeout: int = 3600) -> tuple[int, str]:
    """上传：本地 -> 云端。"""
    host, port, user, _ = parse_cred()
    cmd = ["scp", "-r", "-P", port, "-o", "StrictHostKeyChecking=accept-new",
           str(local), f"{user}@{host}:{remote}"]
    return pexpect_cmd(cmd, password, timeout)


def scp_download(remote: str, local: str, password: str, timeout: int = 300) -> tuple[int, str]:
    """下载：云端 -> 本地。"""
    host, port, user, _ = parse_cred()
    Path(local).parent.mkdir(parents=True, exist_ok=True)
    cmd = ["scp", "-P", port, "-o", "StrictHostKeyChecking=accept-new",
           f"{user}@{host}:{remote}", str(local)]
    return pexpect_cmd(cmd, password, timeout)


def main() -> None:
    ap = argparse.ArgumentParser(description="云端官方 CTC 指标评测")
    ap.add_argument("--res-dir", required=True, type=Path)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--num-digits", type=int, default=3)
    ap.add_argument("--out", type=Path, default=None, help="metrics json 输出路径")
    ap.add_argument("--cloud-gt-root", default=CLOUD_CTC,
                    help="云端 GT 所在根目录（默认 CTC 原始数据根）")
    ap.add_argument("--keep-remote", action="store_true")
    args = ap.parse_args()

    if not args.res_dir.is_dir():
        sys.exit(f"结果目录不存在: {args.res_dir}")

    host, port, user, password = parse_cred()
    tag = f"{args.dataset}_{args.seq}_{time.strftime('%Y%m%d-%H%M%S')}"
    remote_dir = f"{CLOUD_EVAL}/{tag}"
    remote_dataset = f"{remote_dir}/{args.dataset}"
    dry = {"host": f"{user}@{host}:{port}", "tag": tag, "remote_dir": remote_dir,
           "dataset": args.dataset, "seq": args.seq, "res_dir": str(args.res_dir)}
    print(f"[cloud] {json.dumps(dry, ensure_ascii=False)}", flush=True)

    # 1) 打包上传结果
    rc, out = ssh(["mkdir", "-p", remote_dir], password)
    if rc != 0:
        sys.exit(f"创建云端目录失败: {out}")
    with tempfile.TemporaryDirectory() as td:
        tar_path = Path(td) / "res.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tf:
            tf.add(args.res_dir, arcname=f"{args.seq}_RES")
        print(f"[1/3] 上传 {tar_path.stat().st_size / 1048576:.1f} MB ...", flush=True)
        rc, out = scp(str(tar_path), f"{remote_dir}/res.tar.gz", password)
        if rc != 0:
            sys.exit(f"上传失败: {out}")

    # 2) 云端组装目录并运行三个官方指标
    script = f"""
set -e
cd {remote_dir}
mkdir -p {remote_dataset}
ln -sfn {args.cloud_gt_root}/{args.dataset}/{args.seq}_GT {remote_dataset}/{args.seq}_GT
rm -rf {remote_dataset}/{args.seq}_RES
tar xzf res.tar.gz -C {remote_dataset}
set +e
for exe in SEGMeasure DETMeasure TRAMeasure; do
  echo "=== $exe ==="
  {CLOUD_TOOLS}/$exe {remote_dataset} {args.seq} {args.num_digits}
done
echo "=== logs ==="
ls {remote_dataset}/{args.seq}_RES/*.txt 2>/dev/null || true
"""
    print("[2/3] 运行官方评测 ...", flush=True)
    rc, out = ssh(["bash", "-s"], password, stdin_data=script)
    print(out)
    if rc != 0:
        sys.exit(f"评测失败 (exit {rc})")

    def grab(name: str) -> float | None:
        m = re.search(rf"{name} measure:\s*([0-9.]+)", out)
        return float(m.group(1)) if m else None

    metrics = {
        "dataset": args.dataset,
        "seq": args.seq,
        "num_digits": args.num_digits,
        "res_dir": str(args.res_dir),
        "evaluator": "CTC official EvaluationSoftware (cloud)",
        "host": f"{user}@{host}:{port}",
        "SEG": grab("SEG"),
        "DET": grab("DET"),
        "TRA": grab("TRA"),
        "raw_output": out.strip().splitlines(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out_path = args.out or (args.res_dir.parent / f"metrics_official_{args.seq}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))

    # 回传官方日志（评测证据）
    log_dir = out_path.parent / "official_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    for name in ("SEG_log.txt", "DET_log.txt", "TRA_log.txt"):
        rc_dl, _ = scp_download(f"{remote_dataset}/{args.seq}_RES/{name}",
                                str(log_dir / f"{name}"), password)
        if rc_dl == 0:
            print(f"      日志 -> {log_dir / name}")

    print(f"[3/3] 指标: SEG={metrics['SEG']} DET={metrics['DET']} TRA={metrics['TRA']}")
    print(f"      -> {out_path}")


if __name__ == "__main__":
    main()
