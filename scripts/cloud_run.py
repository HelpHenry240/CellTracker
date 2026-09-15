#!/usr/bin/env python3
"""在云服务器上执行命令（每次交互式输入密码）。

凭据从本地 `本地文档/ssh&key` 读取，不写入受版本控制的文件。

用法:
    python3 scripts/cloud_run.py "nvidia-smi"
    python3 scripts/cloud_run.py --upload local.txt /root/remote.txt
    python3 scripts/cloud_run.py --download /root/remote.txt local.txt
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pexpect

ROOT = Path(__file__).resolve().parent.parent
CRED = ROOT / "本地文档" / "ssh&key"


def parse_cred() -> tuple[str, str, str, str]:
    text = CRED.read_text(encoding="utf-8").splitlines()
    m = re.search(r"ssh\s+-p\s+(\d+)\s+(\S+)@(\S+)", " ".join(text))
    if not m:
        sys.exit(f"无法从 {CRED} 解析 ssh 命令")
    port, user, host = m.group(1), m.group(2), m.group(3)
    password = ""
    for line in text:
        line = line.strip()
        if line and not line.startswith("ssh "):
            password = line
            break
    if not password:
        sys.exit(f"无法从 {CRED} 解析密码")
    return host, port, user, password


def run(argv: list[str], echo: bool = True) -> int:
    host, port, user, password = parse_cred()
    cmd = ["ssh", "-p", port, "-o", "StrictHostKeyChecking=accept-new",
           "-o", "ServerAliveInterval=30", f"{user}@{host}", *argv]
    if echo:
        print(f"$ ssh -p {port} {user}@{host} {' '.join(argv)}", flush=True)
    child = pexpect.spawn(cmd[0], cmd[1:], timeout=None, encoding="utf-8",
                          codec_errors="replace")
    password_sent = False
    try:
        while True:
            i = child.expect([r"[Pp]assword:\s*$", pexpect.EOF, pexpect.TIMEOUT], timeout=600)
            if i == 0:
                if password_sent:
                    print("密码错误或被再次要求输入", file=sys.stderr)
                    child.close(force=True)
                    return 1
                child.sendline(password)
                password_sent = True
            elif i == 1:
                break
            else:
                break
        sys.stdout.write(child.before or "")
        sys.stdout.flush()
    finally:
        child.close()
        rc = child.exitstatus if child.exitstatus is not None else 0
        if echo:
            print(f"\n[exit {rc}]")
        return rc


def main() -> None:
    ap = argparse.ArgumentParser(description="云端执行/传输（密码登录）")
    ap.add_argument("command", nargs="*", help="在云端执行的命令")
    ap.add_argument("--upload", nargs=2, metavar=("LOCAL", "REMOTE"))
    ap.add_argument("--download", nargs=2, metavar=("REMOTE", "LOCAL"))
    args = ap.parse_args()

    host, port, user, password = parse_cred()
    if args.upload:
        local, remote = args.upload
        cmd = ["scp", "-P", port, "-o", "StrictHostKeyChecking=accept-new",
               str(local), f"{user}@{host}:{remote}"]
    elif args.download:
        remote, local = args.download
        cmd = ["scp", "-P", port, "-o", "StrictHostKeyChecking=accept-new",
               f"{user}@{host}:{remote}", str(local)]
    else:
        if not args.command:
            ap.error("需要提供命令，或使用 --upload/--download")
        sys.exit(run(args.command))

    print(f"$ {' '.join(cmd)}", flush=True)
    child = pexpect.spawn(cmd[0], cmd[1:], timeout=1800, encoding="utf-8",
                          codec_errors="replace")
    try:
        i = child.expect([r"[Pp]assword:\s*$", pexpect.EOF], timeout=300)
        if i == 0:
            child.sendline(password)
            child.expect(pexpect.EOF, timeout=1800)
        sys.stdout.write(child.before or "")
    finally:
        child.close()
    print(f"[exit {child.exitstatus}]")


if __name__ == "__main__":
    main()
