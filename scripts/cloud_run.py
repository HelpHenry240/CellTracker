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
import os
import re
import sys
from pathlib import Path

import pexpect

ROOT = Path(__file__).resolve().parent.parent
CRED = ROOT / "本地文档" / "ssh&key"


def parse_creds() -> list[tuple[str, str, str, str]]:
    """解析凭据文件里的**所有**连接（支持主用 + 备用）。

    文件格式（见 `本地文档/ssh&key`）：

        ssh -p 10505 root@<host>      # 主用
        <password>

        备用连接：
        ssh -p 21921 root@<host>      # 备用
        <password>

    中文标签行（如"备用连接："）会被跳过，不作为密码。
    """
    entries: list[dict[str, str]] = []
    cur: dict[str, str] | None = None
    for raw in CRED.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        m = re.match(r"ssh\s+-p\s+(\d+)\s+(\S+)@(\S+)", line)
        if m:
            cur = {"port": m.group(1), "user": m.group(2), "host": m.group(3),
                   "password": ""}
            entries.append(cur)
            continue
        if cur is None or cur["password"]:
            continue
        if re.search(r"[\u4e00-\u9fff:：]", line):
            continue                       # 中文标签行
        cur["password"] = line
    return [(e["host"], e["port"], e["user"], e["password"])
            for e in entries if e["password"]]


def parse_cred(which: str | None = None) -> tuple[str, str, str, str]:
    """选择要用的连接：`primary`（默认）/ `backup`，或用环境变量 `CT_CLOUD_CONN`。

    云服务器有时只开放备用端口，此时 `CT_CLOUD_CONN=backup bash scripts/cloud_run.sh ...`。
    也可以直接给端口号或主机名进行匹配。
    """
    creds = parse_creds()
    if not creds:
        sys.exit(f"无法从 {CRED} 解析任何 ssh 连接")
    sel = (which or os.environ.get("CT_CLOUD_CONN") or "primary").strip().lower()
    if sel in ('backup3','backup_3','备用3','d','4'):
        if len(creds)<4:
            sys.exit(f'{CRED} 中未找到备用3连接')
        return creds[3]
    if sel in ("backup2", "backup_2", "备用2", "c", "3"):
        if len(creds) < 3:
            sys.exit(f"{CRED} 中未找到备用2连接")
        return creds[2]
    if sel in ("backup", "backup1", "b", "2"):
        if len(creds) < 2:
            sys.exit(f"{CRED} 中未找到备用连接")
        return creds[1]
    if sel in ("primary", "main", "a", "1"):
        return creds[0]
    for c in creds:
        if sel in (c[1].lower(), c[0].lower()):
            return c
    sys.exit(f"未知连接选择 {sel!r}（可用：primary/backup/backup2/backup3，或端口号/主机名）")


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
