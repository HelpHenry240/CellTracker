"""实验运行器：保证每个实验自动留下"八件套"痕迹。

产物::

    experiments/<exp_id>/
      config.yaml  command.sh  env.txt  git_commit.txt
      metrics.json  logs/run.log  artifacts/  figures/  notes.md(人工)
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import shlex
import time
from pathlib import Path

import yaml


class Experiment:
    def __init__(self, exp_id: str, purpose: str, params: dict | None = None,
                 root: str | Path = ".", resume: bool = False):
        self.root = Path(root).resolve()
        self.exp_id = exp_id
        self.dir = self.root / "experiments" / exp_id
        if self.dir.exists() and not resume and (self.dir / "metrics.json").exists():
            raise FileExistsError(
                f"{self.dir} 已有 metrics.json。结果只增不改，请换新编号（如 -fix1）。")
        for sub in ("logs", "artifacts", "figures"):
            (self.dir / sub).mkdir(parents=True, exist_ok=True)

        self.config = {
            "experiment_id": exp_id,
            "date": time.strftime("%Y-%m-%d"),
            "purpose": purpose,
            "params": params or {},
            "argv": sys.argv,
            "seed": (params or {}).get("seed", 20260915),
            "status": "running",
        }
        (self.dir / "config.yaml").write_text(
            yaml.safe_dump(self.config, allow_unicode=True, sort_keys=False))
        (self.dir / "command.sh").write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\ncd \"$(dirname \"$0\")/../..\"\n"
            "# 复现命令（自动记录）\n" + shlex.join([sys.executable,*sys.argv]) + "\n")
        self._write_env()
        self._write_git()
        self.log_path = self.dir / "logs" / "run.log"
        self._log_lines: list[str] = []
        self.log(f"experiment {exp_id} start: {purpose}")

    # ---------- 留痕 ----------

    def _write_env(self) -> None:
        try:
            freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                                    capture_output=True, text=True, timeout=120).stdout
        except Exception:  # noqa: BLE001
            freeze = "(pip freeze failed)"
        lines = [
            f"date: {time.strftime('%Y-%m-%dT%H:%M:%S')}",
            f"host: {platform.node()}",
            f"platform: {platform.platform()}",
            f"cpu_cores: {os.cpu_count()}",
            f"python: {sys.version.split()[0]} ({sys.executable})",
            f"gpu: {self._gpu_info()}",
            "--- pip freeze ---",
            freeze,
        ]
        (self.dir / "env.txt").write_text("\n".join(lines))

    @staticmethod
    def _gpu_info() -> str:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total",
                 "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
            return out.stdout.strip() or "none"
        except Exception:  # noqa: BLE001
            return "none"

    def _write_git(self) -> None:
        try:
            commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=self.root,
                                    capture_output=True, text=True, timeout=30).stdout.strip()
            dirty = subprocess.run(["git", "status", "--porcelain"], cwd=self.root,
                                   capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception:  # noqa: BLE001
            commit, dirty = "unknown", ""
        flag = "dirty" if dirty else "clean"
        if not commit and (self.root/'SOURCE_VERSION.json').is_file():
            source=json.loads((self.root/'SOURCE_VERSION.json').read_text())
            commit=source.get('git_commit','unknown')
            flag='cloud source release '+source.get('source_sha256','unknown')
        (self.dir / "git_commit.txt").write_text(f"{commit}\n# {flag}\n")

    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self._log_lines.append(line)
        self.log_path.write_text("\n".join(self._log_lines) + "\n")

    # ---------- 产物 ----------

    def save_metrics(self, metrics: dict, name: str = "metrics.json") -> Path:
        path = self.dir / name
        path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
        self.log(f"metrics -> {path}")
        return path

    def save_table(self, text: str, name: str) -> Path:
        path = self.dir / name
        path.write_text(text)
        return path

    def artifact_dir(self, name: str) -> Path:
        d = self.dir / "artifacts" / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def figure_path(self, name: str) -> Path:
        return self.dir / "figures" / name

    def finish(self, status: str = "done", summary: str = "") -> None:
        self.config["status"] = status
        if summary:
            self.config["summary"] = summary
        (self.dir / "config.yaml").write_text(
            yaml.safe_dump(self.config, allow_unicode=True, sort_keys=False))
        self.log(f"experiment {self.exp_id} finished: {status}")
