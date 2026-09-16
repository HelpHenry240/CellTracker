#!/usr/bin/env python3
"""解析官方 TRA_log.txt，输出 AOGM 误差分解（定位主要失分项）。

用法::

    python scripts/analyze_tra_log.py experiments/*/official_logs/TRA_log.txt
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# CTC AOGM 权重（Matula et al. 2015）
WEIGHTS = {"NS": 5.0, "FN": 10.0, "FP": 1.0, "ED": 1.0, "EA": 1.5, "EC": 1.0}

SECTIONS = [
    ("NS", r"Splitting Operations", "NS"),
    ("FN", r"False Negative Vertices", "FN"),
    ("FP", r"False Positive Vertices", "FP"),
    ("ED", r"Redundant Edges To Be Deleted", "ED"),
    ("EA", r"Edges To Be Added", "EA"),
    ("EC", r"Edges with Wrong Semantics", "EC"),
]


def parse(path: Path) -> dict:
    lines = path.read_text(errors="replace").splitlines()
    counts = {k: 0 for k in WEIGHTS}
    current = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("-----"):
            current = None
            for key, pattern, _ in SECTIONS:
                if re.search(pattern, stripped, re.IGNORECASE):
                    current = key
                    break
            continue
        if stripped.startswith("TRA measure"):
            current = None
            continue
        if current and stripped and not stripped.startswith("="):
            counts[current] += 1
    tra = None
    m = re.search(r"TRA measure:\s*([0-9.]+)", path.read_text(errors="replace"))
    if m:
        tra = float(m.group(1))
    return {"counts": counts, "TRA": tra}


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        print(__doc__)
        return
    print(f"{'log':60s} {'NS':>5s} {'FN':>5s} {'FP':>5s} {'ED':>5s} {'EA':>5s} {'EC':>5s}"
          f" {'AOGM':>8s} {'TRA':>9s}")
    for p in paths:
        r = parse(p)
        c = r["counts"]
        aogm = sum(WEIGHTS[k] * c[k] for k in WEIGHTS)
        name = str(p.parent.parent.name)[:58] if p.parent.parent.name else str(p)[:58]
        print(f"{name:60s} {c['NS']:5d} {c['FN']:5d} {c['FP']:5d} {c['ED']:5d} "
              f"{c['EA']:5d} {c['EC']:5d} {aogm:8.1f} "
              f"{('%.6f' % r['TRA']) if r['TRA'] is not None else '—':>9s}")


if __name__ == "__main__":
    main()
