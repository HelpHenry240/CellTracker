#!/usr/bin/env python3
"""仓库的 ideas pipeline 入口；与 paperpipe 入口使用同一套实现。"""
from pathlib import Path
import runpy

if __name__=='__main__':
    runpy.run_path(str(Path(__file__).resolve().parents[1]/'paperpipe/scripts/run_paper_pipeline.py'),run_name='__main__')
