#!/usr/bin/env bash
# 把 paperpipe/ 打包成压缩包（含清单 MANIFEST.txt），并**从压缩包里解出来做一次校验**。
#
# 用法（仓库根目录）::
#
#     bash paperpipe/scripts/make_package.sh            # 输出到 dist/
#     OUT=/tmp bash paperpipe/scripts/make_package.sh   # 换输出目录
#
# 产物：
#   dist/paperpipe_<日期>.tar.gz   （Linux/macOS 友好）
#   dist/paperpipe_<日期>.zip      （Windows 友好，zip 可用时）
#
# 打包内容 = `paperpipe/` 全量，**排除** `__pycache__/` 与 `*.pyc`；
# 额外写入 `paperpipe/MANIFEST.txt`：git commit、打包时间、文件清单与 md5、
# 自包含说明、验证命令、以及端到端官方结果指针。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"     # 仓库根
SRC="$ROOT/paperpipe"
STAMP="$(date +%Y%m%d)"
OUT_DIR="${OUT:-$ROOT/dist}"
NAME="paperpipe_${STAMP}"
STAGE="$(mktemp -d)"

mkdir -p "$OUT_DIR"
echo "=== [1/4] 暂存（排除 __pycache__/*.pyc）→ $STAGE/$NAME ==="
rsync -a --exclude '__pycache__' --exclude '*.pyc' "$SRC/" "$STAGE/$NAME/"

echo "=== [2/4] 写 MANIFEST.txt ==="
{
  echo "paperpipe —— 严格按 ideas.pdf 重建的 OT+GNN 细胞追踪 pipeline"
  echo "打包时间   : $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "git commit : $(git -C "$ROOT" log --oneline -1 2>/dev/null || echo '(非 git 工作区)')"
  echo "git dirty  : $(git -C "$ROOT" status --porcelain paperpipe 2>/dev/null | wc -l) 项未提交改动"
  echo
  echo "自包含说明 : src/vendor/celltracker/ 是复用到的原仓库模块副本；"
  echo "             paperpipe/src/papertrack/_paths.py 会把它插到 sys.path 最前，"
  echo "             因此本压缩包**不依赖外层仓库**即可运行。"
  echo "自检       : python -c \"import sys;sys.path.insert(0,'paperpipe/src');"
  echo "             import papertrack, celltracker; print(celltracker.__file__)\""
  echo "             → 应打印 .../paperpipe/src/vendor/celltracker/__init__.py"
  echo "测试       : python -m pytest paperpipe/tests -q   （16 项）"
  echo
  echo "端到端官方指标（2026-09-24，云端 nnU-Net + CTC 官方二进制）："
  echo "  seq01 SEG 0.673196 / DET 0.938833 / TRA 0.898249"
  echo "  seq02 SEG 0.690867 / DET 0.938995 / TRA 0.899119（留出序列）"
  echo "  细节见 FORMULA_MAP.md 与 experiments/P4_final_gnn_ce01/notes.md"
  echo
  echo "--- 文件清单（相对 paperpipe/，含 sha256 与大小）---"
  ( cd "$STAGE/$NAME" && find . -type f -not -name MANIFEST.txt | sort | \
      while read -r f; do printf '%s  %10s  %s\n' "$(sha256sum "$f" | cut -c1-16)" \
          "$(stat -c %s "$f")" "${f#./}"; done )
  echo
  echo "--- 关键权重指纹 ---"
  ( cd "$STAGE/$NAME" && for f in runs/*/*.pt; do [ -e "$f" ] && \
      printf '%s  %s\n' "$(md5sum "$f" | cut -d' ' -f1)" "$f"; done )
} > "$STAGE/$NAME/MANIFEST.txt"

echo "=== [3/4] 打包 → $OUT_DIR/$NAME.{tar.gz,zip} ==="
tar czf "$OUT_DIR/$NAME.tar.gz" -C "$STAGE" "$NAME"
if command -v zip >/dev/null 2>&1; then
  ( cd "$STAGE" && zip -qr "$OUT_DIR/$NAME.zip" "$NAME" )
else
  # 系统没有 zip 时用 Python 的 zipfile 兜底（Windows 用户更常用 zip）
  python3 -c "
import shutil, sys
shutil.make_archive(sys.argv[1], 'zip', sys.argv[2], sys.argv[3])
" "$OUT_DIR/$NAME" "$STAGE" "$NAME"
fi

echo "=== [4/4] 从压缩包解出并校验（自包含 import）==="
VERIFY="$(mktemp -d)"
tar xzf "$OUT_DIR/$NAME.tar.gz" -C "$VERIFY"
( cd /tmp && PYTHONPATH="$VERIFY/$NAME/src" python3 -c "
import papertrack, celltracker
assert 'vendor' in celltracker.__file__, celltracker.__file__
from papertrack.runtime import run_pipeline
from papertrack.reconstruction import export_ctc
print('  解包校验通过：celltracker ->', celltracker.__file__.split('/')[-4:])
print('  run_pipeline/export_ctc 可导入 ✔')" )
ls -lh "$OUT_DIR/$NAME.tar.gz" | awk '{print "  "$9"  "$5}'
[ -f "$OUT_DIR/$NAME.zip" ] && ls -lh "$OUT_DIR/$NAME.zip" | awk '{print "  "$9"  "$5}'
rm -rf "$STAGE" "$VERIFY"
echo "=== 完成 ==="
