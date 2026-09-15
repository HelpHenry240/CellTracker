#!/usr/bin/env bash
# 多线程分段下载（服务器支持 Range 时），支持断点续传。
# 用法: bash scripts/pdownload.sh <url> <输出文件> [分片数=16]
set -euo pipefail

url="${1:?用法: pdownload.sh <url> <out> [chunks]}"
out="${2:?用法: pdownload.sh <url> <out> [chunks]}"
nchunk="${3:-16}"

parts="${out}.parts"
mkdir -p "$parts"

# 取总大小：依次尝试 3 种方法（代理缓存偶尔不带 Content-Range）
probe_size() {
  local u="$1"
  # 1) Range 请求的 Content-Range: bytes 0-0/TOTAL
  curl -sL -o /dev/null -D - -r 0-0 "$u" 2>/dev/null \
    | tr -d '\r' | awk 'tolower($1)=="content-range:"{split($2,a,"/"); if(a[2] ~ /^[0-9]+$/) print a[2]}' | tail -1
}
probe_size_head() {
  local u="$1"
  curl -sIL "$u" 2>/dev/null \
    | tr -d '\r' | awk 'tolower($1)=="content-length:"{print $2}' | tail -1
}

total="$(probe_size "$url" || true)"
if [[ -z "${total:-}" ]]; then
  total="$(probe_size "$url" || true)"
fi
if [[ -z "${total:-}" ]]; then
  total="$(probe_size_head "$url" || true)"
fi
if [[ -z "${total:-}" ]]; then
  echo "无法获取文件大小，回退单线程 curl" >&2
  curl -fL --retry 5 -C - -o "$out.partial" "$url"
  mv "$out.partial" "$out"
  exit 0
fi
echo "总大小: $total bytes ($(( total / 1048576 )) MB), 分片: $nchunk"

chunk_size=$(( (total + nchunk - 1) / nchunk ))
pids=()

for ((i=0; i<nchunk; i++)); do
  start=$(( i * chunk_size ))
  [[ $start -ge $total ]] && break
  end=$(( start + chunk_size - 1 ))
  [[ $end -ge $total ]] && end=$(( total - 1 ))
  expect=$(( end - start + 1 ))
  part="$parts/part_$(printf '%03d' "$i")"

  if [[ -f "$part" ]] && [[ "$(stat -c%s "$part")" == "$expect" ]]; then
    echo "分片 $i 已完成，跳过"
    continue
  fi

  (
    # 逐块续传：每次只请求"尚未拿到的字节区间"，最多重试 60 次
    for attempt in $(seq 1 60); do
      cur=$(stat -c%s "$part" 2>/dev/null || echo 0)
      if [[ "$cur" -ge "$expect" ]]; then
        # 多下或刚好：按需截断
        [[ "$cur" -gt "$expect" ]] && truncate -s "$expect" "$part"
        exit 0
      fi
      if curl -fsL --max-time 600 -r "$(( start + cur ))-${end}" -o - "$url" >> "$part"; then
        continue
      fi
      echo "分片 $i 第 $attempt 次中断（已获取 $cur/$expect 字节），续传中..." >&2
      sleep $(( attempt < 5 ? attempt * 2 : 10 ))
    done
    echo "分片 $i 下载失败" >&2
    exit 1
  ) &
  pids+=($!)
done

echo "等待 ${#pids[@]} 个分片..."
fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done
[[ $fail -eq 0 ]] || { echo "存在失败分片，重跑本脚本可续传" >&2; exit 1; }

echo "合并分片..."
cat "$parts"/part_* > "$out"
actual=$(stat -c%s "$out")
if [[ "$actual" != "$total" ]]; then
  echo "大小不符: 期望 $total 实际 $actual" >&2
  exit 1
fi
rm -rf "$parts"
echo "完成: $out ($(( actual / 1048576 )) MB)"
