#!/bin/bash
# WSL の Ubuntu 22.04 で llama.cpp の配布版 (b11836・ubuntu・CUDA 12.8) の道具を動かす。
# evaluation/gpu_local/llama_server_noble.sh と同じやり方で、最初の引数に道具の名前を取る。
#   llama_noble.sh llama-perplexity -m ... / llama_noble.sh llama-server -m ...
# 配布版は glibc 2.38 以上が要るので、Ubuntu 24.04 の libc を LLAMA_WORK/noble-libs/root に展開し、その ld.so で起動する。
# 道具のフォルダに移ってから起動する (計算部品を今のフォルダから探すため)。引数の道は絶対にする。
set -euo pipefail
TOOL=${1:?道具の名前 (llama-server など) が要る}
shift
W=${LLAMA_WORK:-$(cd "$(dirname "$0")/../../.work" && pwd)}
LIBS=$W/noble-libs/root/usr/lib/x86_64-linux-gnu
BIN=$(find "$W/llama-pre" -maxdepth 2 -name llama-server -printf '%h\n' | head -1)
CUDART=$(find "$W/llama-pre" -name 'libcudart.so*' -printf '%h\n' | head -1)
for need in "$LIBS/ld-linux-x86-64.so.2" "$BIN/$TOOL" "$CUDART"; do
  [ -n "$need" ] && [ -e "$need" ] || { echo "準備が足りない: $W に配布版・cudart・noble-libs を置く ($need)" >&2; exit 1; }
done
cd "$BIN"
exec "$LIBS/ld-linux-x86-64.so.2" --library-path "$LIBS:$BIN:$CUDART:/usr/lib/wsl/lib" "$BIN/$TOOL" "$@"
