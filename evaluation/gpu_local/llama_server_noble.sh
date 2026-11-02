#!/bin/bash
# 配布版の llama-server (b11836・ubuntu・CUDA 12.8) は glibc 2.38 以上が要るので、Ubuntu 22.04 (glibc 2.35) では起動しない。
# Ubuntu 24.04 の公式パッケージ (libc6・libstdc++6・libgcc-s1・libgomp1) を .work/noble-libs/root に展開し、
# その ld.so で同じ配布版を起動する。重みと計算は配布版のまま (T4 の回と同じ実行ファイル)。
# CUDA の本体 (libcuda) は WSL2 が /usr/lib/wsl/lib に出す物を使う。
set -euo pipefail
W=${LLAMA_WORK:-$(cd "$(dirname "$0")/../../.work" && pwd)}
LIBS=$W/noble-libs/root/usr/lib/x86_64-linux-gnu
BIN=$(find "$W/llama-pre" -maxdepth 2 -name llama-server -printf '%h\n' | head -1)
CUDART=$(find "$W/llama-pre" -name 'libcudart.so*' -printf '%h\n' | head -1)
for need in "$LIBS/ld-linux-x86-64.so.2" "$BIN/llama-server" "$CUDART"; do
  [ -n "$need" ] && [ -e "$need" ] || { echo "準備が足りない: $W に配布版・cudart・noble-libs を置く (README.md)" >&2; exit 1; }
done
# ld.so から起動すると実行ファイルの場所が ld.so 側になり、計算部品 (libggml-cuda.so・libggml-cpu-*.so) を
# 探しに行かない。部品は今いるフォルダからも探すので、配布版のフォルダに移ってから起動する (重みは絶対パスで渡す)
cd "$BIN"
exec "$LIBS/ld-linux-x86-64.so.2" --library-path "$LIBS:$BIN:$CUDART:/usr/lib/wsl/lib" "$BIN/llama-server" "$@"
