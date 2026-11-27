#!/bin/bash
# Colab の VM（L4）の上で 1 モデル分を最初から最後まで流す。launch.py が裏で起動する（計画 版 8 の 12 章）。
# 使い方: job.sh <名前> <Hugging Face の置き場> <載せる層の数> <家で作った BF16 の sha256>
# 測るコード（evaluation/quant_probe）は固定したコミットの物をそのまま使う。ここは流す順番と確かめだけ。
# 途中の結果は段ごとに /content/out/<名前>/<段>.tgz に出す（家から 1 つずつ取る）。大きいファイルは入れない。
set -u
N=$1; R=$2; NGL=$3; SHA=$4
B=/content/qp; C=$B/code; O=/content/out/$N; S=$O/status.txt
W=$B/work8; M=$B/models/$N-bf16.gguf; BIN=$B/src/build/bin; PY=$B/venv/bin/python
LLAMA=f5027fb21307eb264aed483242603d68c87b0fe3     # b11836（家と同じ）
T0=$(date +%s); LIMIT=${QP_LIMIT_SEC:-37800}         # 10.5 時間を過ぎたら新しい段を始めない（1 回は 12 時間で切れる）
VARIANTS="Q6_K-none Q6_K-en Q6_K-ja Q5_K_M-none Q5_K_M-en Q5_K_M-ja Q4_K_M-none Q4_K_M-en Q4_K_M-ja Q3_K_M-none Q3_K_M-en Q3_K_M-ja Q2_K-none Q2_K-en Q2_K-ja"
mkdir -p "$O" $B/models $W/$N $B/logs
export PYTHONPATH=$C PYTHONUTF8=1 HF_HUB_DISABLE_PROGRESS_BARS=1

say() { echo "$(date -u '+%F %T') $*" | tee -a "$S"; }
pack() { # pack <札> <work8/名前 の下の道の型...>
  local tag=$1; shift
  ( cd $W/$N || exit 1; shopt -s nullglob; f=()
    for p in "$@"; do for x in $p; do [ -e "$x" ] && f+=("$x"); done; done
    [ ${#f[@]} -gt 0 ] || exit 1
    tar -czf "$O/$tag.tgz.tmp" "${f[@]}" && mv "$O/$tag.tgz.tmp" "$O/$tag.tgz" ) \
    && echo "$tag.tgz" >> "$O/index.txt"
}
setup_logs() { tar -C $B -czf "$O/setup-logs.tgz.tmp" logs 2>/dev/null && mv "$O/setup-logs.tgz.tmp" "$O/setup-logs.tgz"; }
die() { say "FAIL $*"; setup_logs; pack fail 'logs/*' >/dev/null 2>&1; echo FAIL > "$O/done"; exit 1; }
left() { [ $(( $(date +%s) - T0 )) -lt "$LIMIT" ]; }
pipe() { ( cd $C && python3 -m evaluation.quant_probe.pipeline --model $N --bf16 $M --texts $C/texts8 --work $W \
           --bin $BIN "$@" ) >> $B/logs/$N-pipeline.log 2>&1; }

say "start $N repo=$R ngl=$NGL code=$(cat $C/COMMIT 2>/dev/null) limit=${LIMIT}s"

# 1) 機械の種類（同じ種類の GPU でそろえる）
G=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1)
say "gpu $G / $(nproc) cpu / $(free -g | awk '/Mem/{print $2}')GB / disk $(df -BG --output=avail /content | tail -1 | tr -d ' ')"
case "$G" in *L4*) ;; *) die "GPU が L4 でない: $G";; esac

# 2) 問・規程・文章が家と同じか
FP=$(cd $C && python3 -c "from evaluation.quant_probe.questions import manifest; print(manifest()['total'])") || die "問の指紋が読めない"
[ "$FP" = "$(cat $C/FINGERPRINT)" ] || die "問と規程の指紋が違う: $FP"
( cd $C && sha256sum -c --quiet texts8.sha256 ) >> $B/logs/texts.log 2>&1 || die "texts8 の指紋が違う"
say "fingerprint ok $FP"

# 3) llama.cpp を同じコミットから組む（公式の配布に Linux の CUDA 版が無い）
if [ ! -x $BIN/llama-server ]; then
  say "build llama.cpp $LLAMA"
  ( cd $B && { [ -d src/.git ] || git clone --filter=blob:none https://github.com/ggml-org/llama.cpp src; } \
    && cd src && git checkout -q $LLAMA && [ "$(git rev-parse HEAD)" = "$LLAMA" ] \
    && cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=OFF \
    && cmake --build build -j"$(nproc)" --target llama-server llama-quantize llama-imatrix llama-perplexity \
  ) > $B/logs/build.log 2>&1 || die "llama.cpp が組めない（logs/build.log）"
fi
say "llama $($BIN/llama-server --version 2>&1 | tr '\n' ' ' | cut -c1-160)"

# 4) 変換（家と同じ版の部品・同じ置き場）。BF16 の sha256 が家と同じ時だけ進む
if [ ! -f $M.ok ]; then
  say "convert $R"
  ( { [ -x $PY ] || uv venv --python 3.12 $B/venv; } \
    && uv pip install --python $PY --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \
         -r $C/colab/convert-requirements.txt \
    && uv pip freeze --python $PY \
    && $PY -I $C/colab/prefetch.py $R \
    && $PY $B/src/convert_hf_to_gguf.py --remote $R --outtype bf16 --outfile $M \
  ) > $B/logs/$N-convert.log 2>&1 || die "変換に失敗（logs/$N-convert.log）"
  GOT=$(sha256sum $M | cut -d' ' -f1)
  [ "$GOT" = "$SHA" ] || die "BF16 の sha256 が家と違う: $GOT（家 $SHA）"
  touch $M.ok
fi
say "bf16 ok $SHA $(stat -c %s $M) bytes"

# 5) 前の台の続き（家から戻した段の結果）を置く
for f in /content/in-$N-*.tgz; do
  [ -f "$f" ] || continue
  tar -C $W/$N -xzf "$f" && say "restored $(basename $f)" || die "戻せない: $f"
done

# 6) 基準 Q8_0 を作り、自動で載る層を読む → 7) 動作確認（見済みの 10 問・家と同じ合格の線）
pipe --trial --print-offload || die "基準 Q8_0 が作れない"
RS=$(sha256sum $W/$N/ref-Q8_0.gguf | cut -d' ' -f1)
if [ -f $W/$N/ref.sha256 ] && [ "$(cat $W/$N/ref.sha256)" != "$RS" ]; then die "基準 Q8_0 が前の台と違う: $RS"; fi
echo "$RS" > $W/$N/ref.sha256; say "ref-Q8_0 $RS"
if [ ! -f $W/_smoke/$N/smoke.ok ]; then
  say "smoke"
  ( cd $C && python3 colab/smoke_colab.py $N $NGL $W $BIN/llama-server ) > $B/logs/$N-smoke.log 2>&1; rc=$?
  ( cd $W/_smoke/$N && tar -czf "$O/smoke-$(date +%s).tgz" . ) 2>/dev/null
  cp $W/_smoke/$N/smoke.json "$O/smoke.json" 2>/dev/null
  [ $rc -eq 0 ] || die "動作確認が線を外れた（smoke.json）。新しい問は流していない"
  touch $W/_smoke/$N/smoke.ok
fi
say "smoke ok $(python3 -c "import json;d=json.load(open('$W/_smoke/$N/smoke.json'));print(d['auto_offload'],d['same_text'],d['sec_per_q'])")"
setup_logs

# 8) 基準（imatrix・KLD の元・基準の答え 2 回）
if left; then
  say "ref"; pipe --ngl $NGL --only ref || die "ref（logs/$N-pipeline.log）"
  pack ref 'runs/ref-Q8_0-*' 'logs/ref-Q8_0*' 'logs/imatrix-*' 'imatrix-*.gguf' timing.jsonl ref_offload.json ref.sha256
  say "done ref"
else say "RESUME next=ref"; echo RESUME > "$O/done"; exit 0; fi

# 9) 15 変種を 1 つずつ（Q4_K_M-en は 2 回目も）
for v in $VARIANTS; do
  if ! left; then say "RESUME next=$v"; setup_logs; echo RESUME > "$O/done"; exit 0; fi
  say "variant $v"; pipe --ngl $NGL --only variants --variants $v || die "$v（logs/$N-pipeline.log）"
  pack "$v" "runs/$v.*" "runs/$v-r2.*" "logs/$v-*" variants.jsonl timing.jsonl
  say "done $v"
done
cp $B/logs/$N-pipeline.log "$O/pipeline.log"; setup_logs
say "ALLDONE $N $(( ($(date +%s) - T0) / 60 )) min"
echo OK > "$O/done"
