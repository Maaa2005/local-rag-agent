#!/bin/bash
# 試験 3 回目の最初の読み。合否用の 2 回（f1 f2）と、ばらつき用の 5 回（乱数 3・入力 2）を順に流す。鍵は全部終わるまで持つ。
L=~/.cache/lr-gpu; PY=$L/env/Scripts/python.exe; export PYTHONUTF8=1
W=$(cygpath -w $L); SV="$W\llama\bin\llama-server.exe"; GG="$W\gguf\gemma-4-E4B-it-Q4_K_M.gguf"
cd ~/dev/.wt/lr-askback; export PYTHONPATH="backend;."
Q=evaluation/v2/ask90b/questions.jsonl; N=$W/ask90b
for i in $(seq 1 360); do mkdir ~/.cache/gpu-lock 2>/dev/null && break; sleep 10; done
[ -d ~/.cache/gpu-lock ] || { echo NOLOCK; exit 9; }
trap 'rmdir ~/.cache/gpu-lock' EXIT
run() {  # 名前 問 温度 種
  for i in $(seq 1 120); do a=$($PY -c "import psutil;print(psutil.virtual_memory().available//2**20)"); [ "$a" -ge 3072 ] && break; sleep 10; done
  echo "== $1 temp=$3 seed=$4 空き $a $(date +%T)"
  timeout 3000 $PY -m evaluation.gpu_local.open_read run --arm A --cards "$N\cards" --questions "$2" --out "$N\run_$1" --server "$SV" --gguf "$GG" --round 1 --temperature "$3" --seed "$4" </dev/null 2>&1 | tail -n 2 | cut -c1-200
  echo "== $1 終了 $(date +%T)"
}
run f1 "$Q" 0 1
run f2 "$Q" 0 1
run t11 "$Q" 0.3 11
run t12 "$Q" 0.3 12
run t13 "$Q" 0.3 13
run n1 "$N\questions_n1.jsonl" 0 1
run n2 "$N\questions_n2.jsonl" 0 1
echo ROUNDDONE $(date +%T)
