#!/bin/bash
# 版 8（qp-judge-8）の本番を、モデルを 1 つずつ・段を 1 つずつ流す。途中で止まっても同じ命令で続きから。
# 使い方: PIN=<固定したコミット> bash evaluation/quant_probe/run_chain8.sh <名前>:<載せる層の数> [<名前>:<数> ...]
#   例: PIN=abc1234 bash evaluation/quant_probe/run_chain8.sh qwen3-8b:37 swallow-8b:33 gemma4-12b:38
# - 段（基準・変種 1 つ）の前に毎回、GPU をほかの処理（ゲーム・別の研究）が使っていないかを見て、使っていたら待つ。
#   待っても答えは変わらない（答えの回は毎回サーバーを立て直し、載せる層の数は固定）。
# - コードが固定したコミットと違う・書きかけがある時は、何も流さずに止まる。
D=${QP_DATA:-$HOME/qp-data}; R=$(cd "$(dirname "$0")/../.." && pwd); cd "$R" || exit 1
export PYTHONUTF8=1
S=$D/logs/run8.status; mkdir -p "$D/logs" "$D/results8"
GPU_MAX_MB=${GPU_MAX_MB:-3200}; GPU_MAX_UTIL=${GPU_MAX_UTIL:-35}
say() { echo "$(date '+%F %T') $*" >> "$S"; }
pinned() {
  [ -n "$PIN" ] || { say "STOP PIN が無い"; return 1; }
  [ "$(git rev-parse --short=7 HEAD)" = "${PIN:0:7}" ] || { say "STOP HEAD $(git rev-parse --short=7 HEAD) が固定 $PIN と違う"; return 1; }
  [ -z "$(git status --porcelain -- evaluation)" ] || { say "STOP evaluation に書きかけがある"; return 1; }
}
wait_gpu() {
  local ok=0 told=0 u m
  while [ $ok -lt 3 ]; do
    read -r u m < <(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | head -1 | tr -d ',')
    if [ "${m:-99999}" -lt "$GPU_MAX_MB" ] && [ "${u:-100}" -lt "$GPU_MAX_UTIL" ]; then ok=$((ok+1)); sleep 20
    else ok=0; [ $told -eq 0 ] && say "wait gpu ($1) util=$u mem=$m" && told=1; sleep 300; fi
  done
  [ $told -eq 1 ] && say "gpu free ($1)"; return 0
}
step() {  # step <名前> <数> <段の名前> <pipeline の引数...>
  local n=$1 ngl=$2 tag=$3; shift 3
  pinned || exit 3
  wait_gpu "$n $tag"
  python -m evaluation.quant_probe.pipeline --model "$n" --bf16 "$D/models/$n-bf16.gguf" --texts "$D/texts8" --work "$D/work8" \
    --bin "$HOME/tools/llama-b11836/bin" --ngl "$ngl" "$@" >> "$D/logs/v8-$n-pipeline.log" 2>&1
  local rc=$?; say "$n $tag rc=$rc"; [ $rc -eq 0 ] || exit $rc
}
VARIANTS=$(python -c "from evaluation.quant_probe.pipeline import ALL_VARIANTS; print(' '.join(ALL_VARIANTS))") || exit 1
for spec in "$@"; do
  n=${spec%%:*}; ngl=${spec##*:}
  say "start $n ngl=$ngl commit=$(git rev-parse --short=7 HEAD)"
  step "$n" "$ngl" ref --only ref
  for v in $VARIANTS; do step "$n" "$ngl" "$v" --only variants --variants "$v"; done
  python -m evaluation.quant_probe.report --work "$D/work8" --models "$n" --out "$D/results8/$n" >> "$D/logs/v8-$n-report.log" 2>&1
  say "report $n rc=$?"
done
say "ALLDONE $*"
