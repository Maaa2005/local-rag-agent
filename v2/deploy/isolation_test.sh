#!/usr/bin/env bash
# 双方向分離試験（N3）。起動済みの compose に対して docker compose exec で実際に接続を試みる。
# ネットワーク設定名だけでは合格にしない。各項目 PASS/FAIL を出し、1 つでも FAIL なら非 0 終了。
#
# 使い方（v2/ から、compose 起動後）:
#   bash deploy/isolation_test.sh
# 環境変数:
#   COMPOSE_FILE    既定 deploy/compose.yml
#   COMPOSE_PROFILE 既定 laya（clef 構成なら clef）
#   FRONTEND_PORT   既定 3000（ホスト 127.0.0.1 に公開した frontend のポート）
#   RELAY_PORT      既定 18080（ホスト上に一時的に立てる中継役リスナのポート）
#   EXTERNAL_HOST   既定 example.com（社内から出られないことを確かめる外部宛先）
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
COMPOSE_FILE="${COMPOSE_FILE:-$HERE/compose.yml}"
COMPOSE_PROFILE="${COMPOSE_PROFILE:-laya}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"
RELAY_PORT="${RELAY_PORT:-18080}"
EXTERNAL_HOST="${EXTERNAL_HOST:-example.com}"
DC=(docker compose -f "$COMPOSE_FILE" --profile "$COMPOSE_PROFILE")

FAILS=0
pass() { printf 'PASS  %s\n' "$1"; }
fail() { printf 'FAIL  %s\n' "$1"; FAILS=$((FAILS + 1)); }
info() { printf 'INFO  %s\n' "$1"; }

# コンテナ内で python を実行する（internal-app / gateway はどちらも python:3.11 系）
pyexec() { local svc="$1"; shift; "${DC[@]}" exec -T "$svc" python - "$@"; }

# TCP 接続を試みる。出力: OPEN / CLOSED:<理由>
PROBE_TCP='
import socket, sys
host, port = sys.argv[1], int(sys.argv[2])
try:
    s = socket.create_connection((host, port), timeout=4)
    s.close(); print("OPEN")
except Exception as e:
    print("CLOSED:" + type(e).__name__)
'
# 名前解決を試みる。出力: RESOLVED:<ip> / NXDOMAIN:<理由>
PROBE_DNS='
import socket, sys
try:
    print("RESOLVED:" + socket.gethostbyname(sys.argv[1]))
except Exception as e:
    print("NXDOMAIN:" + type(e).__name__)
'
# HTTP GET を試みる。出力: HTTP:<code> / ERR:<理由>
PROBE_HTTP='
import sys, urllib.request, urllib.error
try:
    r = urllib.request.urlopen(sys.argv[1], timeout=6); print("HTTP:%d" % r.status)
except urllib.error.HTTPError as e:
    print("HTTP:%d" % e.code)
except Exception as e:
    print("ERR:" + type(e).__name__)
'
# デフォルトゲートウェイ（ホスト側ブリッジ）の IP。無ければ空。
PROBE_GW='
import socket, struct
try:
    for line in open("/proc/net/route").read().splitlines()[1:]:
        f = line.split()
        if f[1] == "00000000":
            print(socket.inet_ntoa(struct.pack("<L", int(f[2], 16)))); break
except Exception:
    pass
'
# パスの存在確認。出力: EXISTS / ABSENT
PROBE_PATH='
import os, sys
print("EXISTS" if os.path.exists(sys.argv[1]) else "ABSENT")
'
# UDS 経由 HTTP。出力: HTTP:<code> / ERR:<理由>
PROBE_UDS='
import socket, sys
path = sys.argv[1]
try:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); s.settimeout(5); s.connect(path)
    s.sendall(b"GET /status/isolation-probe HTTP/1.1\r\nHost: gateway\r\nConnection: close\r\n\r\n")
    line = s.recv(64).split(b"\r\n")[0].decode()
    print("HTTP:" + line.split()[1] if line.startswith("HTTP/") else "ERR:bad-response")
except Exception as e:
    print("ERR:" + type(e).__name__)
'

tcp()  { echo "$PROBE_TCP"  | pyexec "$1" "$2" "$3" 2>/dev/null | tail -1; }
dns()  { echo "$PROBE_DNS"  | pyexec "$1" "$2" 2>/dev/null | tail -1; }
http() { echo "$PROBE_HTTP" | pyexec "$1" "$2" 2>/dev/null | tail -1; }
path() { echo "$PROBE_PATH" | pyexec "$1" "$2" 2>/dev/null | tail -1; }

expect_closed() {  # 説明 結果
  case "$2" in
    CLOSED:*|ERR:*|NXDOMAIN:*) pass "$1 ($2)";;
    "") fail "$1 (試験を実行できない: exec 失敗)";;
    *) fail "$1 ($2)";;
  esac
}

# 社内サービスのコンテナ IP（名前解決を経ずに直接接続を試すため）
ip_of() {
  local cid; cid="$("${DC[@]}" ps -q "$1" 2>/dev/null | head -1)"
  [ -n "$cid" ] || return 0
  docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}} {{end}}' "$cid"
}

echo "== 前提確認 =="
for svc in internal-app gateway; do
  if [ -z "$("${DC[@]}" ps -q "$svc" 2>/dev/null)" ]; then
    fail "$svc が起動していない（試験不能）"
  fi
done
[ "$FAILS" -eq 0 ] || { echo "前提を満たさないため中断"; exit 1; }

# ホスト上の中継役を模したリスナを 0.0.0.0 に立てる（社内→ホスト中継の試験用）
RELAY_PID=""
python3 -m http.server "$RELAY_PORT" --bind 0.0.0.0 >/dev/null 2>&1 &
RELAY_PID=$!
trap '[ -n "$RELAY_PID" ] && kill "$RELAY_PID" 2>/dev/null' EXIT
sleep 1
if curl -s -o /dev/null -m 3 "http://127.0.0.1:$RELAY_PORT/"; then
  info "ホスト中継役リスナ 0.0.0.0:$RELAY_PORT 起動（陽性対照: ホストからは到達可）"
else
  fail "ホスト中継役リスナを起動できない（社内→ホスト中継の試験が無効）"
fi

echo "== 1. 社内 → 外部 =="
expect_closed "internal-app → https://$EXTERNAL_HOST (HTTP)" "$(http internal-app "https://$EXTERNAL_HOST/")"
expect_closed "internal-app → 1.1.1.1:443 (IP 直指定 TCP)" "$(tcp internal-app 1.1.1.1 443)"
expect_closed "internal-app → 8.8.8.8:53 (IP 直指定 TCP)" "$(tcp internal-app 8.8.8.8 53)"
expect_closed "internal-app → host.docker.internal:$RELAY_PORT (ホスト中継)" "$(tcp internal-app host.docker.internal "$RELAY_PORT")"
GW_INT="$(echo "$PROBE_GW" | pyexec internal-app 2>/dev/null | tail -1)"
if [ -n "$GW_INT" ]; then
  expect_closed "internal-app → デフォルトGW $GW_INT:$RELAY_PORT (ホスト中継)" "$(tcp internal-app "$GW_INT" "$RELAY_PORT")"
else
  pass "internal-app にデフォルトルートが無い（ホストブリッジ経由の中継経路なし）"
fi
# frontend は frontend-publish（非 internal）にも参加するため、社内から frontend を踏み台にした外部到達は
# frontend 自体の挙動に依存する。プレースホルダ nginx はプロキシしないので現時点は対象外（README 参照）。

echo "== 2. Gateway → 社内 =="
for name in internal-app qdrant vllm judge judge-laya judge-clef frontend; do
  expect_closed "gateway で $name を名前解決できない" "$(dns gateway "$name")"
done
# 名前解決を経ずに IP 直指定でも届かないこと
for spec in "qdrant 6333" "qdrant 6334" "vllm 8000" "judge-laya 8080" "judge-clef 8080" "frontend 8080"; do
  set -- $spec
  ips="$(ip_of "$1")"
  if [ -z "$ips" ]; then info "$1 は未起動（profile 外）のため IP 直指定試験を省略"; continue; fi
  for ip in $ips; do
    expect_closed "gateway → $1 ($ip:$2) IP 直指定" "$(tcp gateway "$ip" "$2")"
  done
done
for p in /docs /internal-data /models /qdrant/storage /var/run/docker.sock /run/docker.sock; do
  r="$(path gateway "$p")"
  if [ "$r" = "ABSENT" ]; then pass "gateway に $p が存在しない"; else fail "gateway に $p が存在する (${r:-exec失敗})"; fi
done

echo "== 3. Gateway → frontend → 社内 の迂回 =="
expect_closed "gateway → host.docker.internal:$FRONTEND_PORT (公開 frontend)" "$(tcp gateway host.docker.internal "$FRONTEND_PORT")"
GW_GW="$(echo "$PROBE_GW" | pyexec gateway 2>/dev/null | tail -1)"
if [ -n "$GW_GW" ]; then
  expect_closed "gateway → デフォルトGW $GW_GW:$FRONTEND_PORT (公開 frontend)" "$(tcp gateway "$GW_GW" "$FRONTEND_PORT")"
else
  fail "gateway にデフォルトルートが無い（egress が機能していない可能性）"
fi
expect_closed "gateway → 127.0.0.1:$FRONTEND_PORT" "$(tcp gateway 127.0.0.1 "$FRONTEND_PORT")"

echo "== 4. 正常経路 =="
r="$(echo "$PROBE_UDS" | pyexec internal-app /run/gateway/gateway.sock 2>/dev/null | tail -1)"
case "$r" in
  HTTP:200|HTTP:404) pass "internal-app → UDS → gateway /status 応答 ($r)";;
  *) fail "internal-app → UDS → gateway /status 応答なし (${r:-exec失敗})";;
esac
r="$(tcp internal-app qdrant 6333)"
if [ "$r" = "OPEN" ]; then pass "internal-app → qdrant:6333 接続可"; else fail "internal-app → qdrant:6333 接続不可 ($r)"; fi
r="$(http gateway "https://$EXTERNAL_HOST/")"
case "$r" in
  HTTP:*) info "gateway → 外部 HTTP 到達可 ($r)（egress 陽性対照）";;
  *) info "gateway → 外部 HTTP 到達不可 ($r)（試験環境に外部接続が無い可能性。分離判定には使わない）";;
esac

echo "== 結果 =="
if [ "$FAILS" -eq 0 ]; then echo "ALL PASS"; exit 0; else echo "FAIL: $FAILS 件"; exit 1; fi
