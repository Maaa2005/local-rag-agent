# v2 配置構成（Docker Compose）初版

## 構成図

```
                      ホスト 127.0.0.1:${FRONTEND_PORT:-3000}
                                   │
                         [frontend-publish (bridge)]
                                   │
 ┌──────────── network: internal (internal: true) ─────────────┐
 │  frontend   internal-app   qdrant   vllm*   judge(laya|clef) │
 └──────────────────────┬──────────────────────────────────────┘
                        │ volume: gateway-sock（Unix ソケット /run/gateway/gateway.sock）
                        │ HTTP over UDS のみ。ネットワークは共有しない
                 ┌──────┴──────┐
                 │   gateway   │── network: egress (bridge) ──> 外部 API（Claude / Codex）
                 └─────────────┘
                   volume: gateway-data（Gateway 専用 SQLite）
                   secrets: anthropic_api_key / openai_api_key（gateway のみ）

 * vllm は profile laya のときだけ起動。profile clef では止めて judge-clef（Clef-flash）が GPU を使う。
```

| ゾーン | サービス | ネットワーク | volume |
|---|---|---|---|
| 社内 | internal-app（Orchestrator・本体未実装のプレースホルダ。purge だけ動く） | internal | docs(ro), internal-data, audit-data, models(ro), gateway-sock |
| 社内 | qdrant | internal | qdrant-data |
| 社内 | vllm（profile laya） | internal | models(ro) |
| 社内 | judge-laya（profile laya・プレースホルダ）/ judge-clef（profile clef） | internal（別名 `judge`） | models(ro) |
| 社内 | frontend（プレースホルダ nginx） | internal, frontend-publish | なし |
| Gateway | gateway | egress のみ | gateway-data, gateway-sock |

## 起動

```
# v2/ で実行。API キーはファイルで渡す（既定 deploy/secrets/*。リポジトリに置かない）
ANTHROPIC_API_KEY_FILE=/安全な場所/anthropic_api_key \
OPENAI_API_KEY_FILE=/安全な場所/openai_api_key \
docker compose -f deploy/compose.yml --profile laya up -d     # または --profile clef
bash deploy/isolation_test.sh                                  # clef 構成なら COMPOSE_PROFILE=clef
```

モデルは `models` volume に事前取得しておく（revision 固定・取得用トークンはコンテナに渡さない）。各サービスは `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` で起動する。

## Gateway は単一ワーカーでしか起動しない

Gateway は起動時に送信中（ATTEMPTING）の行をすべて「結果不明」にするため、同じ DB を 2 つ以上のプロセスで使うと、
後から起動した方が他方の送信中の行を不明扱いにしてしまう。これを防ぐため、Gateway は起動時に DB と同じ場所の
ロックファイル（`/data/gateway.db.lock`）に排他ロック（`flock`）を取り、取れなければ
「別のプロセスが使用中」というメッセージで起動に失敗する。

- `uvicorn --workers 2` 以上、`docker compose up --scale gateway=2`、同じ `gateway-data` volume を別コンテナに付ける、のいずれも 2 つ目が起動しない。
- ロックはプロセスが終われば解放される。`/data/gateway.db.lock` は残ってよい（消さない）。
- `python -m gateway.purge` はロックを取らないので、Gateway の稼働中に実行してよい。
- `flock` はローカルの FS（Docker の local volume）前提。NFS 等の共有 FS に DB を置かない。
- ロックは uvicorn を起動する前に取る。uvicorn 0.54 は起動に失敗してもソケットのパスを消すため、以前は拒否された
  2 つ目が動いている 1 つ目の `/run/gateway/gateway.sock` を消し、社内から届かなくなっていた（10/26 コンテナで発見・修正）。
  修正後は `docker exec … python -m gateway.app` と `docker compose run --rm --no-deps gateway` の 2 つ目がどちらも
  終了コード 1 で止まり、1 つ目はソケット経由で応答し続けることを確認した。
- 起動は必ず `python -m gateway.app`（イメージの既定）で行う。`uvicorn` コマンドで同じソケットを指定して直接起動すると、
  拒否されたときに上と同じくソケットが消える。

## 監査 DB と定期 purge

- 監査 DB（`orchestrator.audit_store`）の場所は環境変数 `ORCHESTRATOR_AUDIT_DB`（絶対パス）で指定する。compose では internal-app に `/audit-data/audit.db`（専用 volume `audit-data`。internal-app だけがマウント）を設定済み。
- 既定パスは持たない。未設定・相対パスなら Orchestrator は起動時にエラーで止まる（勝手な場所に書くと purge や保全の対象から漏れるため）。volume を付け忘れて親ディレクトリが無い場合も、ディレクトリを作らずにエラーになる。
- 保持期限（メタデータ 90 日）の削除は、Orchestrator に常駐プロセスがまだ無いため、ホストの cron から 1 日 1 回 `python -m orchestrator.purge` を internal-app 内で実行する。定義例は `deploy/audit-purge.cron`:

```
17 3 * * * cd "$V2_DIR" && docker compose -f deploy/compose.yml exec -T internal-app python -m orchestrator.purge >> "$HOME/audit-purge.log" 2>&1
```

- `--db` を省略すると `ORCHESTRATOR_AUDIT_DB` を使う（アプリと同じ DB を必ず対象にするため、cron 側でパスを書かない）。`--dry-run` で削除せず件数だけを確認できる。
- 出力は件数だけの JSON（run_id・本文は出ない）。DB ファイルが無い・場所が未設定なら終了コード 1。
- WSL 内 Engine で動かす場合は WSL 側で cron サービスを起動しておく（WSL は既定で cron を起動しない）。Docker Desktop の場合は Windows のタスクスケジューラから同じコマンドを 1 日 1 回呼ぶ。
- internal-app は本体未実装のプレースホルダだが、イメージ（`deploy/internal-app.Dockerfile`）に orchestrator パッケージを入れてあるので purge は動く。10/26 に OrbStack で `up -d --no-deps internal-app` → 上の cron と同じ `exec -T internal-app python -m orchestrator.purge` が終了コード 0・`{"runs": 0, "events": 0}` を返すことを確認した。依存パッケージは入れていないので、動くのは標準ライブラリだけで書いた audit_store と purge のみ。ホストの cron／タスクスケジューラへの登録は、実行環境が決まってから試す。常駐プロセスができたら、Gateway と同様にプロセス内で定期実行する方式へ移す。
- audit-data / internal-data の所有者は internal-app イメージで 10002:10001・0700 にしている。名前付き volume は空のときにイメージ内の同じパスの所有者を写すため。以前は python:3.11-slim をそのまま使っていて volume が root 所有 755 になり、internal-app が監査 DB を作れなかった（10/26 発見・修正）。中身の入った volume には写らないので、所有者を変えたいときは volume を作り直す。

## 保証すること（静的に検査済み: `tests/deploy/test_compose_static.py`）

- 社内サービスは `internal: true` のネットワークだけに参加し、公開ポートを持たない。公開は frontend の 127.0.0.1 のみ。
- gateway は egress ネットワークだけに参加し、社内 volume（docs / internal-data / models / qdrant-data）と Docker ソケットをマウントしない。
- Orchestrator↔Gateway の受け渡しは `gateway-sock` volume だけ（共有するのは internal-app と gateway の 2 つ）。
- gateway の DB は `gateway-data`（gateway 専用）。監査 DB は `audit-data`（internal-app 専用。`tests/orchestrator/test_audit_deploy.py` で検査）。API キーは compose secrets で gateway にだけ渡し、環境変数に置かない。
- 全サービス read_only / cap_drop ALL / no-new-privileges / 非 root ユーザ。
- Clef-flash 構成（profile clef）では vllm を起動しない。
- gateway イメージは v2/common と v2/gateway だけを含む。internal-app イメージは v2/common と v2/orchestrator だけを含み、/run/gateway を作らない（gateway-sock の所有者は gateway イメージで決める）。
- 書き込む volume（:ro でない物。internal-app の gateway-sock を除く）は、イメージ側でサービスの user に chown してある。

静的検査は「設定がそう書かれている」ことしか示さない。分離が成立したかは `isolation_test.sh` を実環境で走らせて判断する。

## isolation_test.sh が試すこと（実際に接続を試みる）

1. 社内→外部: internal-app から外部 HTTPS、IP 直指定 TCP、ホスト上に一時的に立てた中継役リスナ（host.docker.internal とデフォルト GW 経由）に届かない。
2. Gateway→社内: gateway で社内サービス名が解決できない、社内コンテナ IP に直接接続できない、社内 volume パスと docker.sock が存在しない。
3. Gateway→frontend→社内: gateway から公開 frontend ポートに host.docker.internal / デフォルト GW / 127.0.0.1 のいずれでも届かない。
4. 正常経路: internal-app から UDS 経由で gateway `/status/{id}` が応答する（404 も応答とみなす）。internal-app から qdrant に接続できる。

## 未試験・未確定

- **分離試験は標準の Docker Engine でだけ通した（10/26）。** Mac 上の docker:29-dind（Docker Engine 29.9.2・iptables）で、vllm と judge-clef を CPU の代役に差し替えて laya 29 件・clef 28 件すべて PASS。GPU 実機（本物の vllm・llama.cpp）では未実行。
- **OrbStack ではゾーン分離が成立しない（10/26）。** 同じ試験を Mac の OrbStack で直接走らせると 6 件 FAIL（gateway から qdrant・vllm・frontend の IP に直接届く、host.docker.internal 経由で公開 frontend に届く）。最小再現でも、普通の bridge から `--internal` ネットワークのコンテナに IP で届いた。OrbStack はネットワーク間を遮断しないため、OrbStack での結果は 1（社内→外部）・4（正常経路）・堅牢化の確認にしか使えない。試験側に OrbStack 用の例外は入れない。
- **Docker Desktop と WSL 内 Engine のどちらで動かすかは未決（10/12 に特定）。** 上の通り分離は実行環境の実装に依存する。決めた方式の実機で isolation_test.sh を通すまで「本番で分離済み」と書かない。
- frontend は公開のため非 internal の `frontend-publish` にも参加する。frontend が社内への中継や外部への出口にならないことは、本物の frontend（リバースプロキシ設定）ができてから試験を追加する。
- UDS の所有者・権限: gateway は uid 10001、internal-app は uid 10002・gid 10001 で共有グループ。10/26 の dind 試験で internal-app からソケットに接続でき、`/status/{id}` が 404 を返した（umask の調整は不要だった）。
- qdrant（unprivileged イメージ）は read_only / 非 root / cap_drop ALL で起動を確認（10/26。snapshots の tmpfs を uid 1000 所有にする必要があった）。vllm・llama.cpp は未確認（v1 では vllm の read_only を見送っていた）。必要な書込先は tmpfs で足りるかを GPU 実機で実測する。
- judge-laya・internal-app・frontend はプレースホルダ。judge-clef のモデルパス・GPU レイヤ数も仮置き。
- vllm の起動値（`--max-model-len 4096` / `--gpu-memory-utilization 0.6`）は 6GB 想定のまま。目標を 8〜12GB に移した後、採用モデルは決まっていない（候補はすべて不合格）。`--quantization awq` は既定モデル（compressed-tensors）で起動を止めるため 10/26 に外した（止まることは固定版 0.25.1 のソースで、外せば読み込めることは 10/19 の Colab・vLLM 0.32.0 で確認。0.25.1 での起動は未確認）。
- IPv6 経路、DNS 以外の名前解決（/etc/hosts の注入等）は試験対象外。
