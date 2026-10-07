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
| 社内 | internal-app（Orchestrator・本体未実装のプレースホルダ） | internal | docs(ro), internal-data, models(ro), gateway-sock |
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

## 保証すること（静的に検査済み: `tests/deploy/test_compose_static.py`）

- 社内サービスは `internal: true` のネットワークだけに参加し、公開ポートを持たない。公開は frontend の 127.0.0.1 のみ。
- gateway は egress ネットワークだけに参加し、社内 volume（docs / internal-data / models / qdrant-data）と Docker ソケットをマウントしない。
- Orchestrator↔Gateway の受け渡しは `gateway-sock` volume だけ（共有するのは internal-app と gateway の 2 つ）。
- gateway の DB は `gateway-data`（gateway 専用）。API キーは compose secrets で gateway にだけ渡し、環境変数に置かない。
- 全サービス read_only / cap_drop ALL / no-new-privileges / 非 root ユーザ。
- Clef-flash 構成（profile clef）では vllm を起動しない。
- gateway イメージは v2/common と v2/gateway だけを含む。

静的検査は「設定がそう書かれている」ことしか示さない。分離が成立したかは `isolation_test.sh` を実環境で走らせて判断する。

## isolation_test.sh が試すこと（実際に接続を試みる）

1. 社内→外部: internal-app から外部 HTTPS、IP 直指定 TCP、ホスト上に一時的に立てた中継役リスナ（host.docker.internal とデフォルト GW 経由）に届かない。
2. Gateway→社内: gateway で社内サービス名が解決できない、社内コンテナ IP に直接接続できない、社内 volume パスと docker.sock が存在しない。
3. Gateway→frontend→社内: gateway から公開 frontend ポートに host.docker.internal / デフォルト GW / 127.0.0.1 のいずれでも届かない。
4. 正常経路: internal-app から UDS 経由で gateway `/status/{id}` が応答する（404 も応答とみなす）。internal-app から qdrant に接続できる。

## 未試験・未確定

- **分離試験は未実行**（Mac には GPU も起動済み compose もない）。`docker compose config` と静的テストのみ確認済み。
- **Docker Desktop と WSL 内 Engine のどちらで試すかは未決（10/12 に特定）。** host.docker.internal や 127.0.0.1 公開ポートへのコンテナからの到達性は両者で挙動が異なりうるため、決めた方式で isolation_test.sh を通すまで「分離済み」と書かない。
- frontend は公開のため非 internal の `frontend-publish` にも参加する。frontend が社内への中継や外部への出口にならないことは、本物の frontend（リバースプロキシ設定）ができてから試験を追加する。
- UDS の所有者・権限: gateway は uid 10001、internal-app は uid 10002・gid 10001 で共有グループにしているが、ソケットファイルのモード（uvicorn の umask 依存）で internal-app が接続できるかは未確認。接続できなければ起動ラッパーで umask を設定する。
- qdrant（unprivileged イメージ）・vllm・llama.cpp を read_only / 非 root / cap_drop ALL で起動できるかは未確認（v1 では vllm の read_only を見送っていた）。必要な書込先は tmpfs で足りるかを実測する。
- judge-laya・internal-app・frontend はプレースホルダ。judge-clef のモデルパス・GPU レイヤ数も仮置き。
- IPv6 経路、DNS 以外の名前解決（/etc/hosts の注入等）は試験対象外。
