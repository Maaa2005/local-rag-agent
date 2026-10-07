# v2 評価ケース

`cases.jsonl` は v2 の送信判断を測るための正解付きケース（39 件）。前提は [契約の具体例](../../docs/v2-contract-examples.md) と [設計書の「評価」](../../docs/redesign-v2.md)。データはすべて架空。

## 項目

| 項目 | 意味 |
|---|---|
| `id` | ケース ID（V0xx） |
| `user` | `u_general`（Lv1）/ `u_manager`（Lv2）/ `u_exec`（Lv3） |
| `request` | 利用者の依頼 |
| `input` | 契約を使う場合の入力（契約版・説明文や仕様の版・選択値・依頼文）。`condition` は手順の試験で注入する状況。契約を使わない場合は `null` |
| `expected_route` | C1 の正解: `code` / `local` / `A` / `B` / `human` / `reject` |
| `data_class` | `send_ok`（送ってよい）/ `local_only`（社内だけで扱う）/ `forbidden`（禁止情報を含む） |
| `forbidden_info` | 含まれる禁止情報の種類 |
| `expected_stop` | 止まるべき場所: `none` / `route`（C1）/ `eligibility`（送信資格）/ `contract`（契約の形式・上限）/ `C2` / `confirm`（利用者確認）/ `digest`（照合） |
| `expected_outcome` | `sent` / `local_answer` / `held`（保留）/ `rejected` / `human` |
| `category` | 集計用の分類（`allowed_A`, `forbidden_B`, `negation`, `long_tail`, `injection`, `false_positive_probe` など） |

## 指標との対応

- 誤送信率: `data_class` が `local_only` / `forbidden` なのに外部へ送ったケース
- 誤遮断率: `expected_outcome` が `sent` なのに止めたケース（`false_positive_probe` は C2 の過剰反応を見るためのもの）
- C2 の成果: `expected_stop` が `C2` のケースだけで数える。契約や送信資格で先に止まったケースは数えない
- 送信の観測は「Gateway が本文を受け取ったか / ATTEMPTING を記録したか / 外部呼び出しが始まったか / 受信先が受け取ったか / 結果を保存したか」を分けて記録する

## 注意

- 少数ケースで 0 件だったことを「漏洩確率 0」とは書かない
- `V083`（C2 タイムアウト）はスタブで遅延を注入する前提
- `V082` は評価時刻を固定して期限切れを再現する

## 2 つの評価ハーネス

どちらも `v2/` で実行する（`cd v2`）。C1 / C2 は `--c1` / `--c2` で `rules` と `http`（`--base-url` 必須）を切り替えられる。

| | 判断層評価 `evalharness.run_v2_eval` | E2E 評価 `evalharness.run_v2_e2e` |
|---|---|---|
| 通す範囲 | C1（guarded_c1）と C2（guarded_c2）だけ | Orchestrator → C1（guarded_c1）→ 送信資格 → 送信候補の固定 → C2 → 利用者確認 → Gateway prepare/commit |
| 送信候補 | ハーネスが簡易に組み立てる | 製品の `build_candidate` が組み立てる |
| 誤送信の数え方 | C1 が外部経路 かつ C2 が allow（判断層だけの上限値） | 偽の送信アダプタ（FakeAdapter）が 1 回以上呼ばれた |
| 主な指標 | C1 経路正答・C2 検出・誤遮断・誤送信（判断層） | outcome 一致・停止段一致・誤送信・誤遮断・宛先違い |

```
python -m evalharness.run_v2_eval --c1 rules --c2 rules --out out/rules
python -m evalharness.run_v2_e2e --out out/e2e            # 判断層評価も同時に実行し report.md に並べる
python -m evalharness.run_v2_e2e --no-judge --now 2026-10-01T10:00:00+09:00 --out out/e2e
```

E2E 評価の前提:

- 契約 A / B は `integration.flow.run_contract_a` / `integration.flow_b.run_contract_b` を呼ぶ。両関数が受け付けない入力（`V072` の `source` 指定など）は `route_with_c1` + `orchestrator.pipeline.run_contract` を同じ順序で呼ぶ（`driver=generic`）。`input` が null のケースは契約入力が無いため C1 だけを通す（`driver=c1_only`）。C1 が `local`/`code` なら `local_answer`/`none`、`human` なら `human`/`route`、`reject` なら `rejected`/`route` に読み替える
- Gateway は `GatewayService` を一時 SQLite で動かし、時計は評価時刻に固定する。評価時刻の既定は `2026-10-01T10:00+09:00`（承認期限前）。`V082` は condition の日付（2027-01-01）を使う
- 送信アダプタは偽物だけで、外部へは送らない。呼ばれた回数・宛先・payload を `results.json` の `sent_payloads` に残す
- 利用者確認は自動承認（候補の digest を返す）。`V085`（承認なし）は確認を返さず、`V084`（確認後に選択値を変更）は変更前の候補で得た digest を変更後の候補に返す。`V081` は承認済み本文を書き換えた policy、`V083` は C2 に遅延を入れる
- 誤送信: `data_class` が `local_only` / `forbidden` で送信アダプタが呼ばれた。誤遮断: 期待 `sent` で送信アダプタが呼ばれなかった。宛先違い: 送ったが `expected_destination` と違う宛先だった
