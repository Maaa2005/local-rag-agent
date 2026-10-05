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
