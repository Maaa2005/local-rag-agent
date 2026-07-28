# FTデータセット・スナップショット

このディレクトリは、最終モデル `20260714-ood-remediation-160step` の提出・監査用データセットスナップショットである。

| ファイル | 件数 | SHA-256 | 役割 |
|---|---:|---|---|
| `train.jsonl` | 884 | `bb86ee37fe136a2761340acfb012112e06384246bb56618a7daf490861a77420` | 学習入力 |
| `validation.jsonl` | 67 | `6fc7b3ce486722feea07c2e6c4658bd5596f882abef9732b1f262ae1dbe59f34` | 固定validation |
| `test.jsonl` | 81 | `20916738d1d7a2ff4defbb1517c866f455e45958e91282612afac5757428ae08` | 固定test |
| `human-authored-ood.jsonl` | 30 | `7f79ff992b51c287e6632313d3545e7ae223b9d40e38a03021f1bbca5c037800` | 人手OOD評価 |
| `manifest.json` | - | ファイル内参照 | 構成、目的、分離条件 |

`validation.jsonl`、`test.jsonl`、`human-authored-ood.jsonl` は学習入力へ含めていない。特に人手OODは、未知表現に対する汎化を評価する固定データであり、今後も学習データへ混ぜない。

データはJSONL形式で、一行が一会話である。system、user、assistantの標準チャットメッセージと、評価・分類用metadataを含む。実在従業員の個人情報や実在企業の機密規程は含まない。

一次生成物は `C:\Users\suika\hr-assistant-ai\training` にあり、このディレクトリはGitHub・授業提出用の固定コピーである。
