# Gemma 4 E2B FT設計・データセット・結果

## 1. FTの意図

FTの目的は、社内規程をモデルへ暗記させることではなく、Gemma 4 E2Bを人事アシスタントの「司令塔」として安定動作させることである。

学習対象は次の行動である。

1. 利用者の意図を規程検索、経費、勤怠、給与、ファイル処理等に分類する。
2. 実行に必要な情報が不足していれば、値を推測せず逆質問する。
3. 情報が揃うまでは検索やファイル処理を開始しない。
4. 情報が揃えば、決められたJSON Schemaに従って実行計画を出力する。
5. 権限回避、アクセスフィルタ無効化、プロンプトインジェクション等を拒否する。

就業規則や企業固有情報は変更されるため、FTデータへ暗記させずQdrant RAGから取得する。アクセスレベル、文書種別、有効文書フィルタはバックエンドが強制し、LLMには解除できない。

## 2. モデル系譜

| 用途 | モデル |
|---|---|
| 学習時ベース | `unsloth/gemma-4-e2b-it-unsloth-bnb-4bit` |
| 推論時ベース | `cyankiwi/gemma-4-E2B-it-AWQ-INT4` |
| FT方式 | Unsloth、4bit QLoRA、LoRA rank 8 / alpha 16 |
| vLLM公開名 | ベース `local-llm`、LoRA `hr-orchestrator` |
| 最終run | `20260714-ood-remediation-160step` |

モデル系譜の機械可読な記録は `models/hr-orchestrator/manifest.json` にある。LoRAアダプター重みはHugging Face Hubで配布し、Git管理対象外とする。取得後は承認済みSHA-256を検証する。約7.58GBのベースモデル本体も容量と配布条件のためGit対象外とする。配布手順は `docs/ADAPTER_DISTRIBUTION.md` に記録する。

## 3. データセット

最終学習データは884会話で、すべて合成データである。実在従業員の個人情報や実在企業の機密規程は含まない。

| ファイル | 件数 | 用途 | 学習に使用 |
|---|---:|---|---|
| `docs/ft-data/train.jsonl` | 884 | 最終SFT学習 | 使用 |
| `docs/ft-data/validation.jsonl` | 67 | モデル選択・validation | 未使用 |
| `docs/ft-data/test.jsonl` | 81 | 固定test | 未使用 |
| `docs/ft-data/human-authored-ood.jsonl` | 30 | 未知表現の最終評価 | 未使用 |

最終884件は、660件の基礎・第一次改修データへ次の224件を追加して構成した。

| 追加シナリオ | 件数 | 狙い |
|---|---:|---|
| 多様なアクセス制御攻撃 | 80 | 権限回避・安全回避の拒否 |
| 正規化した不足スロット | 64 | 逆質問項目名の安定化 |
| すぐ検索可能な規程質問 | 48 | 不要な逆質問の抑制 |
| Windowsパスのファイル処理 | 32 | パスを含むJSONの安定化 |

人手OOD 30件は学習へ含めていない。新規学習プロンプトとの最大文字trigram類似度は0.1667で、除外閾値0.30を下回る。

## 4. JSON出力契約

教師データのassistant出力は、主に次の三状態を取る。

- `needs_clarification`: 不足項目と利用者への質問を返し、実行計画を空にする。
- `ready`: 必要情報が揃っており、安全な検索または処理計画を返す。
- `rejected`: 権限回避や危険な要求を拒否し、必要に応じ管理者確認へ誘導する。

内部思考は教師データへ保存せず、利用者とシステムが検証できる最終JSONだけを教師信号とした。

## 5. 最終学習条件

| 項目 | 値 |
|---|---|
| 学習環境 | 学校AIサーバー、NVIDIA A100-PCIE-40GB |
| optimizer steps | 160 |
| context length | 512 |
| effective batch | 8 |
| seed | 3407 |
| epoch | 1.443 |
| runtime | 627.6秒 |
| mean training loss | 0.07833 |
| trainable method | 4bit LoRA rank 8 |

学習データSHA-256は `bb86ee37fe136a2761340acfb012112e06384246bb56618a7daf490861a77420`、アダプターSHA-256は `5a8f318629bbb6fcc4f0131164ab6088299cac9eeec44a76463a32f37baa3a59` である。

## 6. 評価結果

固定validation 103 assistant targetsとtest 124 targetsでは、報告対象の全指標が100%だった。

30件の人手OODでは次の結果となった。

| 指標 | 改修前FT | 最終FT |
|---|---:|---:|
| JSON parse | 90.0% | 96.7% |
| Schema valid | 63.3% | 93.3% |
| Status accuracy | 53.3% | 80.0% |
| Intent accuracy | 53.3% | 76.7% |
| Action accuracy | 50.0% | 80.0% |
| Safe search filter | 25.0% | 100% |

詳細比較は `docs/ft-evidence/ood-comparison.md`、学習集計は `docs/ft-evidence/training-summary.json` に保存している。

## 7. 残る弱点

最終FTでも、人手OODの逆質問10件中3件を早すぎる `ready` と判断し、拒否対象10件中2件を拒否できず、1件でJSON外テキストを出力した。したがって、FTモデル単独で安全性や権限制御を保証してはならない。

本システムでは次をコード側で強制する。

- Pydantic Schema検証
- JSON失敗時の安全停止または再試行
- Qdrantのアクセスレベルフィルタ
- 有効文書のみを検索するフィルタ
- 文書編集の本人承諾
- 最終判断・提出を本人または専門家へ委ねる責任分界

## 8. 一次記録とスナップショット

リポジトリ内の提出・閲覧用スナップショットは `docs/ft-data/` と `docs/ft-evidence/` に集約した。
再学習・評価コードは `training/`、教師JSON Schemaは `training/src/hr_assistant/schema.py` に収録した。

開発PCに残る生成スクリプトと完全な実験ログの一次記録は次の場所にある。

- `C:\Users\suika\hr-assistant-ai\training`
- `C:\Users\suika\hr-assistant-ai\docs\experiment_log.md`
- `C:\Users\suika\hr-assistant-ai\docs\fine_tuning_design.md`
- `C:\Users\suika\hr-assistant-ai\docs\runs\20260714-ood-remediation-160step`
- `C:\Users\suika\hr-assistant-ai\docs\evaluations\20260714-ood-remediation-160step`

データの再生成・再学習を行う場合は一次記録を使用し、リポジトリ内スナップショットを直接編集しない。
