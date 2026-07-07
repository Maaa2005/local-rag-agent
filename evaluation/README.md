# RAG 評価ハーネス

自作 RAG パイプライン（retrieve → LLM 生成）を 50 問のデータセットで定量評価する。
評価コーパス（架空企業の社内文書）を `corpus/` に同梱しており、どのマシンでも再現できる。

## コーパスの配置（GPU 実行機での準備）

`corpus/` 配下を監視フォルダへコピーし、管理画面で access_level を設定する:

| コピー元 | コピー先 | access_level |
|---|---|---|
| `evaluation/corpus/general/` | `/watched/general/` | 1 |
| `evaluation/corpus/manager/` | `/watched/manager/` | 2 |
| `evaluation/corpus/executive/` | `/watched/executive/` | 3 |

配置後、watcher の自動インデックス完了（documents が全件 done）を待ってから評価を実行する。

## データセット構成（dataset.jsonl・50問）

| category | 問数 | 内容 |
|---|---|---|
| general | 20 | Lv1 文書への通常 QA（言い換え・複数文書横断・数値問い含む） |
| manager | 8 | Lv2 文書への QA（Lv2 以上のユーザー） |
| executive | 6 | Lv3 文書への QA（Lv3 ユーザー） |
| permission_test | 8 | 低権限ユーザーが上位文書の内容を質問 → 拒否が正解（should_refuse=true） |
| hallucination_test | 8 | コーパスに存在しない事項を質問 → 「見つからない」が正解（should_refuse=true） |

キーワード・出典はすべて `corpus/` の本文に接地している。整合性は
`tests/test_dataset_corpus.py` が担保する（出典実在・キーワード包含・ID 重複なし等）。

## 実行

```bash
cd backend
.venv/bin/python ../evaluation/run_eval.py            # 現在の settings で評価
.venv/bin/python ../evaluation/run_eval.py --rerank both   # リランク on/off を A/B 比較
.venv/bin/python ../evaluation/run_eval.py --rerank off --no-judge --dataset path/to/other.jsonl
```

- `--rerank on|off|both`: Cross-Encoder リランクの有無を切り替え。`both` は 2 回評価しレポートに並記。未指定なら `settings.rerank_enabled` を尊重
- `--no-judge`: LLM-as-a-Judge をスキップ（検索メトリクスのみ）
- 評価基準・合格ラインは `CRITERIA.md` を参照
