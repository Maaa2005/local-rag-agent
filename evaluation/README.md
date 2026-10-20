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

repo ルートから `-m` で呼ぶ。`run_eval.py` を直接呼ぶと、スクリプトのある `evaluation/` が import の先頭に入り、
`evaluation/types.py` が標準ライブラリの `types` を隠して起動しない（`cannot import name 'GenericAlias' from 'types'`）。

```bash
backend/.venv/bin/python -m evaluation.run_eval            # 現在の settings で評価
backend/.venv/bin/python -m evaluation.run_eval --rerank both   # リランク on/off を A/B 比較
backend/.venv/bin/python -m evaluation.run_eval --rerank off --no-judge --dataset path/to/other.jsonl
# 50 問 + 数値の境目 12 問から一部だけ流し、結果の置き場所と実行条件を決める
backend/.venv/bin/python -m evaluation.run_eval --provider vllm --no-judge --rerank off \
  --dataset evaluation/dataset.jsonl --dataset evaluation/v2/numeric_boundary.jsonl \
  --ids Q010,Q039,N02 --out-dir /tmp/run1 --meta gpu=T4 --meta gpu_util=0.50
```

- `--rerank on|off|both`: Cross-Encoder リランクの有無を切り替え。`both` は 2 回評価しレポートに並記。未指定なら `settings.rerank_enabled` を尊重
- `--no-judge`: LLM-as-a-Judge をスキップ（検索メトリクスのみ）
- `--judge-from <raw.jsonl>`: `--no-judge` で保存した回答を、`--judge` の採点役だけで採点する。候補を比べるときは候補ごとに `--no-judge` で回答を保存し、採点役のモデルを vLLM に載せ直してから全候補をこれで採点する（`--judge` だけだと候補が自分の答えを採点する）
- `--dataset <jsonl>`: 問のファイル。繰り返すと順につなげる（ファイルをまたいで同じ ID があれば止める）
- `--ids Q010,N02`: 流す問の ID をカンマ区切りで。無い ID は止める
- `--out-dir <dir>`: 結果の置き場所（既定 `evaluation/results/<UTC ts>`）。空でなければ止める
- `--meta KEY=VALUE`: 実行条件（GPU・GPU_UTIL・モデルの revision など）を `run_meta.json` の extras に残す。繰り返し可
- 評価基準・合格ラインは `CRITERIA.md` を参照

### 出力

`--out-dir` に次の 3 つを書く。

- `run_meta.json`: 実行条件。最初の問を流す前に書くので、途中で落ちても残る（argv・データセットの sha256・`--ids`・settings・git のコミット・vLLM の `/v1/models`・`--meta` の extras・`index`＝索引の状態）。`--judge-from` は元の `run_meta.json` を `source_run_meta` に写す
- `raw.jsonl`: 1 問 1 行の回答・検索結果・`prompt_tokens`（vLLM の usage）
- `report.md`: 集計

## Colab での試運転（Step 0.5・`docs/v2-light-model-plan.md` §4）

合否には使わない。50 問の本番の測定の前に、流れが Colab 無料 T4 で最後まで通るかと、決める値（max-model-len・8GB/12GB の GPU_UTIL）を出す。

- `colab/step05_trial.ipynb`: 「Run all」で全部流れるノート。索引 → 入力トークン数の測定（`LLM_MAX_TOKENS=1`）→ 8GB・12GB の予算ごとに起動して 8 問 + 入力が最長の問 → 採点役を別に起動して事前確認と `--judge-from` → `/content/step05/summary.json`。計画 6（Gemma 4 E2B を llama-server で起動できるか）はやらない。採点役は流れを確かめる代役（既定は候補と同じモデル・`judge_role=flow-check-only`）
- `colab/corpus_index.py`: `evaluation/corpus` を本番と同じ取り込みで索引にする（`python evaluation/colab/corpus_index.py --repo . --work W --qdrant-url http://127.0.0.1:6333 --collection C`）。既にある app.db・空でないコレクションは拒否。`W/index.json` の `env` を評価側の環境変数にする
- `colab/step05.py`: ノートの部品（標準ライブラリだけ）。予算から GPU_UTIL（`(予算 − 常駐) ÷ カード全体`）、max-model-len（`ceil((入力の最大 + 1024) / 512) × 512`）、GPU の記録の切り出し、vLLM のログと `/metrics` の確認、試運転の確かめ（取れなかった項目は None＝未確認で、True にしない）
- `judge_precheck.py`: 採点役の事前確認。正解の分かっている 5 例（`v2/judge_precheck.jsonl`）で、本文どおりの答えが誤答より、答えた方が拒否より高い点になるかを見る。`python -m evaluation.judge_precheck --out <json>`。終了コード 0=pass / 1=fail / 3=未判定（JSON が取れない・値が欠ける採点が 1 つでもある）
