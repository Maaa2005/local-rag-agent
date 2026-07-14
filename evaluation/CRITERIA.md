# RAG エージェント評価軸定義書

本書は、社内 RAG エージェント (Maaa2005/local-rag-agent) の品質を多面的に
評価するためのメトリクスと合格基準を定義する。
要件定義書§5「評価基準（フェーズ1 リリース判定）」を出発点に、検索品質・
生成品質・運用安全性・性能の 4 カテゴリへ展開した。

すべてのメトリクスは `evaluation/run_eval.py` が `dataset.jsonl` を用いて
自動算出する。スコア閾値を満たさない場合は不合格とし、原因を `report.md`
に記録する。

---

## 1. メトリクス一覧

| # | カテゴリ | メトリクス | 計算方法 | 範囲 | 合格基準 | 出典 |
|---|---|---|---|---|---|---|
| M1 | 検索 | **Recall@5** | 期待ソースが取得チャンク上位 5 件に含まれた質問の割合 | 0–1 | ≥ **0.80** | 標準 IR |
| M2 | 検索 | **MRR** | 期待ソースの最高順位の逆数の平均 | 0–1 | ≥ **0.60** | 標準 IR |
| M3 | 検索 | **Context Relevance** | LLM-judge: 取得チャンクが質問にどれだけ関連するか (0/0.5/1 平均) | 0–1 | ≥ **0.70** | RAGAS |
| M4 | 生成 | **Faithfulness** | LLM-judge: 回答中の各主張が取得チャンク内に根拠を持つ比率 | 0–1 | ≥ **0.90** | RAGAS |
| M5 | 生成 | **Answer Relevance** | LLM-judge: 回答が質問に直接答えているか (0/0.5/1) | 0–1 | ≥ **0.80** | RAGAS |
| M6 | 生成 | **Correctness** | 期待キーワードを正しく言及できているか (一致比率) | 0–1 | ≥ **0.80** | 要件§5 |
| M7 | 運用 | **権限フィルタ正答率** | access_level 超過チャンクが応答に混入しないケース率 | 0–1 | = **1.00** | 要件§5 (Must) |
| M8 | 運用 | **適切な拒否率** | `should_refuse=true` のとき「情報なし」と返した率 | 0–1 | ≥ **0.95** | 要件§4.2 |
| P1 | 性能 | **First Token Latency p50** | 質問送信〜最初のトークン受信までの中央値 | 秒 | ≤ **5.0s** | 要件§4.3 |
| P2 | 性能 | **First Token Latency p95** | 同 95%タイル | 秒 | ≤ **12.0s** | 拡張 |
| P3 | 性能 | **Total Latency p50** | 質問送信〜完了までの中央値 | 秒 | ≤ **15.0s** | 拡張 |

---

## 2. メトリクス詳細

### M1. Recall@5
- 期待ソース集合 `E` と、取得チャンクの上位 5 件のソースファイル集合 `R5` について `|E ∩ R5| > 0` なら 1、それ以外は 0。各質問で算出し平均。
- 妥当な検索ができていないとどんなに良い LLM でも答えられない、最も基本的な指標。

### M2. MRR (Mean Reciprocal Rank)
- 各質問について、期待ソースが取得チャンクのうち最初に現れた順位 `rank ∈ {1,...,k}` の逆数 `1/rank` を計算。期待ソースが上位 k に無ければ 0。平均をとる。
- "正解を一番上に出せているか" を測る。

### M3. Context Relevance
- ジャッジ LLM に `question` と `retrieved chunks` を渡し、各チャンクが質問に関連しているかを 0/0.5/1 で評価。平均をとる。
- "ノイズが混ざっていないか" を測る。

### M4. Faithfulness (忠実性 / Groundedness)
- ジャッジ LLM が回答を 1 文ずつ抽出し、各文が取得チャンク内のテキストから推論可能かを判定。`(根拠のある文数) / (全文数)`。
- ハルシネーション検出の中核指標。要件§4.2「RAG 経路ハルシネーション抑止」に対応。

### M5. Answer Relevance
- ジャッジ LLM に質問と回答だけを渡し、質問への応答度を 0/0.5/1 で評価。
- "質問にきちんと答えている回答か" を測る。M4 と独立。

### M6. Correctness
- データセットの `expected_answer_keywords` が回答中に含まれている割合。
- 形式的な評価で安価。LLM-judge と相互チェックに使う。

### M7. 権限フィルタ正答率 (CRITICAL)
- `user_access_level` を持つ質問に対し、取得チャンクの `access_level` がすべて `user_access_level` 以下であること。
- 1 件でも違反があれば該当質問は 0、それ以外は 1。要件§5 で 100% 必須なので不合格時はリリース不可。

### M8. 適切な拒否率
- `should_refuse=true` の質問について、回答が「情報なし」「資料には含まれていません」等のテンプレを含むかを判定。
- 知らないことを正直に「知らない」と返せるか。

### P1–P3. 性能
- HTTP `/api/chat` を SSE で叩き、`type=token` の最初のイベントまでの時間 (P1, P2) と `type=done` までの時間 (P3) を計測。
- p50 と p95 をプロバイダごとに算出。

---

## 3. 評価対象と rerank A/B 比較

本アプリは社内機密 RAG 専用であり、LLM プロバイダはローカル vLLM 1 つに
**固定**されている（`backend/app/services/providers/` には `vllm_provider`
のみが登録されている）。外部送信を伴うフロンティアプロバイダ
（Anthropic / OpenAI / Gemini / Bedrock 等）は社内文書の流出経路となるため
本アプリのコードベースには存在せず、`evaluation/run_eval.py` からも選択
できない（別アプリ `frontier-llm-agent` に分離済み）。

そのため「プロバイダ間比較」は行わず、代わりに Cross-Encoder リランクの
on/off を同一データセットで横並びに評価する (`--rerank both`) 。
`report.md` には以下の形式で結果を並記する。

| 実行 | Recall@5 | MRR | Faithfulness | Answer Rel. | Correctness | Permission | Refusal | Latency p50 | 合否 |
|---|---|---|---|---|---|---|---|---|---|
| vllm+rerank_on | … | … | … | … | … | … | … | …s | ✓/✗ |
| vllm+rerank_off | … | … | … | … | … | … | … | …s | ✓/✗ |

合否は「全メトリクスが合格基準を満たすか」で判定する。

---

## 4. データセット要件

`evaluation/dataset.jsonl` は JSONL 形式で 1 行 1 質問。スキーマ:

| フィールド | 型 | 説明 |
|---|---|---|
| `id` | string | 質問 ID (連番) |
| `question` | string | ユーザーが投げる質問 |
| `expected_answer_keywords` | string[] | 正解回答に含まれるべきキーワード |
| `expected_sources` | string[] | 検索で当たるべき文書パスのうち少なくとも 1 つ |
| `user_access_level` | int | 1\|2\|3 (一般\|管理職\|役員) |
| `category` | string | "general" / "manager" / "executive" / "permission_test" / "hallucination_test" |
| `should_refuse` | bool | true なら "情報なし" 系の応答を期待 |

データセットは以下のバランスで用意する:
- 一般文書質問: 40%
- 管理職向け質問: 20%
- 役員向け質問: 20%
- 権限テスト (低レベルユーザーで高レベル文書を聞く): 10%
- ハルシネーションテスト (DB に無い質問): 10%

---

## 5. ジャッジ LLM について

- `evaluation/judges.py` は `app.services.providers` レジストリからプロバイダを
  取得する実装になっているが、レジストリには `vllm` しか登録されていないため、
  実際に使えるジャッジは **ローカル vLLM のみ**である。
- したがって「評価対象プロバイダとは別のジャッジで自己評価バイアスを避ける」
  という一般的な RAGAS 流の理想は、このアプリの設計（社内文書を外部へ一切
  送信しない）とは両立しない。ジャッジも自己評価バイアスを含むローカル LLM
  であることを前提にスコアを解釈すること（特に M3〜M5 は参考値と位置づけ、
  M1・M2・M6・M7・M8 の機械的メトリクスを一次判断材料とする）。
- 外部 LLM（Anthropic 等）をジャッジに使う手順は実装上存在しない。将来的に
  実データではなく本リポ同梱の架空コーパス（`evaluation/corpus/`、機密なし）
  のみを対象に外部ジャッジを試す場合でも、実文書を扱う本番運用では絶対に
  使用しないこと。

---

## 6. 実行手順

```bash
cd backend

# 現在の settings (rerank 含む) で評価
.venv/bin/python ../evaluation/run_eval.py --provider vllm

# リランク on/off を A/B 比較
.venv/bin/python ../evaluation/run_eval.py --provider vllm --rerank both

# 性能・機械的メトリクスのみ (LLM-judge をスキップ)
.venv/bin/python ../evaluation/run_eval.py --provider vllm --no-judge
```

`--provider` には `vllm` 以外指定できない（他プロバイダはレジストリ未登録
のため `Unknown provider` で失敗する）。`--all` も同様に `vllm` のみが
評価対象になる。

---

## 7. リリース判定 (フェーズ 1)

ローカル vLLM で以下を全て満たしたときフェーズ 1 リリース可:

1. **M7 = 1.00** (権限フィルタ違反ゼロ) — 必須
2. **M4 ≥ 0.90** (Faithfulness) — 必須
3. **M6 ≥ 0.80** (Correctness)
4. **P1 ≤ 5.0s** (First Token Latency p50)

外部プロバイダとの比較は行わない（本アプリはローカル LLM 固定のため）。
rerank on/off の差分は `report.md` に可視化し、運用時のデフォルト設定
（`settings.rerank_enabled`）を決める判断材料とする。
