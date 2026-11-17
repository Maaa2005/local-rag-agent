# 必須の事実による採点と、揺れ幅つきの比較

RAG の答えを「規程の必須の事実を何個述べたか」で採点し、2 つの設定の差が揺れの中かを見る道具。
M6（キーワード）は変えず、並べて見る追加の物差し（`CRITERIA.md` の M6n）。
標準ライブラリだけで動く。判定役は OpenAI 互換の窓口（既定 `http://127.0.0.1:8080`、温度 0・seed 固定）。

## 手順

```bash
# 1. 下書き → 人が直す → 検査して固定（回答を見る前に）
python -m evaluation.nugget.cli draft  --questions q.jsonl --corpus set/corpus --out set/nuggets.jsonl --model <名前>
python -m evaluation.nugget.cli freeze --nuggets set/nuggets.jsonl        # set/freeze.json に sha256・時刻・版

# 2. 答え（JSONL {"id"|"qid","text"|"answer"}・1 ファイル = 1 回分）を判定
python -m evaluation.nugget.cli judge --nuggets set/nuggets.jsonl --answers runs/a1.jsonl runs/a2.jsonl --out-dir judged --model <判定役>

# 3. 点・比較・レポート（--a/--b/--aa は判定ファイルを K 個ずつ）
python -m evaluation.nugget.cli score   --nuggets set/nuggets.jsonl --judged judged/a1.judged.jsonl judged/a2.judged.jsonl
python -m evaluation.nugget.cli compare --nuggets set/nuggets.jsonl --a judged/a*.jsonl --b judged/b*.jsonl --aa judged/aa*.jsonl --cluster rule
python -m evaluation.nugget.cli report  --nuggets set/nuggets.jsonl --a ... --b ... --aa ... --top 10 --out report.md
```

- `nuggets.jsonl`: 1 行 1 問。`importance` は vital/okay、vital は 1 つ以上、`quote` は規程本文（`<set>/corpus/<rule_id>.md`）に逐語で含まれる（改行・空白の違いは無視）。質問文は行の `question`（draft が入れる）か `judge --questions`。
- freeze と合わない `nuggets.jsonl` では judge / score / compare / report は止まる。同じ版の中身は変えられない。直したら `version` を上げて freeze し直す（前の版の判定ファイルは sha256 が合わず読めない）。
- 判定の JSONL には判定の版（`NJ-3`、`--judge-version` で `NJ-2` も選べる）・判定役のモデル名・seed・nuggets の sha256 が入る。版やモデルが混ざった比較は止める。
- A/A（`--aa`）は **設定 A を seed だけ替えて流し直した K 回分**。A とその差で揺れ幅 w を出す。温度 0・seed 固定のままだと w がほぼ 0 になり揺れを小さく見誤るので、本番と同じ設定で seed を替える。

## 判定（2 回の呼び出し）

1. 答えを文に分け（`split_sentences` + 改行でも分ける）、文ごとに 必須 / 補足 / 定型 / 不要 / 記載なし / 相反（内部名 vital / okay / boilerplate / irrelevant / not_stated / contradiction）。
2. vital の事実ごとに full / partial / none と、回答からそのまま写した根拠（`evidence`）。数値・条件・境目が回答に無ければ full にしない。
- 機械の検査（NJ-2）: full / partial なのに `evidence` が空、または回答に（空白・改行・全角空白を除いて）部分一致しない事実は none に下げる。下げた事実の nid は記録の `downgraded` に残る（`evidence` も残る）。none は `evidence` が無くてよい。
- NJ-3（既定。`judge --judge-version {NJ-2,NJ-3}` で選ぶ。記録の `judge_version` に選んだ版が入る）: 呼び出し 2 は回答を番号付きの文で渡し、事実ごとに `{"nid", "score", "sentences": [番号]}` を返させる。full / partial なのに番号が空、または 1..文の数 の範囲外だけなら none に下げ、nid を `downgraded` に残す。`evidence` には引いた文の本文をつないだものが入る（人が読む用）。8B の判定役は答えでなく事実や規程の文を写してしまい NJ-2 で答えにある事実まで下がったため、写しでなく番号にした。
- 呼び出し 1 は「結論: …」の行を定型にしない（規程に合えば 必須、食い違えば 相反）。
- JSON が読めない・長さが合わない・知らないラベルは 1 回だけ頼み直す。それでも駄目な回は「判定できず」。点にも数にも入れない（0 点にしない）。

## 式

- V_i = (vital の事実の点の合計) / (vital の数)、full=1・partial=0.5・none=0。V_i^strict は full だけ 1。
- C_i = 相反の文が 1 つ以上あれば 1。集計は「相反のある答えの割合」。V とは足さない。
- K 回: s_i = 判定できた回の V_i の平均（問ごとに平均してから集計。K 回分を混ぜて SE を出さない）。K 回すべて判定できない問は外し、数を出す。
- M6 の並記: vital の事実の `keywords` を合わせた一覧に対する `correctness_keyword_score`（keywords が無い問は `-`）。
- 差 d_i = s_B,i − s_A,i（両方にある問だけ）、d̄ = 平均。
- SE_CLT = sqrt(Var(d, ddof=1) / n)
- SE_clustered² = SE_CLT² + (1/n²) Σ_c Σ_i Σ_{j≠i} (d_ic − d̄)(d_jc − d̄)（Error Bars 論文 式 4）。まとまり c = 規程（`--cluster rule`、主はこちら）。`--cluster none` なら CLT が主。
- 95% 区間 = d̄ ± 1.96·SE。
- w = A/A の区間の端の絶対値の大きい方。
- 信じる差 = B−A の区間が 0 を含まず、かつ |d̄| > w。それ以外は「差は揺れの中」。A/A が無い時は「判定保留」。
- n<2: 平均だけ出し、SE・区間は出さず「判定不能」。値がすべて同じ: SE=0（区間は 1 点）。

## 試験

`python -m pytest -q evaluation/tests/test_nugget_*.py`（作り物の規程・偽の判定役。ネット・GPU・モデル不要）。
