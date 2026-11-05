# Issue #14 E4: 検索の時間切れと送り直し（線は測る前に固定）

11/2 の 6GB 実機の 2 回目は、本体のメモリが尽きた時（Windows の空き 49MB）に Q011 の検索が qdrant-client の既定 5 秒で時間切れになった（Issue #1・PR #12）。調査（`docs/research/2026-11-02/low-memory.md` §2・§4 の 3）は「時間切れを 10 秒にし、1 回だけ送り直す」を試す案にしていた。原因の本体はページ読み（majflt）の側で、これは症状への対処。

## 何を変えたか（既定は今までと同じ）

| 設定（環境変数） | 既定 | 意味 |
|---|---|---|
| `QDRANT_TIMEOUT_S` | 0 | 0 は qdrant-client の既定のまま（1.12.1 は timeout を渡さないと httpx の既定 5 秒）。整数の秒 |
| `QDRANT_SEARCH_RETRIES` | 0 | 読むだけの要求（`query_points`・`scroll`）を、時間切れの時だけ送り直す回数（0〜2） |

- 送り直すのは時間切れ（`httpx.TimeoutException` と、それを包んだ `ResponseHandlingException`）だけ。接続拒否・4xx/5xx・応答の誤り・外側の打ち切り（v2 の `rag_backend` の 30 秒）は送り直さない
- 書き込み（upsert・delete・set_payload）は送り直さない
- 送り直しでも権限の絞り込み（`access_level` の filter）は同じ要求のまま。その後の SQLite の確かめ直しも変わらない

## 腕と条件

- 腕: Q0 = 既定（5 秒・送り直しなし）、Q1 = `QDRANT_TIMEOUT_S=10`・`QDRANT_SEARCH_RETRIES=1`
- 絞り方: S1 = Windows の空き 1,024 MiB（有効: 検索の間の空きの中央値が ± 256 MiB に入り、全部放し 0 回）、S2 = Windows の空き 300 MiB（有効: ± 128 MiB・全部放し 0 回・`--floor-mib 128`）。絞る道具は E3 の `evaluation/gpu_local/mem_pressure.py`（E3 の PR を取り込んだ後に流す）
- 1 回 = 50 問 + N 12 問を順に繰り返して 300 回の検索（`retrieve()`・rerank off・e5-large は CPU）。最初の 1 回は読み込みを含む（含めたまま数える）
- 有効の条件（S1・S2 共通で追加）: 測定の始めから終わりまで絞る道具の記録の抜けが 10 秒以内で、測り終えた時に道具が生きている。記録の間隔の既定 1 秒の 10 倍として決めた値
- 順番: Q0-S1 → Q1-S1 → Q0-S2 → Q1-S2。有効でない回は同じ設定で 1 回だけ流し直してよい

## 線（Q1 の S1・S2 それぞれ。全部満たして「採る」・`search_bench.e4_line_checks`）

| # | 線 |
|---|---|
| 1 | 300 回とも検索が返る（時間切れの失敗 0・ほかの失敗 0） |
| 2 | 検索の時間 p95 ≤ 1.0 秒（失敗した回は無限大として数える） |
| 3 | 検索の時間 p99 ≤ 3.0 秒 |

記録だけ: Q0 の同じ数、送り直した回数、最大、最初の 1 回、読み込みを除いた p95・p99、空きの最小・中央値。

Q0 が時間切れで落ちて Q1 が通れば「送り直しで失敗を消せる」、Q1 の p99 が 3 秒を超えれば「失敗は消えても遅れが残る（原因の側を直す・E3）」と書く。どれでも線は動かさない。

## 流し方（学校 PC・WSL2・GPU は使わない）

`README.md` の「準備」のうち Qdrant と `.work/rag-env` だけでよい（llama.cpp と GGUF は要らない）。ほかのアプリは閉じる。

```bash
P=".work/rag-env/bin/python -m evaluation.gpu_local.mem_pressure --probe windows --log {log}"
.work/rag-env/bin/python -m evaluation.gpu_local.search_bench --work .work --out .work/out-e4-q0-s1 \
  --n 300 --pressure-target-mib 1024 --pressure-cmd "$P --target-mib 1024"
.work/rag-env/bin/python -m evaluation.gpu_local.search_bench --work .work --out .work/out-e4-q1-s1 \
  --n 300 --qdrant-timeout-s 10 --qdrant-retries 1 --pressure-target-mib 1024 --pressure-cmd "$P --target-mib 1024"
.work/rag-env/bin/python -m evaluation.gpu_local.search_bench --work .work --out .work/out-e4-q0-s2 \
  --n 300 --pressure-target-mib 300 --pressure-valid-band-mib 128 \
  --pressure-cmd "$P --target-mib 300 --band-mib 64 --floor-mib 128"
.work/rag-env/bin/python -m evaluation.gpu_local.search_bench --work .work --out .work/out-e4-q1-s2 \
  --n 300 --qdrant-timeout-s 10 --qdrant-retries 1 --pressure-target-mib 300 --pressure-valid-band-mib 128 \
  --pressure-cmd "$P --target-mib 300 --band-mib 64 --floor-mib 128"
```

各回の `summary.json` の `lines.verdict` が判定。記録（`rows.jsonl`）には問の id・時間・失敗の型だけが入る。

## 確かめていない事（11/4 時点）

- 学校 PC での E4 そのもの。クラウドの VM では e5-large の取得先（huggingface.co）がネットワークの決まりで止められていて、本物の検索は流せなかった。確かめたのは、本物の qdrant-client 1.12.1 と httpx で遅い相手に対して時間切れが効き、送り直しで答えが取れる事（`backend/tests/test_qdrant_timeout.py`）と、この道具の集計と線（`evaluation/tests/test_gpu_local_search_bench.py`）だけ
- S2（空き 300 MiB）で Windows が固まらないか。固まりそうなら S2 は止めて、止めた事を書く
- 送り直しが Qdrant の負荷を増やして遅れを長くしないか（Qdrant の FAQ は時間切れの原因の側を直すよう勧めている。送り直しが負荷を増やす事を示す一次資料は見つからなかった）
