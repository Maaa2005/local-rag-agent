# 2026-10-17 Laya-multilingual の初期採否

判定と読み方は [設計書「Laya の初期採否」](../../../../docs/redesign-v2.md)。結論は C1・C2 とも不採用。データはすべて架空（`../../cases.jsonl` の 39 件）。

## 条件

- Mac（Apple Silicon・mps）、laya 0.4.9、`convaiinnovations/laya-multilingual` rev `1720e3e3357cfe1e281542e223f8273b0890ca34`
- shim: `v2/judge/laya_server.py`（応答の revision は `<sha>+laya-shim@1`）。質問の版は `c1-route@3` / `c2-forbidden@1`
- 判断層評価 `evalharness.run_v2_eval`（`--c1` / `--c2` に `rules` か `http`）、`--timeout 15`（V083 は遅延注入でこの 15 秒の打ち切りに掛かる）。USE_TF=0・HF_HUB_OFFLINE=1

## ファイル

| 場所 | 中身 |
|---|---|
| `max_len8192/` | shim の max_len 8192 で 4 通り（rules / laya / rules-laya / laya-rules）。`compare.md` が 4 通りの表 |
| `max_len1024/` | max_len 1024 で Laya を使う 3 通り。rules だけの組は max_len に関係しないので `max_len8192/rules` を使った。`compare.md` は 8192 と同一 |
| `c2_prompt_probe.py` / `.txt` | C2 の質問文を 4 通りに変えた AUC 試験と、その出力（max_len 1024 の shim に対して実行） |

フォルダ名の `laya-rules` は C1 = Laya・C2 = rules、`rules-laya` はその逆。

## 再現

```sh
cd v2
JUDGE_MAX_LEN=1024 JUDGE_MODEL=convaiinnovations/laya-multilingual \
  JUDGE_MODEL_REVISION=1720e3e3357cfe1e281542e223f8273b0890ca34 JUDGE_DEVICE=mps USE_TF=0 HF_HUB_OFFLINE=1 \
  <laya を入れた venv>/bin/python -m judge.laya_server --port 18080 &
.venv/bin/python -m evalharness.run_v2_eval --c1 http --c2 http --base-url http://127.0.0.1:18080 --timeout 15 --out <出力先>
.venv/bin/python ../evaluation/v2/results/2026-10-17-laya/c2_prompt_probe.py http://127.0.0.1:18080
```
