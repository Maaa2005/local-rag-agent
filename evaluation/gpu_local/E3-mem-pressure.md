# Issue #14 E3: 本体のメモリを絞った状態での遅れと失敗（線は測る前に固定）

E2（11/2）は `-lzm on` でファイル由来の常駐とページ読みが減ったのに、線にした常駐の合計が減らず不合格だった。
狙いは「本体のメモリが足りない時の遅れと失敗」なので、それを直接測る。E2 の線は動かさない。

## 腕と条件

- 腕: A0 = 今の起動（`-ngl 999 -c <測った文脈> --jinja --reasoning off -np 1 -v`）、A1 = A0 + `-lzm on`。モデル・C・e5-large・62 問は E2 と同じ。各 1 回、A0 → A1 の順
- 絞り方 S1: 回答の回の間だけ、Windows の空きを 1,024 MiB に保つ（`evaluation/gpu_local/mem_pressure.py`）。WSL の中で匿名メモリを持ち、Windows の空き（`Win32_OperatingSystem.FreePhysicalMemory`）を 1 秒ごとに読んで、持つ量を増減する。Windows か WSL の空きが 256 MiB を下回ったら持った分を全部放す
- 有効な回: 回答の間の Windows の空きの中央値が 1,024 ± 256 MiB に入り、全部放しが 0 回、絞る道具が途中で落ちていない。有効でない回は合否を出さない（`無効`）。同じ設定で 1 回だけ流し直してよい。2 回とも無効なら「測れなかった」と書く

## 線（A1。全部満たして「本測定へ進める」・`summary.e3_line_checks`）

| # | 線 | 元 |
|---|---|---|
| 1 | 起動し、全層 GPU、62 問全部回答、メモリ不足・文脈あふれ 0、記録に lazy 読みの行が出る | E2 の線 1 |
| 2 | 最初の文字 p50 ≤ 5.0 秒 | 製品の線 P・調査 `docs/research/2026-11-02/low-memory.md` §4 の 2（S1） |
| 3 | 完了 p95 ≤ 12.0 秒 | 同じ §4 の 2（S1 の P2 p95） |
| 4 | N の一致数と M8 が E2 の A0（11/12・16/16）と同じ、M6 の差 ≤ 0.02（E2 の A0 は 0.9733） | E2 の線 4 |

記録だけ（合否に使わない）: A0 の同じ数、空きの最小・中央値・最大と幅に入った割合、持った量の最大、PSI（`/proc/pressure/memory` の some / full の増え方と avg10 の最大）、llama-server と評価のプロセスの常駐・ページ読みの回数、検索の時間の p95・最大、時間切れの問。

A0 が落ちて A1 が通れば「`-lzm on` が絞った時に効く」、両方通れば「S1 では差が出ない」、両方落ちれば「S1 では足りない」と書く。どれでも線は動かさない。

## 流し方（学校 PC・WSL2）

`README.md` の「準備」に加えて、WSL から `powershell.exe` が呼べる事を確かめる（`powershell.exe -NoProfile -Command "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory"` が数を返す）。ほかのアプリは閉じ、各回の前の Windows の空きを記録する（`summary.json` の `pressure.before_free_mib` にも残る）。

```bash
.work/rag-env/bin/python -m evaluation.gpu_local.step2_local --work .work --out .work/out-e3-a0 \
  --runs 1 --skip-g0 --label E3-A0-S1 --pressure-target-mib 1024
.work/rag-env/bin/python -m evaluation.gpu_local.step2_local --work .work --out .work/out-e3-a1 \
  --runs 1 --skip-g0 --label E3-A1-S1 --pressure-target-mib 1024 --server-extra '-lzm on'
```

判定（A1）:

```bash
.work/rag-env/bin/python -c "import json,sys; from evaluation.gpu_local import summary as sm; \
r=json.load(open('.work/out-e3-a1/summary.json'))['runs'][0]; print(sm.e3_line_checks(r, expect_lazy=True))"
```

- 回答の本文を含む `raw.jsonl` と絞る道具の記録（`pressure_*.jsonl`）は `.work/` に残し、リポジトリには数だけを入れる
- 絞る道具だけを試す時: `python -m evaluation.gpu_local.mem_pressure --target-mib 1024 --probe windows --log .work/p.jsonl --duration 60`

## 確かめていない事（11/4 時点）

- 学校 PC での E3 そのもの（GPU が要る）。クラウドの VM（GPU なし・WSL でない）では、絞る道具が Linux の空きを目標まで下げて保ち、止めると放す事と、step2_local から起動・停止・区間のまとめができる事だけを確かめた
- WSL の中で持った匿名メモリが、Windows の空きを目標どおりに減らすか（WSL の VM の上限・WSL の swap・Windows のページングによる）
- `powershell.exe` を 1 本流し続ける事による Windows 側の負荷
