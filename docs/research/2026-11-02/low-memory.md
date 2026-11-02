# 低メモリ化と遅延ばらつきの調査報告（local-rag-agent）

調査日: 2026-11-02。原文は `sources/` に 138 件（1 行目に URL と取得日）。調べ役（サブエージェント）の報告を親セッションが保存した（調べ役はファイルを書けない決まり）。全ノブの表と出典一覧は調べ役の返信にあり、ここは要点。設定は何も変えていない。
証拠の強さ: A=一次情報に数字 / B=一次情報に記述 / C=個人・二次 / D=推測。

## 0. 結論

1. 本機（RTX 3050 6GB）で効いているのは VRAM ではなく本体の RAM。VRAM は 3.6GB で空き 5.15GB に収まる
2. RAM の大物は 2 つ。e5-large fp32（2,235.7MB・うち埋め込み表 1,024MB）と Gemma 4 の PLE 表（CPU 側 2,208MiB）
3. 一番確実に減らせるのは検索用の埋め込みを小さくすること。JMTEB 検索平均: e5-large 70.98、ruri-v3-30m 78.08（37M）、ruri-v3-130m 81.89 [A]。自社コーパスでは未確認
4. 今の llama.cpp には関係する新フラグがある: `-lzm`（PLE を必要な行だけ読む）・`-lm dio`（O_DIRECT）・`-cram`・`-fit` [A]
5. 最初の文字が 5.76 秒に伸びた第一候補は「PLE の行引きがディスク読みの page fault になった」（H1・未検証・ソースからの推論）
6. vLLM では Gemma 4 の PLE 表は GPU 上で、`--cpu-offload-gb` 単体では動かせない。PLE を CPU に出す PR #62354 は未マージ [A]
7. 測り方が先。空きメモリが毎回違う状態で比べると、どの案も判定できない

## 1. 仮説（全部未検証）

| ID | 仮説 | 確かめ方 |
|---|---|---|
| H1 | PLE 表（2.2GiB・mmap）の行引きが major fault になる。冷えた PLE は 207,913 回の fault で 18 秒の例（NVMe・Linux）[A] | `/proc/<pid>/stat` の majflt を 1 問ごとに差分、`vmstat 1` の bi |
| H2 | Windows の空きが尽きて WSL の VM やブラウザがページアウトされる | `\Memory\Available MBytes`・`Pages Input/sec`・vmmem の作業セット |
| H3 | e5-large・Python・Qdrant・ページキャッシュで 7.7GB の WSL が圧迫され、mmap ページが追い出される | `/proc/pressure/memory`・`free -m` |
| H4 | 検索の 5 秒はデータ量でなく全体の停止（40 塊は約 0.16MB）。qdrant-client の既定 timeout は 5 秒で検索の再試行は無い [A] | 検索時間を別ログに取り Qdrant の `/metrics` と照合 |
| H5 | VRAM が足りない日に Windows の Sysmem Fallback で黙って遅くなる [A/C] | nvidia-smi と共有 GPU メモリを同時記録 |

## 2. 主なノブと判定

| ノブ | 判定 | 要点 |
|---|---|---|
| llama.cpp `-lzm on` | 試す（最優先） | `auto` は 4GiB 超だけ lazy なので 2.2GiB の PLE は対象外。E4B Q8_0 で RSS 4.17 → 1.39GiB 起点、tg −3.8%、出力は同一 [A]（Linux・64GB RAM） |
| `-lm dio -lzm on` | 試す | ページキャッシュに載せない。RSS 1.31GB の例 [A]。WSL の ext4 で O_DIRECT が通るか未確認 |
| `-lm mmap+mlock` | 比較用に試す | 非退避メモリが +2.2GiB |
| `-cram 0 -ctxcp 1` | 試す（安全側） | プロンプト状態を貯める上限（既定 8192MiB）。RAG は再利用が少ない |
| `-ot per_layer_token_embd=CUDA0` | 8GB 以上で試す・6GB は採らない | 6GB では peak 約 5,817MiB で予算 5,632 を超える（算術） |
| `-fit` と `-ngl` 未指定 | 8GB 機の保険として試す | 足りない分だけ層を CPU へ |
| Windows ネイティブ llama-server（b12032 win-cuda-12.4） | 試す | WSL の VM を介さない。Gemma 4 + FA のクラッシュ報告あり（`-fa off` で回避） |
| 公式 QAT Q4_0 GGUF | 試す（品質目的） | 大きさは Q4_K_M とほぼ同じ |
| KV 量子化・`--swa-full`・MTP ドラフト・`--no-mmap` | 採らない | KV は元々小さい・q4_0 は壊れる例・速度退行の前例・VRAM を食う |
| vLLM PR #62354（PLE を CPU へ） | 試す（本番経路の最優先） | E2B の GPU 重み 9.73 → 5.36GiB、遅延同じ（4090・未マージ）。AWQ E2B なら GPU 側約 2.7GiB の見込み（算術） |
| vLLM `--cpu-offload-gb` 単体 | 採らない | E2B で 0.79 → 40.6 秒 |
| vLLM `--kv-cache-dtype fp8` | 採らない | sm_86 では TRITON_ATTN の fp8 KV が未対応 |
| vLLM `--max-num-seqs 1〜2`・`--limit-mm-per-prompt` | 採る | 並列と画像音声のプロファイルを削る |
| vLLM `--gpu-memory-utilization` | 採る（値は実測） | 総容量 × 値を確保し、空きが足りなければ起動しない。本機は 0.80 以下 |
| vLLM の注意計算（head_dim 512） | 速度は要測定 | sm_86 は 1 ブロック 99KB で足りる計算。T4 は 64KB で起動しない。4090・v0.19 で 9.2 tok/s の報告 |
| ruri-v3-30m / 130m | 試す（最優先） | 256 / 512 次元。索引し直しと前置き「検索クエリ: 」「検索文書: 」が要る |
| e5-large の fp16/int8 | 採らない | 埋め込み表が fp32 のまま残る |
| Qdrant の on_disk・量子化・tier | 採らない | 40 塊では不要 |
| qdrant-client の timeout を 10 秒 + 1 回再試行 | 試す | 症状への対処。原因の証拠は majflt 側 |
| モデルと Qdrant を WSL の ext4 側に置く | 採る（現状確認） | `/mnt/c` を避ける |
| `read_ahead_kb`・`.wslconfig`・Sysmem Fallback・HAGS の変更 | 今回はしない | 設定変更。根拠も個人報告か数字なし |

## 3. 測り方

- RAM 要件: 部品ごとに匿名メモリと触るファイルページを分けて測る（smaps_rollup の Pss_Anon / Pss_File）。使える RAM を段階的に絞り、p95 ≤ 12 秒が崩れる直前（膝）を探す。WSL 内は cgroup の memory.max、Windows は Sysinternals Testlimit。製品の要件 = 膝 + 余裕（割合は測る前に決める・案 25%）
- 遅延: 条件（build・GGUF の sha256・フラグ・WSL の kernel・電源モード）を記録。冷え = 新しいプロセス + キャッシュを空に。温まり = 5 件捨てた後。50 問 × 3 周、アームは順番を交互に。圧迫を S0（なし）・S1（空き約 1.0GiB）・S2（約 0.3GiB）で固定。毎秒 GPU・Windows のメモリ・PSI・majflt を記録。熱や電源で絞られた回は捨てる。線は測る前に固定

## 4. 実験の優先順位（調べ役の案・線は測る前に確定する）

1. 埋め込みを縮める（B0 e5-large / B1 ruri-v3-30m / B2 ruri-v3-130m）。線: M1 ≥ 0.80・M2 ≥ 0.60、現行より 0.02 以上下がらない、常駐 ≤ 0.5GiB（B1）/ 0.8GiB（B2）、クエリ p95 ≤ 100ms、節約 ≥ 1.5GiB
2. llama-server のメモリ方式 4 アーム（A0 現行・A1 `-lzm on`・A2 `-lm dio -lzm on`・A3 mlock）。線: S0 で退行なし（P1 p50 ≤ A0 × 1.10 かつ ≤ 5 秒、tg ≥ 37）、S1 で P1 p50 ≤ 5 秒・P2 p95 ≤ 12 秒・失敗 0、出力が A0 と同一
3. qdrant-client の timeout 10 秒 + 1 回再試行。線: S1・S2 各 300 件で時間切れ 0・検索 p95 ≤ 1 秒・p99 ≤ 3 秒
4. llama-server を Windows ネイティブに。線: S1 で P1・P2 の線、失敗 0、全体の常駐が WSL 版より 1.0GiB 以上小さい、VRAM ≤ 5,632MiB、冷え起動 ≤ 60 秒
5. vLLM で PLE を CPU へ（PR #62354 を当てる・sm_86）。線: 起動・GPU 重み ≤ 3.5GiB・VRAM 予算内・P1/P3 の線・M6〜M8 が同モデルの非オフロードと同じ

## 5. 証拠が薄い所

- 6GB ノート + WSL2 + Gemma 4 E4B の測定は本人の 1 件だけ
- `-lzm on` の 2.2GiB PLE への効果は未検証（公開値は Linux・Q8_0・64GB RAM のみ）
- H1 はソースの読みと別条件の測定からの推論
- WSL2 の `read_ahead_kb=8192` は個人報告だけ。ホストが vmmem をページアウトするかの一次情報なし
- NVIDIA の Sysmem Fallback の原文（KB 5490）は 403 で読めず
- vLLM PR #62354 は未マージ・4090 のみ・WSL2 の pinned 制約は未検証。vLLM の Gemma 4 デコード速度は v0.32.0 で不明
- ruri-v3 の自社コーパスでの性能は不明
