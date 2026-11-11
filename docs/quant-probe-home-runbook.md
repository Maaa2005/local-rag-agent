# 量子化の物差し: 自宅 PC で本番を回す手順（life#24）

計画は `docs/quant-probe-plan.md`（版 7）。**判定の線は変えない。** ここは回し方だけ。

## 0. 前提と止める条件

- 機械: Windows・RTX 3060 12GB・RAM 16GB。GPU を長く使うので、本人が使っていない時間に回す。重い処理は 1 度に 1 つ（道具が優先度を下げて起動する）。
- 止める: 問い（何を測るか）を変えたくなった時・同じ失敗で 2 回直しても通らない時・`report.py` が「判定なし」を出した時（理由を life#24 に書いて止める）。有料の API は使わない。封をした 240 問は使わない。
- モデルは作者の Hugging Face から `convert_hf_to_gguf.py --remote` で流し読みし、BF16 の GGUF だけ書く（safetensors は置かない）。流し読みの量は約 56GB（8B 2 つで約 16GB ずつ・12B 約 24GB）。

## 1. 道具をそろえる（初回だけ）

1. この枝（取り込み後は `redesign/v2`）を取る。`python -m pytest -q evaluation/tests/test_quant_probe.py` が通ること。
2. llama.cpp **b11836**（commit f5027fb21）の Windows・CUDA 12 版を、公式の置き場 `https://github.com/ggml-org/llama.cpp/releases/tag/b11836` から取る。置き場に出ている sha256 と照らす。同じ版の `convert_hf_to_gguf.py` と `gguf-py` は同じ commit の source から取る。
3. 変換用の Python の環境に `requirements/requirements-convert_hf_to_gguf.txt` を入れる。`texts.py` 用に `pyarrow`。

## 2. 文章をそろえる（初回だけ）

公式の置き場から 4 つ取り、`texts.py` で作る。出来た文の sha256 が `evaluation/quant_probe/texts.manifest.json` と同じであること（違えば止める）。

| 名前 | 取る所 |
|---|---|
| wikitext2-test.parquet | `https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/test-00000-of-00001.parquet` |
| jawiki-20231101-train-00014.parquet | `https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.ja/train-00014-of-00015.parquet` |
| text_en_tiny.parquet | `https://huggingface.co/datasets/eaddario/imatrix-calibration/resolve/main/text_en_tiny.parquet` |
| text_jp_tiny.parquet | `https://huggingface.co/datasets/eaddario/imatrix-calibration/resolve/main/text_jp_tiny.parquet` |

```bash
python -m evaluation.quant_probe.texts --wikitext-test src/wikitext2-test.parquet --ja-wiki src/jawiki-20231101-train-00014.parquet --calib-en src/text_en_tiny.parquet --calib-ja src/text_jp_tiny.parquet --out texts
```

## 3. モデルごとに回す（qwen3-8b → swallow-8b → gemma4-12b の順）

| 作業の名前 | 作者の置き場 | BF16 が GPU と RAM に載るか | 足す引数 |
|---|---|---|---|
| qwen3-8b | `Qwen/Qwen3-8B` | 載る（約 16GB） | `--bf16-imatrix-check` |
| swallow-8b | `tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5` | 載る（約 16GB） | `--bf16-imatrix-check` |
| gemma4-12b | `google/gemma-4-12B-it` | 載らない（約 24GB） | なし |

1. **先に 3 つ全部を変換する（どのモデルの答えも流す前）**: 3 つそれぞれ `python convert_hf_to_gguf.py --remote <作者の置き場> --outtype bf16 --outfile models/<名前>-bf16.gguf` と `llama-quantize models/<名前>-bf16.gguf models/<名前>-q8.gguf Q8_0` が通るか確かめる（この Q8_0 は確かめるためだけの物で、回す時は道具が作り直す。消すかどうかは本人）。
   どうしても通らない物（同じ失敗で 2 回直しても通らない）があれば、`work/<名前>/unconvertible.txt` に理由を書き、代わりに `Qwen/Qwen3-14B`（作業の名前 `qwen3-14b`・`--bf16-imatrix-check` なし）を同じく変換して確かめる。替えは 1 回まで。
   そのあと、使う 3 つ（替えた時はその理由も）を life#24 にコメントで書いて固定する。**このコメントより前に答えを流さない。** 答えを流し始めた後で変換できない物が出ても替えない（そのモデルはデータ欠損・計画 4 章）。答えの記録がある作業フォルダに印を置くと、集計はデータ欠損にする。集計はこの順番（コメントが答えより先か）を確かめられない。証拠は、コメント（使う 3 つの名前・替えた理由・計画の版を書く。後で消さない・書き換えない）の時刻と、本番の最初の答えの起動時刻を、日本時間にそろえて比べたもの。起動時刻は `work/<名前>/timing.jsonl` で最初の `:answers` の行の `at` から `sec` を引いた時刻（`at` は終わった時刻で、その PC の時計の時間帯）。始める前に `date` で WSL の時間帯を見て、コメントに書いておく。順番を守れなかった時は、その事を life#24 に書き、計画どおりの結果として扱わない。
2. 回す（途中で止めても同じ命令で続きから）:
   ```bash
   python -m evaluation.quant_probe.pipeline --model <名前> --bf16 models/<名前>-bf16.gguf --texts texts --work work --bin <llama.cpp の exe のフォルダ> [足す引数]
   ```
   既定: 15 変種・KLD 80 区切り・imatrix 128 区切り・基準 2 回・Q4_K_M-en をもう 1 回・基準が GPU に載り切らない時は Q6_K-none を基準の載せ方でもう 1 回。
3. モデル 1 つ分が済んだら、そのモデルだけで集計して表を見る（判定は 3 つそろってから）:
   ```bash
   python -m evaluation.quant_probe.report --work work --models <名前> --out results/<名前>
   ```
   表の頭の「状態」を見る。「データ欠損」なら理由を直してから次へ（決めた 3 つの全体の答えは、この時点では「決めたモデルと違う」で出ないのが正しい）。「基準不適格」（基準が線 0 を割った）は**替えずにそのまま**次のモデルへ（そのモデルは「測れない」と数える・計画 4 章）。
4. 3 つそろったら全体の集計:
   ```bash
   python -m evaluation.quant_probe.report --work work --models qwen3-8b swallow-8b gemma4-12b [qwen3-14b] --out results/final
   ```
   `results/final/README.md` を枝に入れ、要点を life#24 に書く。

## 4. ディスク（1 モデルあたりの最大）

| | BF16 | Q8_0 | 変種（上書き 1 つ） | KLD の比べ元 3 つ | 計 |
|---|---|---|---|---|---|
| qwen3-8b | 16GB | 8.7GB | 6.7GB 以下 | 約 17GB | 約 49GB |
| swallow-8b | 16GB | 8.5GB | 6.6GB 以下 | 約 14GB | 約 45GB |
| gemma4-12b | 24GB | 12.7GB | 9.8GB 以下 | 約 29GB | 約 76GB |

消さずに続けると約 170GB。足りなければ、モデルの集計を枝に入れた後で、そのモデルの `work/<名前>` と BF16 を本人が消す（AI は消さない）。

## 5. 時間の見積り

学校 PC の試し（4B・90 問・区切り 20・RTX 3050 6GB・基準は 37 層のうち 31 層だけ GPU）の時間から伸ばした**目安**。外れうるので、最初のモデルの `work/<名前>/timing.jsonl` で見直す。

| 段 | 試しの時間 | 本番の 4B に直すと（問 4 倍・区切り 4 倍） |
|---|---|---|
| Q8_0 を作る・校正 2 つ | 3 分・9 分 | 同じ |
| KLD の比べ元 3 つ | 4.5 分 | 約 18 分 |
| 基準の答え 2 回 | 19 分 | 約 75 分 |
| 変種 1 つ（作る 3 分・KLD 3 つ 2.5 分・答え 4.3 分） | 約 10 分 | 約 30 分 × 15 = 約 7.5 時間 |
| 揺れの組（Q4_K_M 英語の 2 回目・載せ方違い） | 3.8 分・10.6 分 | 約 15 分・約 43 分 |
| BF16 から校正して KLD だけ | 21 分 | 約 28 分 |

- 4B をこの PC で本番どおりに回すと約 11 時間。
- 自宅 PC（RTX 3060 12GB）はメモリの速さが約 1.7〜1.9 倍だが、モデルは 2〜3 倍重い。目安は 8B が 1 つ 10〜15 時間、gemma4-12b が 15〜25 時間（Q8_0 と BF16 が載り切らないので遅い）。3 つで 2〜3 日、ほかに変換（流し読み約 56GB）。
- 1 つずつ低い優先度で回す。途中で止まっても、同じ命令で流し直せば済んだ段は飛ばす（試しで確かめた）。答えは済んだ問を飛ばして続きから、KLD と校正は済んでいない物だけやり直す。止まった後は、答えの記録の最後の行が途中で切れていないかを先に見る（切れていたら、その 1 行を消してから流し直す）。
