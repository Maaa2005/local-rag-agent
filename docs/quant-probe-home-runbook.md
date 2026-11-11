# 量子化の物差し: 自宅 PC で本番を回す手順（life#24）

計画は `docs/quant-probe-plan.md`（版 4）。**判定の線は変えない。** ここは回し方だけ。

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

1. 変換: `python convert_hf_to_gguf.py --remote <作者の置き場> --outtype bf16 --outfile models/<名前>-bf16.gguf`
   変換か `llama-quantize` がどうしても通らない時だけ（同じ失敗で 2 回直しても通らない）、答えを 1 問も流す前に `work/<名前>/unconvertible.txt` に理由を書き、そのモデルの代わりに `Qwen/Qwen3-14B`（作業の名前 `qwen3-14b`・`--bf16-imatrix-check` なし）を回す。替えは 1 回まで。
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

（学校 PC の試しの所要時間から書く。試しの後に埋める）
