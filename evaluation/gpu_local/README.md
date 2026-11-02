# 手元の GPU で Step 2 を流す（Issue #1）

`evaluation/colab/step2_measure_llamacpp.ipynb` と同じ手順を、Colab ではなく手元の Linux GPU（WSL2 でよい・Docker 不要）で流す。
2026-11-02 に学校 PC（RTX 3050 6GB Laptop）で使った。線は Issue #1 の 11/2 のコメントで測る前に決めた。

## 準備（`.work/` に置く。`.work/` はコミットしない）

Colab のノートと同じ版・同じ配布元だけを使う。

```bash
cd <repo>
mkdir -p .work/llama-pre .work/qdrant/bin .work/gguf
base=https://github.com/ggml-org/llama.cpp/releases/download/b11836
curl -sfL $base/llama-b11836-bin-ubuntu-cuda-12.8-x64.tar.gz | tar xz -C .work/llama-pre
curl -sfL $base/cudart-llama-b11836-bin-ubuntu-cuda-12.8-x64.tar.gz | tar xz -C .work/llama-pre
curl -sSL https://github.com/qdrant/qdrant/releases/download/v1.12.6/qdrant-x86_64-unknown-linux-gnu.tar.gz | tar xz -C .work/qdrant/bin
uv venv -p 3.11 .work/rag-env
uv pip install --python .work/rag-env/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .work/rag-env/bin/python "transformers<5" sentence-transformers==3.3.1 \
    qdrant-client==1.12.1 aiosqlite==0.20.0 pydantic-settings==2.6.1 openai==1.57.0 httpx==0.28.1 pyyaml
# ruri-v3（埋め込みの比較・Issue #14 E1）は SentencePiece の分かち書きを使う
uv pip install --python .work/rag-env/bin/python sentencepiece protobuf
curl -sfL -o .work/gguf/gemma-4-E4B-it-Q4_K_M.gguf \
  https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/bfc15c382204943c3a8fff0c750b94ae2364d7a3/gemma-4-E4B-it-Q4_K_M.gguf
```

WSL2 では CUDA の本体（`libcuda`）は Windows のドライバが `/usr/lib/wsl/lib` に出すので、CUDA Toolkit は入れない。

## 流す

```bash
.work/rag-env/bin/python -m evaluation.gpu_local.step2_local --work .work --out .work/out-$(date +%Y%m%d-%H%M)
```

- 索引 → 入力トークン数の測定（文脈 8192・出力 1）→ 必要な文脈で別々に起動して 62 問 × 2 回 → G0 → `summary.json`
- 予算は `--budget-mib`（既定 5,632 = 6,144 − 512）。帯域は `--bandwidth-gbs`（既定 168 = RTX 3050 6GB Laptop の 96bit × 14Gbps）
- 回答の本文を含む `raw.jsonl` は `.work/` に残し、リポジトリには数だけ（`summary.json` から本文の無い欄）を入れる

## 線（Issue #1・測る前に固定）

| 線 | 合格 |
|---|---|
| V | カード全体の最大 ≤ 予算・全層 GPU・OOM/文脈あふれ 0・62 問完走 |
| P | 最初の文字 p50 ≤ 5.0 秒・完了 p50 ≤ 15.0 秒（測定値のまま。換算しない） |
| Q | M6 ≥ 0.80・M7 = 1.00・M8 16/16 |
| N | 12/12 |
| G0 | 全項目 PASS |

換算の確かめ（合否と別）: T4 の出力速度 51.1 トークン/秒 × 帯域の比 を予測にし、実測が 0.75〜1.25 倍に入るか。
