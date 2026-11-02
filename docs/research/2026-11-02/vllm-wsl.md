# vLLM v0.25.1 を WSL2 の Docker で動かす: 調査報告

調査日: 2026-11-02。原文は `sources/` に 92 件（1 行目に URL と取得日）。調べ役（サブエージェント）の報告を親セッションが保存した（調べ役はファイルを書けない決まり）。調べ役は機械を変える操作をしていない。

## 0. 結論

1. 壊れているのはこの PC の中。Docker Hub 上の層 `54a69dfd…` は正常（サイズ 3,927,445,220 B・SHA-256・gzip 全部一致。調べ役が 2026-11-02 06:29–06:41Z に保存なしのストリームで測定）
2. この層はイメージ最大（圧縮 3.66GiB・PyTorch などの層）。イメージは 33 層・圧縮 8.21GiB。展開後は約 30GB（上流の記述）
3. 原因候補: (a) 保存済みの層ファイルが壊れている（containerd#14084: c8d は壊れた層を残し、再 pull でも同じ失敗）。(b) unpigz の不調。保存済みファイルの SHA-256 を測れば分かる（読むだけ）
4. エラー文 `too many length or distance symbols` は zlib が「ブロック頭の値が不正」と見た時の文言（途中切れなら別の文言）
5. 同じタグ v0.25.1 は 7/28 にこの PC で動いた記録がある（`origin/backup/20260728-word-edit`）

## 1. 直す順（弱 → 強）

| 段 | やること | 変更 |
|---|---|---|
| A | 他アプリを閉じて空きを増やす | なし |
| B | 診断: 保存済みの層の SHA-256 を 2 回、`gzip -t`・`unpigz -t`、`pigz -V`、dockerd/containerd の環境変数 | 読むだけ（約 3.7GB を読む） |
| C1 | (a) なら `ctr -a /var/run/docker/containerd/containerd.sock -n moby content rm sha256:54a69dfd…` でこの層 1 個だけ消し、ダイジェスト指定で再 pull | 3.7GB のファイル 1 個を削除 |
| C2 | (b) なら dockerd を `CONTAINERD_DISABLE_PIGZ=1 CONTAINERD_DISABLE_IGZIP=1 MOBY_DISABLE_PIGZ=1` 付きで起動し直して再 pull（dockerd が起動する containerd は環境変数を引き継ぐ） | 再起動のみ |
| C3 | 再 pull はダイジェスト指定 `vllm/vllm-openai@sha256:f0b9a0dc75a9fca3b6811e3279367b2d6a448055a000bfd13859587d74cef268`（amd64）。pull 中に `image prune` を走らせない（moby#53321） | なし |
| D | graphdriver に切替 `"features":{"containerd-snapshotter":false}` | 設定変更。既存イメージは見えなくなる。戻せる |
| E | containerd ストアの全消去 | やらない |

診断の読み方: SHA 不一致・gzip 失敗 → (a) で C1。SHA 一致・gzip OK・unpigz 失敗 → (b) で C2。全部 OK → 展開中の問題で C2 → D。

## 2. 別の道

- ダイジェスト固定: compose に `image: vllm/vllm-openai:v0.25.1@sha256:e4f88a835143cd22aee2397a26ec6bb80b3a4a6fe0c882bcbc63822904766089`（2026-11-02 時点のインデックス）。採る
- 別 PC で pull → save/load: 最後の手段
- Dockerfile からビルド: RAM 7.7GB では非現実的。採らない
- WSL に venv で pip/uv: 最終手段（Docker の read-only 試験にはならない）。`+cu129` wheel が安全側。ドライバ 581.86（R580）は CUDA 12.x/13.x とも可。nvcc が無いと FlashInfer のサンプラー JIT が落ちる → `VLLM_USE_FLASHINFER_SAMPLER=0`

## 3. WSL2 固有の既知問題

- v0.25.0 から V2 ランナーが既定で、WSL2 では `UVA is not available` になる。`VLLM_WSL2_ENABLE_PIN_MEMORY=1` か `VLLM_USE_V2_MODEL_RUNNER=0` で直る（Docker `--runtime nvidia` + WSL2 で複数人確認）。根本修正 #56908 は 9/15 マージで v0.25.1 には無い
- `--cpu-offload-gb` は V2 ランナーで黙って無視される（`set_offloader` は V1 のみ）。7/28 の「オフロードあり」の実測が効いていたかは要確認
- `--cpu-offload-gb` と CUDA グラフの併用で pinned メモリを大量に使い Windows が固まる恐れ。`--enforce-eager` なら避けられる

## 4. 読み取り専用・root 以外

- 公式は root が既定。`--user 2000:0`（同梱ユーザー vllm）なら動く。ほかの UID はサポート外
- UID 10003 は公式イメージの `/etc/passwd` に無い → torch 2.11 の Inductor が `getpass.getuser()` で KeyError になりうる → `USER=vllm`・`LOGNAME=vllm` を渡す
- Docker の tmpfs は既定で `noexec`。Triton / FlashInfer が `/tmp` に作った `.so` を読めない恐れ → `/tmp:exec,size=1g`。第三者の read-only 成功例（vllm#59250・v0.30.0）は exec 付き
- NVIDIA のフックが read-only root で失敗する報告（nvidia-container-toolkit#2131・podman/CRI-O）。Docker + WSL2 では報告なし → 先に小さな CUDA イメージで `docker run --gpus all --read-only --entrypoint nvidia-smi` を確かめる
- 書込先の洗い出し: read_only なしで一度動かし `docker diff` を見る

compose に足す環境変数の案（未実測）:
```yaml
environment:
  HOME: /tmp
  USER: vllm
  LOGNAME: vllm
  XDG_CACHE_HOME: /tmp/.cache
  VLLM_CACHE_ROOT: /tmp/vllm-cache
  VLLM_CONFIG_ROOT: /tmp/vllm-config
  VLLM_NO_USAGE_STATS: "1"
  DO_NOT_TRACK: "1"
  TRITON_CACHE_DIR: /tmp/triton
  TORCHINDUCTOR_CACHE_DIR: /tmp/inductor
  NUMBA_CACHE_DIR: /tmp/numba
  FLASHINFER_WORKSPACE_BASE: /tmp
  VLLM_USE_FLASHINFER_SAMPLER: "0"
  PYTHONDONTWRITEBYTECODE: "1"
  HF_HOME: /tmp/hf
  VLLM_WSL2_ENABLE_PIN_MEMORY: "1"
tmpfs:
  - /tmp:exec,size=1g
shm_size: 1gb
```
`VLLM_NO_USAGE_STATS` を付けないと usage_stats.json を書き stats.vllm.ai へ送ろうとする（社内ゾーンは外へ出られないが、明示して止める）。

## 5. Gemma 4 E2B（sm_86・6GB）

- PLE（層ごとの埋め込み）は vLLM では GPU 上で、退避の仕組みが無い（v0.25.1）。6GB に載らない主因（7.06GiB のうち PLE 4.38GiB）
- attention: Gemma 4 は head_dim 512 の層があり TRITON_ATTN に固定。T4 は 98,304B 必要で上限 65,536B を超え失敗。sm_86 の上限は 101,376B で収まる（差 3,072B）。L4・A100 では同構成が動いた報告あり。v0.25.1 での sm_86 実測は無い
- 速度: Gemma 4 の TRITON_ATTN は遅い報告が多い（4090 で約 9 tok/s・v0.19、4070 Ti Super + WSL2 で 3.5〜4 tok/s・v0.22）
- 4bit: sm_86 は Marlin。cyankiwi AWQ-INT4 は 7/28 に v0.25.1 で動いた実績
- 公式の小型版 `google/gemma-4-E2B-it-qat-mobile-ct`（約 2.5GiB）。v0.25.1・sm_86 で動くか未確認

## 6. やらないこと

- `apt remove pigz`（docker.io ごと外れる）・pigz を gzip へ symlink（機械全体が変わる）
- containerd ストアの全消去・Docker の版上げ（効く根拠なし）
- pull 中の prune
- WSL に Linux 用 NVIDIA ドライバを入れる（NVIDIA が禁止）
- `--ipc=host`（隔離を弱める。まず `shm_size`）

## 7. 証拠が薄い所

- 原因の確定は診断待ち。メモリ圧迫で gzip が壊れる一次報告は無い
- Triton が noexec で落ちる直接報告は無い。Docker(runc) + WSL2 + read_only + nvidia の報告も無い
- Gemma 4 E2B + sm_86 の直接報告は無い（共有メモリは数字の比較による推定）
- mobile-ct の v0.25.1 対応、PyPI 既定 wheel の CUDA 版は未確認
