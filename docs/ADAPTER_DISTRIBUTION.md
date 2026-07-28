# LoRAアダプター配布・インストール方針

## 方針

Gemma 4 E2B用の司令塔LoRA重み `adapter_model.safetensors` はGitおよび
Git LFSへ同梱せず、Hugging Face Hubから配布する。Gitには実験系譜、期待する
SHA-256、設定ファイル、取得・検証コードを保存する。

この変更方針はMaaa2005からの指示に基づく。重み同梱を前提として従来の
インストールフローを実装したyuu-0617には、ブランチをpushして共有する前に、
HF Hub配布へ変更することと実際のHubリポジトリを伝えること。

## 必要な設定

クリーンインストール前に `.env`へ次を設定する。

```dotenv
ADAPTER_HF_REPO=your-organization/hr-orchestrator
ADAPTER_HF_REVISION=main
# 非公開またはgated repositoryの場合のみ
HF_TOKEN=
```

共有リリースでは `ADAPTER_HF_REVISION`をHubのcommit hashまたは変更しないtagへ
固定する。実際のHubリポジトリIDは配布先の作成後に設定し、推測したIDをコードへ
埋め込まない。

## インストール順序

1. `.env`を読み、`ADAPTER_HF_REPO`、revision、必要ならtokenを取得する。
2. ローカルに必須ファイルがなければ `hf download`でHub snapshotを取得する。
3. `adapter_config.json`と`adapter_model.safetensors`の存在を検証する。
4. safetensorsのSHA-256が承認済み実験と一致することを検証する。
5. すべて成功した場合のみDockerのpull、build、upへ進む。
6. vLLMは従来どおり `./models/hr-orchestrator`をread-onlyでマウントし、
   `--lora-modules hr-orchestrator=/models/hr-orchestrator`で読み込む。

Linux/WSLでは `install.sh`から `scripts/ensure_adapter.sh`を呼び出す。
Windowsでは `install.ps1`から `scripts/Ensure-Adapter.ps1`を呼び出す。

## 失敗時の扱い

- 重みがなくHub repoも未設定: 設定方法を表示してDocker起動前に停止する。
- HF downloaderがない: `huggingface_hub[cli]`の導入方法を表示して停止する。
- download失敗: Dockerを起動せず停止する。
- 必須ファイル欠損: Dockerを起動せず停止する。
- SHA-256不一致: 承認外の重みとしてDockerを起動せず停止する。

承認済み重みSHA-256は
`5a8f318629bbb6fcc4f0131164ab6088299cac9eeec44a76463a32f37baa3a59`。
機械可読な系譜は `models/hr-orchestrator/manifest.json`に保存する。
