# Gemma 4 E2B 再学習・評価コード

このディレクトリには、親プロジェクト `C:\Users\suika\hr-assistant-ai` から選別した最終FTの再現に必要なコードだけを収録している。

## 構成

- `train_unsloth.py`: Gemma 4 E2Bの4bit LoRA学習
- `run_remote_experiment.sh`: dataset hash、GPU、環境、ログ、adapterをrun単位で保存
- `validate_dataset.py`: 全assistant出力をPydantic Schemaで検証
- `generate_dataset.py`: 基礎720会話の合成データ生成
- `build_remediation_dataset.py`: 第一次改修データの構築
- `build_ood_remediation_dataset.py`: OOD向け224件を加えた最終884件の構築
- `freeze_ood_dataset.py`, `check_ood_overlap.py`: OOD固定とtrain重複検査
- `evaluate_model.py`, `compare_*.py`: base/FT評価と比較
- `src/hr_assistant/schema.py`: 司令塔JSONの教師・評価Schema

固定データは `docs/ft-data/`、最終結果は `docs/ft-evidence/`、承認済みadapterは `models/hr-orchestrator/` にある。

## データ検証

PowerShell:

```powershell
$env:PYTHONPATH='training/src;.'
python training\validate_dataset.py docs\ft-data\train.jsonl
python training\validate_dataset.py docs\ft-data\validation.jsonl
python training\validate_dataset.py docs\ft-data\test.jsonl
```

Linux:

```bash
PYTHONPATH=training/src:. python training/validate_dataset.py docs/ft-data/train.jsonl
```

## 再学習

学校AIサーバー等のCUDA環境で実行する。

```bash
python -m pip install -r training/requirements-training.txt
CONDA_ROOT="$HOME/miniconda3" \
DATASET="docs/ft-data/train.jsonl" \
MAX_STEPS=160 \
RUN_ID="gemma4-orchestrator-reproduction" \
bash training/run_remote_experiment.sh
```

出力は `outputs/experiments/<RUN_ID>/` に保存される。最初は `MAX_STEPS=10` でsmoke testを行い、GPUメモリ、loss、adapter保存を確認する。

## 重要な分離条件

- `validation.jsonl`、`test.jsonl`、`human-authored-ood.jsonl` を学習へ入れない。
- 企業規程の本文をSFTで暗記させない。規程はQdrant RAGで管理する。
- アクセス制御をFTモデルへ委任しない。バックエンドで必ず再検証する。
- Hugging Face token、SSH password、学校アカウントの秘密情報をrun logやGitへ保存しない。
