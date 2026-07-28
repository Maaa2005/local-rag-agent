# Local RAG Agent with HR Orchestrator

This package runs the complete local stack with Docker Compose:

- frontend: browser UI
- backend: authentication, audit logs, ingestion, retrieval, and orchestration
- qdrant: local vector database
- vllm: one quantized base model exposed as `local-llm`, plus the fine-tuned
  LoRA adapter exposed as `hr-orchestrator`
- Word editing: authenticated DOCX find/replace drafts with explicit preview,
  user confirmation, and one-time download

The orchestrator runs before retrieval. It either asks for missing information,
rejects unsafe or unauthorized requests, or permits the existing RAG pipeline to
continue. Document access levels are still enforced in backend code; model output
cannot raise a user permission level or disable retrieval filters.

The Word workflow never overwrites the uploaded original and does not finalize,
sign, or submit documents. Edited bytes remain in memory for up to 15 minutes and
are returned only after the same user explicitly confirms the preview. The user is
responsible for final review and submission. See `DEMO_GUIDE_20260728.md` for the
demonstration flow and supported document scope.

## 資料の場所

資料の入口は [docs/README.md](docs/README.md) です。ローカルの絶対パスは次のとおりです。

`C:\Users\suika\hr-assistant-ai\local-rag-agent\docs\README.md`

| 内容 | 資料 | リポジトリ内パス |
|---|---|---|
| 操作方法、ログイン、デモ手順、Word操作 | [デモ・操作ガイド](DEMO_GUIDE_20260728.md) | `DEMO_GUIDE_20260728.md` |
| FTの意図、モデル、データセット、学習条件、結果 | [FT設計・結果](docs/FT_DESIGN_AND_RESULTS.md) | `docs/FT_DESIGN_AND_RESULTS.md` |
| 実際のFT用JSONLとSHA-256 | [FTデータセット](docs/ft-data/README.md) | `docs/ft-data/` |
| RAGの精度、権限、整合性試験 | [RAG検証結果](RAG_VALIDATION_REPORT_20260728.md) | `RAG_VALIDATION_REPORT_20260728.md` |
| Word編集の承諾・非保存・描画試験 | [Word検証結果](WORD_EDIT_VALIDATION_REPORT_20260728.md) | `WORD_EDIT_VALIDATION_REPORT_20260728.md` |
| RAG要求仕様 | [RAG要件定義](RAG要件定義.md) | `RAG要件定義.md` |
| 企画目的 | [企画書](企画書.md) | `企画書.md` |
| 評価データと採点条件 | [evaluation README](evaluation/README.md) | `evaluation/` |

## 技術をどこに使っているか

### 全体構成

| 技術 | このシステムでの用途 | 主な実装場所 | 実行時のデータ・接続先 |
|---|---|---|---|
| Docker Compose | frontend、backend、Qdrant、vLLMを分離して起動 | `docker-compose.yml` | 4コンテナと4つの永続volume |
| FastAPI / Uvicorn | 認証、チャット、管理、RAG、Word編集API | `backend/app/main.py`, `backend/app/api/` | `http://localhost:8000` |
| SQLite / aiosqlite | ユーザー、会話状態、メッセージ、監査、文書管理、監視フォルダ、処理タスクを保存 | `backend/app/database.py`, `backend/migrations/init.sql` | コンテナ内 `/app/data/app.db`、Docker volume `sqlite_data` |
| Qdrant | 文書チャンクのdense/sparseベクトルと検索用payloadを保存 | `backend/app/services/qdrant.py`, `indexer.py`, `retriever.py`, `access_sync.py` | `/qdrant/storage`、volume `qdrant_data`、ホスト `127.0.0.1:6333` |
| Gemma 4 E2B | 回答生成と、意図判定・逆質問・JSON計画を作る司令塔 | `backend/app/services/llm.py`, `orchestrator.py`, `prompts/orchestrator.txt` | vLLMの `local-llm` と `hr-orchestrator` |
| vLLM | Gemma量子化ベースモデルとLoRAをOpenAI互換APIで提供 | `inference/Dockerfile`, `docker-compose.yml` | ホスト `127.0.0.1:8001/v1`、volume `llm_cache` |
| Unsloth / QLoRA | Gemma司令塔を授業用データでSFT | 設計・結果は `docs/FT_DESIGN_AND_RESULTS.md` | LoRA配置 `models/hr-orchestrator/` |
| multilingual-e5-large | 質問と文書を1024次元denseベクトルへ変換 | `backend/app/services/embedder.py`, `backend/app/config.py` | CPU実行、モデルcache `embed_cache` |
| Qdrant sparse + RRF | dense検索だけで拾いにくい固有語を補い、dense/sparse候補を融合 | `backend/app/services/sparse.py`, `retriever.py` | Qdrantのsparse vectorとfusion query |
| BGE reranker v2 m3 | 候補チャンクを質問との関連度で再順位付け | `backend/app/services/reranker.py` | CPU実行、上位候補を最終top-kへ絞る |
| Docling | PDF、DOCX、PPTXをMarkdown相当のテキストへ変換してRAG投入 | `backend/app/services/parser.py` | 監視対象 `/watched` を読み取り専用で参照 |
| pandas / openpyxl | Excelのシート、結合セル、表をテキスト化してRAG投入 | `backend/app/services/parser.py` | `.xlsx`, `.xls`, `.xlsm` の解析 |
| watchdog | 監視フォルダの追加・更新・削除を検知しインデックス処理を登録 | `backend/app/services/watcher.py`, `task_processor.py` | ホスト `volumes/watched/` → コンテナ `/watched` |
| PyJWT / bcrypt | JWTログイン、パスワードハッシュ、変更後の旧トークン失効 | `backend/app/core/security.py`, `backend/app/api/auth.py` | ユーザー情報はSQLite `users` table |
| python-docx | Word本文・表・ヘッダー・フッターの安全な置換 | `backend/app/services/docx_editor.py`, `backend/app/api/documents.py` | 編集案はRAM内に最大15分。原本・下書きをサーバー保存しない |
| Vanilla HTML/CSS/JavaScript | Claude Code風のチャット、実行計画、管理、Word編集UI | `frontend/src/index.html`, `app.js`, `style.css` | Nginxから `http://localhost:3000` で配信 |
| nginx-unprivileged | 静的UI配信とAPIリバースプロキシ | `frontend/Dockerfile`, `frontend/nginx.conf` | コンテナ8080 → ホスト3000 |

### SQLiteに何を保存しているか

SQLiteはベクトル検索用ではなく、アプリケーションの正本となる構造化データに使用する。テーブル定義は `backend/migrations/init.sql` にある。

| SQLite table | 保存内容 |
|---|---|
| `users` | ユーザー名、bcryptパスワード、アクセスレベル、管理者状態、パスワード変更時刻 |
| `conversations` | 利用者ごとの会話スレッド、タイトル、作成・更新日時 |
| `messages` | user/assistantの会話内容と、画面表示に必要な出典識別情報 |
| `watch_folders` | 監視対象フォルダ、アクセスレベル、有効状態 |
| `documents` | 文書名、状態、権限、ハッシュ、インデックス世代、チャンク数 |
| `tasks` | 文書インデックス処理キュー、試行回数、エラー |
| `audit_logs` | 質問、取得件数、応答文字数、エラー等の監査情報 |
| `admin_events` | 管理者操作の記録 |
| `schema_migrations` | DB Schema更新の適用状態 |

SQLite上の文書状態を権限の正本とし、Qdrant検索後にも `retriever.py` で再照合する。これにより、Qdrant payload更新の遅延や削除直後でも権限外・無効文書をfail-closedで除外する。

### SQLiteとQdrantの役割分担

```text
SQLite: ユーザー、権限、会話、監査、文書状態、処理タスク
   ↓ 文書ID・権限・有効状態を照合
Qdrant: 文書チャンク本文、dense/sparseベクトル、検索用payload
   ↓ 権限フィルタ済み上位候補
Gemma: 取得した根拠だけを使って回答
```

当初案のChromaDBは現在の実装では使用しておらず、実接続・権限制御・hybrid検索のためQdrantへ置き換えた。LangChainも現在の中核パイプラインでは使用せず、FastAPIサービス層で処理順序と安全条件を明示的に実装している。

### アダプターと再学習コード

- LoRAアダプター一式: `models/hr-orchestrator/`
- アダプター重み: HF Hubから `models/hr-orchestrator/adapter_model.safetensors` へ取得（Git対象外）
- tokenizer: `models/hr-orchestrator/tokenizer.json`
- モデル・データ系譜: `models/hr-orchestrator/manifest.json`
- 再学習・評価手順: `training/README.md`
- Unsloth学習: `training/train_unsloth.py`
- 実験証跡収集: `training/run_remote_experiment.sh`
- JSON Schema: `training/src/hr_assistant/schema.py`

Git clone後は `.env` に `ADAPTER_HF_REPO`を設定し、インストーラーでアダプター重みを取得・検証する。約7.58GBのGemmaベースモデルはGitへ含めず、vLLMの `llm_cache` にダウンロードする。

## Model packaging

Model weights are not baked into the container image. On first installation, vLLM
downloads the base model into the persistent `llm_cache` Docker volume. Embedding
weights use a separate `embed_cache` volume so the non-root backend can own its
cache safely. The approved
LoRA adapter metadata is stored under `models/hr-orchestrator`. The safetensors
weight is distributed from the Hugging Face Hub repository configured by
`ADAPTER_HF_REPO`; it is downloaded before Docker starts and kept locally for
subsequent restarts.

This keeps application images small and makes model upgrades auditable. It also avoids
redistributing gated model weights without the model owner's terms being accepted.

## Requirements

- Linux, or Windows 11 with Docker Desktop
- Docker Engine with Compose v2, or Docker Desktop
- NVIDIA driver and NVIDIA Container Toolkit
- NVIDIA GPU; the default profile supports 6 GB VRAM by offloading to system RAM
- enough free disk for the roughly 7.1 GB base-model file and container images
- internet access for the first model/image download

The 6 GB profile uses `CPU_OFFLOAD_GB=4` with eager execution and is slower than
full-GPU inference. Eager mode avoids unsafe CUDA-graph pinned-memory pressure on
WSL2. Set offload to `0` when the model fits in VRAM. Quantized-base plus LoRA compatibility
must be smoke-tested on the target GPU and vLLM version. For production, 12 GB or
more VRAM is recommended.

## Install

Set the adapter repository in `.env` before the first clean installation:

```dotenv
ADAPTER_HF_REPO=yuu0617/hr-orchestrator
ADAPTER_HF_REVISION=26e5631a7750c3c27d032d8fa375dc3f77917b1d
# HF_TOKEN=...  # not required for the public adapter repository
```

Use a commit hash or immutable tag for `ADAPTER_HF_REVISION` in a shared release.
Distribution rationale, failure behavior, and the collaborator handoff are documented
in [docs/ADAPTER_DISTRIBUTION.md](docs/ADAPTER_DISTRIBUTION.md).

Then run one of:

Windows PowerShell:

```powershell
.\install.ps1
```

Linux or WSL:

```bash
bash install.sh
```

If the adapter weight is absent, the installer downloads it with the Hugging Face
CLI (or the compatible Python library fallback), verifies the approved SHA-256,
creates `.env` with a random application secret, and only then pulls/builds and
starts the containers. Monitor the base-model download with:

```bash
docker compose logs -f vllm
```

When ready, open <http://localhost:3000>. The initial administrator credentials from
the upstream project must be changed immediately after first login.

## Verify and operate

Windows PowerShell:

```powershell
.\verify-install.ps1
docker compose ps
docker compose logs --tail=200
docker compose down
docker compose up -d
```

Linux or WSL:

```bash
bash verify-install.sh
docker compose ps
docker compose logs --tail=200
docker compose down
docker compose up -d
```

Do not use `docker compose down -v` unless all Qdrant, SQLite, and model-cache volumes
are intentionally being deleted.

## Model lineage

The packaged adapter is the `20260714-ood-remediation-160step` experiment. Its
submission snapshots are stored under `docs/ft-data` and `docs/ft-evidence`; the
complete primary experiment logs remain in the sibling `hr-assistant-ai` project.
The fixed human-authored OOD set was not included in training. See
`docs/FT_DESIGN_AND_RESULTS.md` for exact hashes, metrics, limitations, and paths.
