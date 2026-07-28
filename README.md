# Local RAG Agent with HR Orchestrator

This package runs the complete local stack with Docker Compose:

- frontend: browser UI
- backend: authentication, audit logs, ingestion, retrieval, and orchestration
- qdrant: local vector database
- vllm: one quantized base model exposed as `local-llm`, plus the fine-tuned
  LoRA adapter exposed as `hr-orchestrator`

The orchestrator runs before retrieval. It either asks for missing information,
rejects unsafe or unauthorized requests, or permits the existing RAG pipeline to
continue. Document access levels are still enforced in backend code; model output
cannot raise a user permission level or disable retrieval filters.

## Model packaging

Model weights are not baked into the container image. On first installation, vLLM
downloads the base model into the persistent `llm_cache` Docker volume. Embedding
weights use a separate `embed_cache` volume so the non-root backend can own its
cache safely. The approved
LoRA adapter is stored under `models/hr-orchestrator`. After the first successful
startup, the same installation can restart without downloading weights again.

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

Copy the approved adapter files into `models/hr-orchestrator`, then run one of:

Windows PowerShell:

```powershell
.\install.ps1
```

Linux or WSL:

```bash
bash install.sh
```

The installer verifies the adapter SHA-256, creates `.env` with a random application
secret, pulls/builds the containers, and starts the first model download. Monitor it
with:

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
training and evaluation evidence remains in the sibling `hr-assistant-ai` project.
The fixed human-authored OOD set was not included in training.
