#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
docker compose config --quiet
docker compose ps

MODELS=$(curl --fail --silent http://127.0.0.1:8001/v1/models)
printf '%s' "$MODELS" | grep --quiet 'local-llm'
printf '%s' "$MODELS" | grep --quiet 'hr-orchestrator'
curl --fail --silent http://127.0.0.1:8000/health >/dev/null
printf 'OK: base model, HR orchestrator, and backend are available.\n'
