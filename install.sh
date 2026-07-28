#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

EXPECTED_ADAPTER_SHA='5a8f318629bbb6fcc4f0131164ab6088299cac9eeec44a76463a32f37baa3a59'
ADAPTER_DIR='models/hr-orchestrator'

fail() {
  printf 'ERROR: %s\n' "$1" >&2
  exit 1
}

command -v docker >/dev/null 2>&1 || fail 'Docker is not installed.'
docker compose version >/dev/null 2>&1 || fail 'Docker Compose v2 is not installed.'
docker info >/dev/null 2>&1 || fail 'Docker daemon is not running.'

if [ ! -f "${ADAPTER_DIR}/adapter_config.json" ]; then
  fail 'adapter_config.json is missing under models/hr-orchestrator.'
fi
if [ ! -f "${ADAPTER_DIR}/adapter_model.safetensors" ]; then
  fail 'adapter_model.safetensors is missing under models/hr-orchestrator.'
fi

ACTUAL_ADAPTER_SHA=$(sha256sum "${ADAPTER_DIR}/adapter_model.safetensors" | awk '{print $1}')
if [ "${ACTUAL_ADAPTER_SHA}" != "${EXPECTED_ADAPTER_SHA}" ]; then
  fail 'The orchestrator adapter SHA-256 does not match the approved experiment.'
fi

if [ ! -f .env ]; then
  cp .env.example .env
  SECRET=$(python3 -c 'import secrets; print(secrets.token_hex(32))')
  sed -i s/^SECRET_KEY=.*/SECRET_KEY=${SECRET}/ .env
  printf 'Created .env with a random SECRET_KEY.\n'
fi

mkdir -p volumes/watched
printf 'Pulling containers and model runtime...\n'
docker compose pull
docker compose build
printf 'Starting services. The first run downloads LLM and embedding weights.\n'
docker compose up -d
printf 'Installation started. Follow model download with: docker compose logs -f vllm\n'
printf 'Web UI: http://localhost:3000\n'
