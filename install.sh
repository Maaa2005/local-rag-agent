#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

EXPECTED_ADAPTER_SHA='5a8f318629bbb6fcc4f0131164ab6088299cac9eeec44a76463a32f37baa3a59'
ADAPTER_DIR="${ADAPTER_DIR:-models/hr-orchestrator}"

fail() {
  printf 'ERROR: %s\n' "$1" >&2
  exit 1
}

if [ ! -f .env ]; then
  cp .env.example .env
  command -v python3 >/dev/null 2>&1 || fail 'python3 is required to generate SECRET_KEY.'
  SECRET=$(python3 -c 'import secrets; print(secrets.token_hex(32))')
  sed -i "s/^SECRET_KEY=.*/SECRET_KEY=${SECRET}/" .env
  printf 'Created .env with a random SECRET_KEY.\n'
fi

dotenv_value() {
  awk -F= -v key="$1" '$1 == key { value=substr($0, index($0, "=") + 1) } END { print value }' .env |
    tr -d '\r' |
    sed -e 's/^"//' -e 's/"$//'
}

ADAPTER_HF_REPO="${ADAPTER_HF_REPO:-$(dotenv_value ADAPTER_HF_REPO)}"
ADAPTER_HF_REVISION="${ADAPTER_HF_REVISION:-$(dotenv_value ADAPTER_HF_REVISION)}"
ADAPTER_HF_REVISION="${ADAPTER_HF_REVISION:-main}"
HF_TOKEN="${HF_TOKEN:-$(dotenv_value HF_TOKEN)}"
export ADAPTER_DIR ADAPTER_HF_REPO ADAPTER_HF_REVISION EXPECTED_ADAPTER_SHA HF_TOKEN

"$(pwd)/scripts/ensure_adapter.sh"

command -v docker >/dev/null 2>&1 || fail 'Docker is not installed.'
docker compose version >/dev/null 2>&1 || fail 'Docker Compose v2 is not installed.'
docker info >/dev/null 2>&1 || fail 'Docker daemon is not running.'

mkdir -p volumes/watched
printf 'Pulling containers and model runtime...\n'
docker compose pull
docker compose build
printf 'Starting services. The first run downloads LLM and embedding weights.\n'
docker compose up -d
printf 'Installation started. Follow model download with: docker compose logs -f vllm\n'
printf 'Web UI: http://localhost:3000\n'
