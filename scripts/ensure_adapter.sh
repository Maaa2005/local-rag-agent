#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ADAPTER_DIR="${ADAPTER_DIR:-${ROOT_DIR}/models/hr-orchestrator}"
ADAPTER_HF_REPO="${ADAPTER_HF_REPO:-}"
ADAPTER_HF_REVISION="${ADAPTER_HF_REVISION:-main}"
EXPECTED_ADAPTER_SHA="${EXPECTED_ADAPTER_SHA:-5a8f318629bbb6fcc4f0131164ab6088299cac9eeec44a76463a32f37baa3a59}"

fail() {
  printf 'ERROR: %s\n' "$1" >&2
  exit 1
}

download_adapter() {
  [ -n "${ADAPTER_HF_REPO}" ] || fail \
    'The LoRA adapter is missing. Set ADAPTER_HF_REPO in .env to its Hugging Face Hub repository (for example, organization/model-name).'

  mkdir -p "${ADAPTER_DIR}"
  printf 'Downloading approved LoRA adapter from Hugging Face Hub: %s@%s\n' \
    "${ADAPTER_HF_REPO}" "${ADAPTER_HF_REVISION}"

  if command -v hf >/dev/null 2>&1; then
    hf download "${ADAPTER_HF_REPO}" \
      --revision "${ADAPTER_HF_REVISION}" \
      --local-dir "${ADAPTER_DIR}"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "${ADAPTER_HF_REPO}" \
      --revision "${ADAPTER_HF_REVISION}" \
      --local-dir "${ADAPTER_DIR}"
  elif command -v python3 >/dev/null 2>&1 &&
       python3 -c 'import huggingface_hub' >/dev/null 2>&1; then
    ADAPTER_DOWNLOAD_DIR="${ADAPTER_DIR}" python3 - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["ADAPTER_HF_REPO"],
    revision=os.environ["ADAPTER_HF_REVISION"],
    local_dir=os.environ["ADAPTER_DOWNLOAD_DIR"],
    token=os.environ.get("HF_TOKEN") or None,
)
PY
  else
    fail 'No Hugging Face downloader is available. Install it with: python3 -m pip install --user "huggingface_hub[cli]"'
  fi
}

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  elif command -v python3 >/dev/null 2>&1; then
    python3 -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$1"
  else
    fail 'SHA-256 verification requires sha256sum, shasum, or python3.'
  fi
}

if [ ! -f "${ADAPTER_DIR}/adapter_config.json" ] ||
   [ ! -f "${ADAPTER_DIR}/adapter_model.safetensors" ] ||
   [ ! -f "${ADAPTER_DIR}/LICENSE" ] ||
   [ ! -f "${ADAPTER_DIR}/NOTICE" ]; then
  download_adapter
fi

[ -f "${ADAPTER_DIR}/adapter_config.json" ] ||
  fail 'adapter_config.json is still missing after the Hugging Face download.'
[ -f "${ADAPTER_DIR}/adapter_model.safetensors" ] ||
  fail 'adapter_model.safetensors is still missing after the Hugging Face download.'
[ -f "${ADAPTER_DIR}/LICENSE" ] ||
  fail 'LICENSE is still missing after the Hugging Face download.'
[ -f "${ADAPTER_DIR}/NOTICE" ] ||
  fail 'NOTICE is still missing after the Hugging Face download.'

ACTUAL_ADAPTER_SHA=$(sha256_file "${ADAPTER_DIR}/adapter_model.safetensors")
if [ "${ACTUAL_ADAPTER_SHA}" != "${EXPECTED_ADAPTER_SHA}" ]; then
  fail "The orchestrator adapter SHA-256 does not match the approved experiment. Expected ${EXPECTED_ADAPTER_SHA}, got ${ACTUAL_ADAPTER_SHA}."
fi

printf 'LoRA adapter verified: %s\n' "${ACTUAL_ADAPTER_SHA}"
