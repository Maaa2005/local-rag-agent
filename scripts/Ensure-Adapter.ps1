param(
    [string]$AdapterDir,
    [string]$Repository,
    [string]$Revision = 'main',
    [string]$ExpectedSha = '5a8f318629bbb6fcc4f0131164ab6088299cac9eeec44a76463a32f37baa3a59'
)

$ErrorActionPreference = 'Stop'

if (-not $AdapterDir) {
    $AdapterDir = Join-Path (Split-Path $PSScriptRoot -Parent) 'models\hr-orchestrator'
}

$adapterConfig = Join-Path $AdapterDir 'adapter_config.json'
$adapterModel = Join-Path $AdapterDir 'adapter_model.safetensors'

if (-not (Test-Path $adapterConfig) -or -not (Test-Path $adapterModel)) {
    if (-not $Repository) {
        throw 'The LoRA adapter is missing. Set ADAPTER_HF_REPO in .env to its Hugging Face Hub repository (for example, organization/model-name).'
    }

    New-Item -ItemType Directory -Force $AdapterDir | Out-Null
    Write-Host "Downloading approved LoRA adapter from Hugging Face Hub: $Repository@$Revision"

    $hf = Get-Command hf -ErrorAction SilentlyContinue
    $legacyHf = Get-Command huggingface-cli -ErrorAction SilentlyContinue
    if ($hf) {
        & $hf.Source download $Repository --revision $Revision --local-dir $AdapterDir
    } elseif ($legacyHf) {
        & $legacyHf.Source download $Repository --revision $Revision --local-dir $AdapterDir
    } else {
        $python = Get-Command python -ErrorAction SilentlyContinue
        if (-not $python) {
            $python = Get-Command python3 -ErrorAction SilentlyContinue
        }
        if (-not $python) {
            throw 'No Hugging Face downloader is available. Install it with: python -m pip install "huggingface_hub[cli]"'
        }
        & $python.Source -c 'import huggingface_hub' 2>$null
        if ($LASTEXITCODE -ne 0) {
            throw 'No Hugging Face downloader is available. Install it with: python -m pip install "huggingface_hub[cli]"'
        }

        $env:ADAPTER_DOWNLOAD_REPO = $Repository
        $env:ADAPTER_DOWNLOAD_REVISION = $Revision
        $env:ADAPTER_DOWNLOAD_DIR = $AdapterDir
        try {
            & $python.Source -c @'
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id=os.environ["ADAPTER_DOWNLOAD_REPO"],
    revision=os.environ["ADAPTER_DOWNLOAD_REVISION"],
    local_dir=os.environ["ADAPTER_DOWNLOAD_DIR"],
    token=os.environ.get("HF_TOKEN") or None,
)
'@
        } finally {
            Remove-Item Env:ADAPTER_DOWNLOAD_REPO -ErrorAction SilentlyContinue
            Remove-Item Env:ADAPTER_DOWNLOAD_REVISION -ErrorAction SilentlyContinue
            Remove-Item Env:ADAPTER_DOWNLOAD_DIR -ErrorAction SilentlyContinue
        }
    }
    if ($LASTEXITCODE -ne 0) {
        throw 'Hugging Face adapter download failed.'
    }
}

if (-not (Test-Path $adapterConfig)) {
    throw "adapter_config.json is still missing after the Hugging Face download: $adapterConfig"
}
if (-not (Test-Path $adapterModel)) {
    throw "adapter_model.safetensors is still missing after the Hugging Face download: $adapterModel"
}

$actualSha = (Get-FileHash -Algorithm SHA256 $adapterModel).Hash.ToLowerInvariant()
if ($actualSha -ne $ExpectedSha.ToLowerInvariant()) {
    throw "The orchestrator adapter SHA-256 does not match the approved experiment. Expected $ExpectedSha, got $actualSha."
}

Write-Host "LoRA adapter verified: $actualSha"
