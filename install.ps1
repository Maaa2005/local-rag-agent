$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$expectedAdapterSha = '5a8f318629bbb6fcc4f0131164ab6088299cac9eeec44a76463a32f37baa3a59'
$adapterDir = Join-Path $PSScriptRoot 'models\hr-orchestrator'

if (-not (Test-Path '.env')) {
    Copy-Item '.env.example' '.env'
    $secretBytes = New-Object byte[] 32
    $rng = [Security.Cryptography.RandomNumberGenerator]::Create()
    try {
        $rng.GetBytes($secretBytes)
    } finally {
        $rng.Dispose()
    }
    $secret = -join ($secretBytes | ForEach-Object { $_.ToString('x2') })
    $content = [IO.File]::ReadAllText((Join-Path $PSScriptRoot '.env'))
    $content = $content -replace '(?m)^SECRET_KEY=.*$', "SECRET_KEY=$secret"
    [IO.File]::WriteAllText(
        (Join-Path $PSScriptRoot '.env'),
        $content,
        [Text.UTF8Encoding]::new($false)
    )
    Write-Host 'Created .env with a random SECRET_KEY.'
}

function Get-DotEnvValue([string]$Name) {
    $line = Get-Content '.env' | Where-Object { $_ -match "^$([regex]::Escape($Name))=" } |
        Select-Object -Last 1
    if (-not $line) { return '' }
    return ($line.Substring($line.IndexOf('=') + 1)).Trim().Trim('"')
}

$adapterRepo = if ($env:ADAPTER_HF_REPO) {
    $env:ADAPTER_HF_REPO
} else {
    Get-DotEnvValue 'ADAPTER_HF_REPO'
}
$adapterRevision = if ($env:ADAPTER_HF_REVISION) {
    $env:ADAPTER_HF_REVISION
} else {
    Get-DotEnvValue 'ADAPTER_HF_REVISION'
}
if (-not $adapterRevision) { $adapterRevision = 'main' }
if (-not $env:HF_TOKEN) {
    $env:HF_TOKEN = Get-DotEnvValue 'HF_TOKEN'
}

& (Join-Path $PSScriptRoot 'scripts\Ensure-Adapter.ps1') `
    -AdapterDir $adapterDir `
    -Repository $adapterRepo `
    -Revision $adapterRevision `
    -ExpectedSha $expectedAdapterSha

$dockerCommand = Get-Command docker -ErrorAction SilentlyContinue
if ($dockerCommand) {
    $docker = $dockerCommand.Source
} else {
    $docker = 'C:\Program Files\Docker\Docker\resources\bin\docker.exe'
}
if (-not (Test-Path $docker)) {
    throw 'Docker Desktop is not installed.'
}
$env:PATH = "$(Split-Path $docker);$env:PATH"

& $docker compose version | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'Docker Compose v2 is not available.' }
& $docker info | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'Docker Desktop is not running.' }

New-Item -ItemType Directory -Force 'volumes\watched' | Out-Null
Write-Host 'Pulling containers and model runtime...'
& $docker compose pull
if ($LASTEXITCODE -ne 0) { throw 'docker compose pull failed.' }
& $docker compose build
if ($LASTEXITCODE -ne 0) { throw 'docker compose build failed.' }
Write-Host 'Starting services. The first run downloads LLM and embedding weights.'
& $docker compose up -d
if ($LASTEXITCODE -ne 0) { throw 'docker compose up failed.' }
Write-Host 'Installation started. Follow the model download with: docker compose logs -f vllm'
Write-Host 'Web UI: http://localhost:3000'
