$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$expectedAdapterSha = '5a8f318629bbb6fcc4f0131164ab6088299cac9eeec44a76463a32f37baa3a59'
$adapterDir = Join-Path $PSScriptRoot 'models\hr-orchestrator'

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

$adapterConfig = Join-Path $adapterDir 'adapter_config.json'
$adapterModel = Join-Path $adapterDir 'adapter_model.safetensors'
if (-not (Test-Path $adapterConfig)) { throw "Missing $adapterConfig" }
if (-not (Test-Path $adapterModel)) { throw "Missing $adapterModel" }
$actualAdapterSha = (Get-FileHash -Algorithm SHA256 $adapterModel).Hash.ToLowerInvariant()
if ($actualAdapterSha -ne $expectedAdapterSha) {
    throw 'The orchestrator adapter SHA-256 does not match the approved experiment.'
}

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
