$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$dockerCommand = Get-Command docker -ErrorAction SilentlyContinue
if ($dockerCommand) {
    $docker = $dockerCommand.Source
} else {
    $docker = 'C:\Program Files\Docker\Docker\resources\bin\docker.exe'
}
if (-not (Test-Path $docker)) { throw 'Docker Desktop is not installed.' }
$env:PATH = "$(Split-Path $docker);$env:PATH"

& $docker compose config --quiet
if ($LASTEXITCODE -ne 0) { throw 'Docker Compose configuration is invalid.' }
& $docker compose ps

$models = Invoke-RestMethod -Uri 'http://127.0.0.1:8001/v1/models' -TimeoutSec 15
$modelIds = @($models.data | ForEach-Object { $_.id })
if ('local-llm' -notin $modelIds) { throw 'Base model local-llm is unavailable.' }
if ('hr-orchestrator' -notin $modelIds) { throw 'LoRA model hr-orchestrator is unavailable.' }

$health = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/health' -TimeoutSec 15
Write-Host "OK: base model, HR orchestrator, and backend are available. Backend health: $($health | ConvertTo-Json -Compress)"
