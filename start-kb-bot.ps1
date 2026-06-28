# Start KB Bot with low-memory Ollama settings (loads .env via config.py)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Host "Ollama is not installed or not on PATH." -ForegroundColor Red
    exit 1
}

# Free RAM: unload models that are not in use
ollama stop 2>$null

$model = if ($env:LLM_MODEL) { $env:LLM_MODEL } else { "tinyllama" }
Write-Host "Using Ollama model: $model" -ForegroundColor Cyan
ollama pull $model | Out-Null

if (-not (Test-Path ".\venv\Scripts\python.exe")) {
    Write-Host "venv not found. Run: python -m venv venv; .\venv\Scripts\pip install -r requirements.txt"
    exit 1
}

Write-Host "Starting KB Bot..." -ForegroundColor Green
& ".\venv\Scripts\python.exe" app.py
