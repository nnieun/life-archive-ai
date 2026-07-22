param(
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"
$RepositoryRoot = Split-Path -Parent $PSScriptRoot
$PythonPath = Join-Path $RepositoryRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $PythonPath)) {
    throw "Virtual environment not found. Complete TASK-002 first."
}

Set-Location -LiteralPath $RepositoryRoot
& $PythonPath -m uvicorn backend.app.main:app --reload --host $HostAddress --port $Port
exit $LASTEXITCODE
