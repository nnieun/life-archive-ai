param(
    [int]$Port = 8501
)

$ErrorActionPreference = "Stop"
$RepositoryRoot = Split-Path -Parent $PSScriptRoot
$PythonPath = Join-Path $RepositoryRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $PythonPath)) {
    throw "Virtual environment not found. Complete TASK-002 first."
}

Set-Location -LiteralPath $RepositoryRoot
& $PythonPath -m streamlit run frontend/app.py --server.port $Port
exit $LASTEXITCODE
