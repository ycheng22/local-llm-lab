# Dot-sourced by the other scripts: puts the repo .venv (which has the kaggle CLI) first on PATH.
$venvScripts = Join-Path $PSScriptRoot "..\..\.venv\Scripts"
if (Test-Path (Join-Path $venvScripts "kaggle.exe")) { $env:PATH = "$((Resolve-Path $venvScripts).Path);$env:PATH" }
if (-not (Get-Command kaggle -ErrorAction SilentlyContinue)) { Write-Error "kaggle CLI not found. Run: pip install kaggle kagglehub"; exit 1 }
