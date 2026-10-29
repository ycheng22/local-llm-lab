# Usage: .\push_dataset.ps1 llm-lab-utils   (or llm-lab-data / llm-lab-wheels)
# Creates the Kaggle dataset on first run, otherwise uploads a new version.
param([Parameter(Mandatory=$true)][string]$DatasetName)
. "$PSScriptRoot\_env.ps1"

$dir = Join-Path $PSScriptRoot "..\datasets\$DatasetName"
$meta = Join-Path $dir "dataset-metadata.json"
if (-not (Test-Path $meta)) { Write-Error "Missing $meta"; exit 1 }

$slug = (Get-Content $meta -Raw | ConvertFrom-Json).id
kaggle datasets status $slug *> $null
if ($LASTEXITCODE -eq 0) {
    Write-Host "Dataset $slug exists -> new version"
    kaggle datasets version -p $dir -m "Update $(Get-Date -Format 'yyyy-MM-dd HH:mm')" --dir-mode zip
} else {
    Write-Host "Creating dataset $slug (private)"
    kaggle datasets create -p $dir --dir-mode zip
}
