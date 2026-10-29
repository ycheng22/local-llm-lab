# Usage: .\push_kernel.ps1 00-build-wheels     (name = folder under kaggle\kernels)
# Pushes the kernel and starts a run. NOTE: the metadata "title" must slugify to the id
# (e.g. title "03 Baseline Eval" -> id ycheng22/03-baseline-eval), otherwise Kaggle uses the title's slug.
param([Parameter(Mandatory=$true)][string]$KernelName)
. "$PSScriptRoot\_env.ps1"

$dir = Join-Path $PSScriptRoot "..\kernels\$KernelName"
if (-not (Test-Path $dir)) {
    Write-Host "Kernel dir not found. Available:"
    Get-ChildItem (Join-Path $PSScriptRoot "..\kernels") -Directory | ForEach-Object { "  " + $_.Name }
    exit 1
}
$id = (Get-Content (Join-Path $dir "kernel-metadata.json") -Raw | ConvertFrom-Json).id
kaggle kernels push -p $dir
Write-Host "Status:  kaggle kernels status $id"
Write-Host "Web:     https://www.kaggle.com/code/$id"
