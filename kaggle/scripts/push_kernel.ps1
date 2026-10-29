# Usage: .\push_kernel.ps1 03-baseline-evaluation
# Pushes kernels\<name>\ to Kaggle and starts a run. Then poll with: kaggle kernels status ycheng22/<name>
param([Parameter(Mandatory=$true)][string]$KernelName)
. "$PSScriptRoot\_env.ps1"

$dir = Join-Path $PSScriptRoot "..\kernels\$KernelName"
if (-not (Test-Path $dir)) {
    Write-Host "Kernel dir not found. Available:"
    Get-ChildItem (Join-Path $PSScriptRoot "..\kernels") -Directory | ForEach-Object { "  " + $_.Name }
    exit 1
}
kaggle kernels push -p $dir
Write-Host "Status:  kaggle kernels status ycheng22/$KernelName"
Write-Host "Web:     https://www.kaggle.com/code/ycheng22/$KernelName"
