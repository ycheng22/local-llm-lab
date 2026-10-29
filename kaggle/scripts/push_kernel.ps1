# Usage: .\push_kernel.ps1 01-env-inference-benchmark -Accelerator NvidiaTeslaT4 -TimeoutSeconds 3600
#   (name = folder under kaggle\kernels)
# Pushes the kernel and starts a run. NOTE: the metadata "title" must slugify to the id
# (e.g. title "03 Baseline Eval" -> id ycheng22/03-baseline-eval), otherwise Kaggle uses the title's slug.
param(
    [Parameter(Mandatory=$true)][string]$KernelName,
    [string]$Accelerator = "",      # e.g. NvidiaTeslaT4 (= T4 x2). Empty = whatever kernel-metadata.json says
    [int]$TimeoutSeconds = 0        # hard cap on run time (cost guard). 0 = Kaggle default
)
. "$PSScriptRoot\_env.ps1"

$dir = Join-Path $PSScriptRoot "..\kernels\$KernelName"
if (-not (Test-Path $dir)) {
    Write-Host "Kernel dir not found. Available:"
    Get-ChildItem (Join-Path $PSScriptRoot "..\kernels") -Directory | ForEach-Object { "  " + $_.Name }
    exit 1
}
$id = (Get-Content (Join-Path $dir "kernel-metadata.json") -Raw | ConvertFrom-Json).id
$extra = @()
if ($Accelerator) { $extra += @("--accelerator", $Accelerator) }
if ($TimeoutSeconds -gt 0) { $extra += @("-t", $TimeoutSeconds) }
kaggle kernels push -p $dir @extra
Write-Host "Status:  kaggle kernels status $id"
Write-Host "Web:     https://www.kaggle.com/code/$id"
