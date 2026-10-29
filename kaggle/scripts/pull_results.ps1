# Usage: .\pull_results.ps1 03-baseline-evaluation
# Downloads the kernel's /kaggle/working output into experiments\kaggle_outputs\<name> (never overwrites: timestamped).
param([Parameter(Mandatory=$true)][string]$KernelName)
. "$PSScriptRoot\_env.ps1"

$root = (Get-Item $PSScriptRoot).Parent.Parent.FullName
$out = Join-Path $root ("experiments\kaggle_outputs\{0}_{1}" -f $KernelName, (Get-Date -Format 'yyyyMMdd_HHmm'))
New-Item -ItemType Directory -Force -Path $out | Out-Null
kaggle kernels output ycheng22/$KernelName -p $out
Get-ChildItem $out -Recurse -File | ForEach-Object { "{0}  ({1:N1} KB)" -f $_.FullName.Substring($out.Length), ($_.Length/1KB) }
