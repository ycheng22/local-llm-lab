# Usage: .\sync_all.ps1   -> pushes utils + data (wheels are pushed separately, see README)
& "$PSScriptRoot\push_dataset.ps1" llm-lab-utils
& "$PSScriptRoot\push_dataset.ps1" llm-lab-data
