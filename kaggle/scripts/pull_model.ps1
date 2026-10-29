# Usage: .\pull_model.ps1 llm-lab-sft-checkpoint v1
param([Parameter(Mandatory=$true)][string]$ModelName, [string]$Version = "v1")
. "$PSScriptRoot\_env.ps1"
python -c "import kagglehub; print(kagglehub.model_download('ycheng22/$ModelName/pyTorch/$Version'))"
