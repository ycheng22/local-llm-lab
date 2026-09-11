# Phase 7: Research Dashboard + Final Report Implementation Plan

Phase 7 aggregates all experiment results to produce the final artifacts, plots, and a comprehensive `experiment_log.csv` as per the project requirements.

## Proposed Changes

We will create a centralized reporting script `src/llm_lab/reporting/dashboard.py` which crawls the `experiments/` directory for all `summary.json` files and GRPO training logs, and then generates the required CSV and plots. We will use `pandas` and `matplotlib` for this.

### `src/llm_lab/reporting/dashboard.py` (New Script)
#### [NEW] `src/llm_lab/reporting/dashboard.py`
This script will:
- Glob all `summary.json` files in `experiments/`.
- Parse each summary to extract model, hyperparams, metrics (Pass@1, Pass@4, Pass@8, OOD Pass@1), hardware stats, and training times.
- Export everything into a consolidated `reports/experiment_log.csv`.
- Generate the following figures in `reports/figures/`:
  - `pass_1_vs_exp.png`: Bar chart of Pass@1 across experiments.
  - `pass_4_vs_exp.png`: Bar chart of Pass@4 across experiments.
  - `ood_pass_1_vs_exp.png`: Bar chart of OOD Pass@1 (if metric is found).
  - `roi_gain_per_gpu_hour.png`: Bar chart of ROI (`(Pass@1 - Base_Pass@1) / gpu_hours`).
  - `hardware_util.png`: Plot of peak VRAM across experiments.

### `src/llm_lab/reporting/plots.py` (Modify)
#### [MODIFY] `src/llm_lab/reporting/plots.py`
- Expose the functions so they can be imported and executed by the main `dashboard.py` script.
- Support reward vs training step plotting from GRPO's `trainer_state.json` or logs if they exist.

## Verification Plan

### Manual Verification
1. Run `python -m llm_lab.reporting.dashboard`.
2. Verify that `reports/experiment_log.csv` is generated and contains rows for all existing experiments (`exp00_eval`, `exp01_sft_v1_eval`, etc.).
3. Verify that `reports/figures/` is populated with the requested PNG plots.

## Open Questions

> [!IMPORTANT]
> The phase 7 requirements mention tracking `OOD Pass@1` and `reward vs training step`. Depending on how these were logged during Phase 3/4/6 (e.g., whether OOD Pass@1 was computed explicitly, or if reward curves were saved to a specific log file like `trainer_state.json`), I will try to extract them if they are available in the JSON files. Should I fallback gracefully and omit them in the plots if those specific logs are missing for some experiments?
