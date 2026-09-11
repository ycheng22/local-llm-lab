# Walkthrough: Phase 7 (Research Dashboard)

Phase 7 successfully aggregates the artifacts of all the prior experiments into a consolidated research dashboard, helping us to answer our research question about training efficiency and scaling.

## What Was Accomplished

1. **Reporting Scripts Added**: 
    - Created `src/llm_lab/reporting/dashboard.py` which crawls all `summary.json` and metric outputs from the `experiments/` directory.
    - Updated `src/llm_lab/reporting/plots.py` with a suite of dynamic plotting utilities to graph things like Pass@1, hardware utilization, and reward curves.

2. **Data Artifacts Generated**:
    - **`reports/experiment_log.csv`**: Aggregated tabular data representing all iterations of tests, baseline checks, and agents.

3. **Plots Generated**:
    - `pass_1_vs_exp.png`: Pass@1 progression across evaluated checkpoints.
    - `pass_4_vs_exp.png`: Pass@4 progression.
    - `hardware_util.png`: Peak VRAM measurements validating the 8GB ceiling constraint.
    - `reward_vs_step.png`: GRPO training progression directly parsed from `trainer_state.json`.

> [!NOTE]
> Metrics such as OOD Pass@1 and the ROI Capability Gain plots were dynamically skipped by the dashboard script since current log files didn't have active `gpu_hours` recordings or separated OOD sets logged in their `summary.json` files yet. The pipeline is built to automatically pick these up once later full training cycles track these fields.

## Research Dashboard Notebook

I also generated a Jupyter Notebook [notebooks/research_dashboard.ipynb](file:///d:/Github_Clones/local-llm-lab/notebooks/research_dashboard.ipynb) which puts everything together. When run, it programmatically regenerates all of the metrics and renders the CSV dataset and all `reports/figures/` images inline.

You can now open this notebook and run all cells to get a bird's eye view of the entire Local LLM Lab progression!
