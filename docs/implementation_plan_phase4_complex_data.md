# Phase 4: Complex Data Mining and Ablation Plan

The goal is to replace the toy dataset with a complex coding dataset (like MBPP or HumanEval), mine hard examples from it, and run the ablation GRPO studies on a subset, ensuring we don't overwrite any artifacts from the previous toy dataset run.

## Proposed Changes

### Data Generation
#### [NEW] src/llm_lab/data/generator_complex.py
- Create a new script that fetches and parses the `mbpp` dataset (or `openai_humaneval`) using the Hugging Face `datasets` library.
- Map the dataset into our `TaskSchema` format (id, prompt, starter_code, test cases, reference_solution).
- Save to new file paths (e.g., `data/train_complex.jsonl`) to ensure we do not overwrite the toy dataset (`data/train_5k.jsonl`).

### Configuration
#### [NEW] configs/mining_complex.yaml
- Copy `configs/mining.yaml` but change the dataset path to the newly generated `data/train_complex_500.jsonl` (a 500-task subset for ablation).
- Set the `output_dir` to `data/complex_mined/` to avoid overwriting existing data.
- Keep the model pointing to the Phase 2 SFT checkpoint.

### Notebooks
#### [NEW] notebooks/mining_complex.ipynb
- A new interactive notebook that runs the data mining script (`miner.py`) utilizing `configs/mining_complex.yaml`. 
- Allows you to monitor the mining progress live without altering the existing `mining.ipynb` notebook.

## Time Estimates

**Training on 500 tasks (GRPO):**
- **Epoch Size**: 500 tasks.
- **Batch Size**: 4 tasks per step (based on Phase 3: `num_generations: 4`, `gradient_accumulation_steps: 4`).
- **Steps per Epoch**: 500 tasks / 4 tasks per step = 125 steps.
- **Speed**: ~50 seconds per step (derived from your Phase 3 run of 3,250 steps in 45 hours).
- **Estimated Time (1 Epoch)**: ~1.75 hours (125 steps * 50s = 6,250s).
- **Estimated Time (2 Epochs)**: **~3.5 hours**.
- **Conclusion**: Training for 2 epochs on 500 tasks is a highly feasible 3.5-hour experiment.

**Mining on 500 tasks (for context):**
- As seen in `notebooks/mining.ipynb`, running the verifier to generate 4 candidate solutions per task on 500 tasks takes **~13.6 hours** (~98 seconds per task). 

## Open Questions

> [!IMPORTANT]
> To fetch MBPP or HumanEval, I plan to use the Hugging Face `datasets` library in `generator_complex.py`. Do you approve of adding `datasets` to the environment, or do you prefer I write a script to download the raw JSON files directly using `urllib` to avoid new dependencies?

## Verification Plan

### Manual Verification
- We will generate a 500-task subset and manually inspect the `.jsonl` to ensure the task format matches `TaskSchema`.
- We will verify that no existing files in `data/` or `configs/` were modified or overwritten.
