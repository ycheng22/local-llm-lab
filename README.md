# Local LLM Lab

**Research Question**

> Can small, consumer-hardware-scale post-training methods improve the verifiable coding capability of a 2B language model, and which intervention gives the highest capability gain per unit compute?

## Interventions

1. SFT
2. RLVR
3. Hard-example mining
4. Test-time compute
5. Tool-use

## Primary Metrics

- Capability: Pass@1, Pass@4, Pass@8, OOD Pass@1
- Efficiency: Capability Gain / GPU Hour

## Progress & Metrics

> **Erratum (found during the Kaggle migration, see [Kaggle Migration](#kaggle-migration-phases-0-3-re-run-on-free-t4x2-with-corrections)):** the local MBPP experiments in Phases 4-6 used a prompt that did not tell the model the name of the function the tests call (only 14 of 420 prompts mentioned it). The "420/420 impossible" mining result and the Best-of-16 / agent numbers below therefore partly measure prompt format, not model capability. With the corrected prompt the same base model scores 40.6% Pass@1 instead of 2.9%. The synthetic benchmark used in Phases 1-3 is also saturated (4 prompt templates, base model 99.6%). Phase 4-6 numbers below are kept as the historical record and are being re-measured.

### Phase 0: Project Scaffolding + Environment + Model Loading
**Status:** Completed

**Tasks Finished (Based on Implementation Plan):**
- Created core project structure (`src/llm_lab`, `configs`, `experiments`, `tests`).
- Created `pyproject.toml` and installed Python ecosystem (`torch 12.4`, `transformers`, `trl`, `peft`, `bitsandbytes`).
- Created diagnostic script (`doctor.py`) and smoke tests (`tests/test_smoke.py`).
- Downloaded `Qwen/Qwen3.5-2B` Hugging Face weights via `.cache/huggingface/hub` (using `hf-mirror.com`).
- Executed `configs/baseline.yaml` via inference script (`generate.py`).

**Metrics & Analysis:**
- **Hardware:** NVIDIA GeForce RTX 2060 SUPER (8GB VRAM), 48GB System RAM, CUDA 13.2 (Driver) / 12.4 (PyTorch).
- **Peak VRAM (Inference):** `1.71 GB` (with `bitsandbytes` NF4 quantization).
- **Latency (Inference):** `33.93s` for 384 tokens (fibonacci raw text generation).
- **Analysis:** The `Qwen3.5-2B` model fits comfortably within the 8GB limit when loaded in 4-bit precision, leaving over 6GB of VRAM available for training activations and batch sizes during the SFT and GRPO phases. The Hugging Face safetensors format has been cached properly to support the upcoming `trl` fine-tuning pipeline.

### Phase 1: Dataset, Verifier & Baseline Evaluation
**Status:** Completed

**Tasks Finished (Based on Implementation Plan):**
- Dataset Layer: Implemented Pydantic `TaskSchema`, JSONL loaders, dataset validators, and generated a synthetic 100-problem Toy Dataset.
- Verifier Layer: Implemented `DockerVerifier` to securely execute LLM-generated candidate code inside ephemeral, network-less containers using standard AWS lambda python images to bypass registry blocks.
- Evaluation Layer: Built `metrics.py` for Pass@K calculation and `evaluator.py` to run inference and test against the verifier.
- Reporting Layer: Implemented `ExperimentTracker` to log `summary.json` and per-task code generations.
- Executed `configs/eval.yaml` mini end-to-end loop for Phase 1 verification.

**Metrics & Analysis (Toy Dataset Mini-Evaluation):**
- **Pass@1 / Pass@2:** `1.0` (on toy arithmetic/string problems)
- **Timeout Rate:** `0.0%`
- **Syntax Error Rate:** `0.0%`
- **Mean Verification/Generation Latency:** `2.40 sec`
- **Peak VRAM:** `1.72 GB`
- **Analysis:** The evaluation pipeline and code execution sandbox works flawlessly on the host machine. The AWS Python image successfully sandboxes the code without triggering local execution vulnerabilities or Windows pathing issues. Memory footprint remains stable across generations. The setup is ready for scaling up to standard datasets like MBPP or HumanEval.

### Phase 2: Supervised Fine-Tuning (SFT) & Evaluation
**Status:** Completed

**Tasks Finished (Based on Implementation Plan):**
- Generated synthetic datasets (`train_5k.jsonl` with 5,000 tasks and `dev_500.jsonl` with 500 tasks) for scaling up.
- Configured QLoRA (4-bit NF4 with LoRA adapters) on `Qwen/Qwen3.5-2B` using `trl.SFTTrainer` and gradient checkpointing.
- Implemented `ProgressLoggingCallback` to stream live VRAM, speed, and loss directly into Jupyter notebooks without OS pipe buffering delays.
- Completed the full SFT pipeline run on an 8GB NVIDIA GPU over 5,000 tasks.
- Evaluated the resulting `checkpoint-final` LoRA weights on the frozen test set to calculate Pass@k metrics.

**Metrics & Analysis:**
- **Training Time:** `3.04 GPU hours`
- **Peak Training VRAM:** `3.97 GB` (effective batch size 8)
- **Evaluation Time:** `4.2 hours` (800 sequential generations)
- **Peak Eval VRAM:** `1.76 GB`
- **Pass@1 / Pass@4 / Pass@8:** `1.0` (100% on synthetic test set)
- **Timeout Rate & Syntax Error Rate:** `0.0%`
- **Analysis:** SFT fits very comfortably within the 8GB limit, capping at ~4GB VRAM. This proves that we can train 2B parameter models locally on consumer hardware. The model easily learned the synthetic task format, yielding perfect evaluation scores. The evaluation phase runs strictly sequentially right now, taking over 4 hours; future phases could implement batched generation via `num_return_sequences` to cut evaluation time by an order of magnitude.

### Phase 3: Group Relative Policy Optimization (GRPO)
**Status:** Training Completed (Stopped Early), Evaluation In Progress

**Tasks Finished (Based on Implementation Plan):**
- Configured GRPO (4-bit NF4 with LoRA adapters) on the SFT checkpoint using `trl.GRPOTrainer`.
- Implemented `DockerVerifier` as the custom reward function to grant `1.0` for code that passes all unit tests within the sandbox container.
- Resolved integration issues with `trl.GRPOTrainer` and Qwen tokenization configuration.
- Completed training run for 3,250 optimizer steps on the 5,000-task dataset. 
- Early stopping applied: Reached >2 effective epochs of training on the 5k dataset (~45 hours runtime).

**Metrics & Analysis (Training):**
- **Training Time (up to step 3250):** `~45 GPU hours`
- **Total Estimated Training Time:** `~197 hours` (15,000 steps)
- **Peak Training VRAM:** `5.11 GB` (with `gradient_accumulation_steps=4`, generating 4 candidates per step)
- **Generation Speed:** `~54 seconds` per step (producing 16 code snippets + running 16 sandbox tests + backward pass).
- **Mean Reward Achieved:** `1.000` (on the training batch at step 3250)
- **Analysis:** GRPO is computationally very heavy due to the online generation and sandbox execution required at every step. On a consumer 8GB GPU, full training on a 5,000 task dataset scales to multiple days. We halted the training at step 3250 as it represents >2 full epochs of data (13,000 tasks processed), which is sufficient to observe policy optimization.

**Metrics & Analysis (Evaluation on OOD Test Set):**
- **Pass@1 / Pass@4 / Pass@8:** `1.0` (100% on synthetic test set)
- **Timeout Rate & Syntax Error Rate:** `0.0%`
- **Mean Generation/Verification Latency:** `46.25 sec` (batch of 8)
- **Peak Eval VRAM:** `1.76 GB`
- **Analysis:** Even after early stopping, the GRPO model successfully retained 100% Pass@k performance on the held-out test set (`test.jsonl`). Due to the simplicity of the synthetic 100-problem test set, both SFT and GRPO maxed out the evaluation score. The next phase (Hard-example mining or harder benchmarks) will be required to measure the true delta in reasoning capabilities provided by GRPO.

### Phase 4: Hard-Example Mining (Ablation Study)
**Status:** Mining Completed (Training Skipped due to dataset difficulty)

**Tasks Finished (Based on Implementation Plan):**
- Migrated dataset pipeline to use the `google-research-datasets/mbpp` (sanitized) complex Python coding benchmark.
- Filtered and formatted 420 MBPP tasks into `TaskSchema` format.
- Executed the `mining_complex.yaml` job using the Phase 2 SFT checkpoint to generate 4 solutions per task.
- Classified the 420 tasks into `solved` (≥3/4), `borderline` (2/4), `hard` (1/4), and `impossible` (0/4).

**Metrics & Analysis (Data Composition & Mining):**
- **Total Tasks Processed:** `420`
- **Samples per Task (n):** `4`
- **Classification Results:**
  - `solved`: 0 (0%)
  - `borderline`: 0 (0%)
  - `hard`: 0 (0%)
  - `impossible`: 420 (100%)
- **Analysis:** The `Qwen3.5-2B` model failed completely on the MBPP complex coding dataset, scoring 0% Pass@4 across all 420 tasks. Because every single task fell into the `impossible` category, it is mathematically impossible to construct the `hard` and `borderline` ablation datasets required for the Phase 4 GRPO curriculum training. This is a crucial finding: a 2B parameter model cannot bootstrap its own capabilities via Hard Example Mining if the dataset is too far outside its distribution. It requires at least a non-zero Pass@K rate to extract learning signals. We will skip the GRPO v2 training on this dataset and proceed directly to Phase 5 (Test-Time Compute Scaling) to observe if massive scale sampling (K=16+) can uncover any hidden capabilities.

### Phase 5: Test-Time Compute Scaling
**Status:** Completed

**Tasks Finished (Based on Implementation Plan):**
- Authored configuration to evaluate `Qwen3.5-2B` SFT checkpoint using test-time compute scaling (sampling K=1 to 16 candidates).
- Increased inference temperature to `0.8` to promote diverse candidate generation.
- Evaluated on a 100-task subset of the complex MBPP dataset.
- Computed scaling capability metrics and generated a `Capability vs Compute` plot.

**Metrics & Analysis:**
- **Pass@1:** `2.9%`
- **Pass@4:** `6.0%`
- **Pass@8:** `7.7%`
- **Pass@16:** `10.0%`
- **Mean Latency per Sample:** `17.15s`
- **Analysis:** This is a breakthrough finding! In Phase 4, the model scored 0% Pass@4 with a standard temperature. However, by slightly increasing the generation temperature and expanding the search budget at test time to K=16, the model successfully solved **10%** of the "impossible" complex tasks. This confirms that small language models (like Qwen3.5-2B) often possess the latent knowledge required to solve complex problems, but they struggle to find the correct reasoning path reliably on the first try. Test-Time Compute Scaling successfully bridges this gap, proving that giving a model "more time to think" (by generating more candidates) directly translates to higher verifiable coding capability, even without additional parameter updates!

### Phase 6: Tool-Use Agent & 4B Parameter Scaling
**Status:** Completed

**Tasks Finished (Based on Implementation Plan):**
- Built an autonomous ReAct agent loop (`llm_lab.agent`) featuring workspace file management (`read_file`, `write_file`), secure sandbox execution (`run_tests`), and task submission (`submit`).
- Implemented robust multi-strategy parsing in `extract_action` supporting both strict ReAct tool syntax and zero-shot fallback routing for raw Python code blocks.
- Developed the agent evaluation runner and metrics tracker measuring `task_completion_rate`, `recovery_rate` (ability to fix broken code following test failure feedback), and `steps_to_success`.
- Cached and loaded the secondary scaling model (`Qwen/Qwen3.5-4B`) in 4-bit NF4 precision on consumer hardware (peak VRAM ~3.5 GB).
- Executed the autonomous agent loop across 20 complex coding tasks from `train_complex_500.jsonl` on both `Qwen3.5-2B` (SFT checkpoint) and `Qwen3.5-4B` (Base model).
- Consolidated all outputs and comparison metrics to root-level `experiments/exp05_agent_2b` and `experiments/exp05_agent_4b`.

**Metrics & Comparison:**

| Metric | Qwen3.5-2B (SFT) | Qwen3.5-4B (Base) | Scaling Impact / Delta |
| :--- | :--- | :--- | :--- |
| **Model Size** | 2 Billion Parameters | 4 Billion Parameters | 2x Parameters |
| **Checkpoint** | SFT `checkpoint-final` | Base Pretrained | Scaling vs. Fine-tuning |
| **Precision** | 4-bit NF4 | 4-bit NF4 | Consumer 8GB GPU |
| **Peak VRAM** | ~1.8 GB | ~3.5 GB | Fits comfortably (<8GB) |
| **Evaluated Tasks** | 20 complex tasks | 20 complex tasks | MBPP complex subset |
| **Task Completion Rate** | **20.0%** (4/20) | **30.0%** (6/20) | **+10.0% (+50% relative gain)** |
| **Recovery Rate** | **20.0%** (4/20) | **26.3%** (5/19) | **+6.3% higher self-repair** |
| **Avg Steps to Success** | 5.00 steps | 4.83 steps | Faster convergence |
| **Tasks Passed** | `mbpp_605`, `mbpp_606`, `mbpp_618`, `mbpp_624` | `mbpp_604`, `mbpp_605`, `mbpp_618`, `mbpp_620`, `mbpp_623`, `mbpp_624` | Solved 2 additional complex tasks |

**Analysis & Key Takeaways:**
1. **Tool Feedback Doubles Coding Capability Over Best-of-N Sampling:**
   - In Phase 5, unguided test-time search (Best-of-16) achieved a 10.0% pass rate.
   - Giving the 2B model access to tools (`write_file`, `run_tests`) doubled the pass rate to **20.0%**, with **100% of the successful tasks resulting from test-failure recovery**. When initial code crashed on hidden tests, the model inspected the error trace, diagnosed the bug, and updated `main.py`.
2. **Model Parameter Scaling Delivers Substantial Zero-Shot Gains:**
   - The un-fine-tuned `Qwen3.5-4B` base model achieved a **30.0% Completion Rate** and **26.3% Recovery Rate**, easily outperforming the fine-tuned 2B SFT model.
   - The 4B model demonstrated significantly stronger semantic error comprehension (e.g., recognizing function name mismatches and algorithm boundary conditions) and faster convergence (4.83 avg steps).
   - Peak VRAM for 4B in 4-bit was ~3.5 GB, confirming that 4B parameter models can run full autonomous coding loops on consumer 8GB GPUs.

## Kaggle Migration (Phases 0-3 re-run on free T4x2, with corrections)

**Status:** In progress. Notebooks 00-05 and the GRPO smoke test are done; the full GRPO run (06) is in progress; notebooks 07-09 are written but not yet run. Plan and the full change log: [docs/kaggle_migration_plan.md](docs/kaggle_migration_plan.md) (section 0).

**Tasks Finished (Based on the Migration Plan):**
- Shared library `kaggle/datasets/llm-lab-utils/llm_lab_utils/` (schema, loader, `SubprocessVerifier`, batched evaluator, miner, GRPO/SFT helpers, paired-bootstrap metrics), 19 unit tests in `kaggle/tests/`. Docker-free: Kaggle's ephemeral, internet-off container plus rlimits and timeouts replaces Docker.
- Three Kaggle datasets (`llm-lab-utils`, `llm-lab-data` with a SHA-256 manifest, `llm-lab-wheels`) and push/pull scripts in `kaggle/scripts/`.
- NB 00 offline wheels, NB 01/01b hardware + dtype benchmarks, NB 02 dataset audit (CPU), NB 03 baseline, NB 04 SFT, NB 05 SFT evaluation, NB 06s GRPO smoke test.
- Notebook generators live in `kaggle/tools/` so every notebook is reproducible from code.

**Environment (measured):** 2x Tesla T4 (15.6 GB each), 33.7 GB RAM, Python 3.13, torch 2.11+cu128. A T4x2 session is billed once by wall-clock time, regardless of how many GPUs are used.

### Key findings
1. **dtype: fp16, not bf16 (NB 01b).** T4 has no native bf16. Rollout throughput 72.5 vs 30.0 tok/s and a LoRA training step 1.9 s vs 6.4-6.8 s (3.5x). The 2B model needs only 3.8 GB in 16-bit, so no quantization is used.
2. **The MBPP prompt was the bottleneck, not the model (NB 03).** Same 100 tasks, same model and sampling:

| Prompt | Pass@1 | Pass@4 | Pass@8 | "name not defined" failures |
| :--- | :--- | :--- | :--- | :--- |
| v1 (original, no function name) | 2.9% | 6.0% | 7.0% | 84.3% |
| v2 (adds one example test) | **40.6%** | 61.4% | 69.0% | 9.5% |

3. **The synthetic benchmark is saturated (NB 02, 03).** 4 prompt templates; train/dev/test IDs overlap; about half of the distinct test items match a training item on prompt + visible test. Base model scores 99.6% Pass@1, so it cannot separate methods. 14 of 420 MBPP tasks also fail their own reference solution and were excluded from the v2 set (406 tasks, split 256 train / 50 dev / 100 eval, disjoint and deterministic).
4. **SFT on 256 MBPP reference solutions gave no detectable gain (NB 04, 05).** Paired bootstrap on the 100 held-out tasks:

| Metric | Base | SFT | Difference | 95% CI |
| :--- | :--- | :--- | :--- | :--- |
| Pass@1 | 40.6% | 39.0% | -1.6 pt | [-8.9, +6.0] |
| Pass@4 | 61.4% | 57.6% | -3.8 pt | [-12.2, +5.0] |
| Pass@8 | 69.0% | 65.0% | -4.0 pt | [-14.0, +6.0] |

   SFT did change style: syntax errors 5.5% -> 1.1%, answers about half as long. With 100 eval tasks the CI is about +/-7 points, so smaller effects cannot be detected. Hyperparameters were fixed before any eval result and not retuned.
5. **GRPO smoke test (NB 06s):** 25.7 s/step (from SFT) and 46.3 s/step (from base, wordier answers), peak VRAM 5.6-6.4 GB, train reward 0.25-0.35. Projected 200 steps with both arms in parallel: about 2.6 h wall, billed once. Risk noted: 50-100% of sampled groups had zero reward variance (no learning signal).
6. **Cost lessons:** an unmerged LoRA adapter made evaluation about 2.4x slower (NB 05 cost 1.35 GPU h instead of ~0.65); adapters are now merged before evaluation. Per-device batch 8 ran out of memory in GRPO because Qwen3.5's 248k-token vocabulary makes the logits tensor huge; batch 2 with accumulation 8 keeps the same 16 completions per step.

### GPU time used so far (Kaggle free tier: 30 h/week)
| Notebook | GPU hours |
| :--- | :--- |
| 01 env + dtype benchmark / 01b (several runs) | ~0.5 total |
| 03 baseline (synthetic + MBPP v1/v2, 100 tasks x 8 samples) | 1.06 |
| 04 SFT (96 steps, 7.3 GB peak VRAM) | 0.16 |
| 05 SFT evaluation | 1.35 |
| 06s GRPO smoke (incl. one OOM run) | ~0.2 |
| **Running total (estimate)** | **~3.3-4.5 of 30** |

### Pending (to be filled in as notebooks complete)
- **NB 06/07 GRPO training and evaluation:** both arms (from SFT, from base) vs base/SFT with paired CIs; peak VRAM, RAM, wall time, GPU hours.
- **NB 08 hard-example mining** (re-run with the corrected prompt) and **NB 09 test-time scaling** (oracle Pass@k and a deployable visible-test selection from one batch of 16 samples).
- Redo of the agent experiments (NB 10/11) with the corrected prompt, then the research dashboard (NB 12).
