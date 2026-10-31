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

7. **GRPO training (NB 06, both arms in parallel, 200 steps, lr 2e-5, G=4, 16 completions/step):** `grpo_from_sft` 24.5 s/step (1.36 h), `grpo_from_base` 41.5 s/step (2.31 h); the session was billed 2.34 h. Peak VRAM 6.4-6.5 GB. Train reward rose from ~0.26-0.35 to ~0.5-0.66. Only 1 NaN-gradient log in total (fp16 scaler is stable). **61% of groups had zero reward variance** in both arms, i.e. most sampled groups taught nothing.
8. **GRPO evaluation (NB 07, 100 held-out MBPP tasks, 8 samples, merged adapters, paired bootstrap 95% CI):**

| Model | Pass@1 | Pass@4 | Pass@8 | OOD Pass@1 (synthetic) |
| :--- | :--- | :--- | :--- | :--- |
| base | 40.6% | 61.4% | 69.0% | 99.6% |
| SFT | 39.0% | 57.6% | 65.0% | - |
| **GRPO from base** | **58.5%** | **74.4%** | **80.0%** | 98.5% |
| GRPO from SFT | 42.6% | 56.4% | 63.0% | 100% |

   - GRPO(base) vs base: Pass@1 **+17.9 pt, CI [+12.4, +23.9]**; Pass@4 +13.0 [+6.7, +20.2]; Pass@8 +11.0 [+4.0, +19.0]. Significant at every k.
   - GRPO(SFT) vs SFT: Pass@1 +3.6 [-1.4, +8.9], not significant. GRPO(SFT) vs GRPO(base): -15.9 [-22.4, -9.4], significant.
   - Analysis: RL on pass/fail unit tests improved the base model more than it merely sharpened the distribution (Pass@8 also rose). Doing SFT first on 256 MBPP tasks did not help and left RL a worse starting point. Caveat: GRPO(base) answers are wordier (8.7% truncated at 384 tokens, syntax-error rate 6.1%). Eval used 1.31 GPU h.
9. **Test-time scaling (NB 09, `grpo_from_base`, 16 samples/task, 0.52 GPU h):**

| k | Oracle Pass@k | Visible-test selection |
| :--- | :--- | :--- |
| 1 | 55.4% | 55.5% |
| 2 | 65.6% | 64.0% |
| 4 | 72.0% | 68.7% |
| 8 | 77.0% | 71.8% |
| 16 | 81.0% | 73.7% |

   Selecting among 16 samples with only the example test shown in the prompt gives about +18 pt over a single sample, but trails the oracle by 7 pt at k=16 because ~10% of samples are "false friends" (pass the visible example, fail hidden tests). Truncation (9.4%) and syntax errors (7.1%) remain a lever to examine.

10. **Truncation analysis (no GPU; from saved NB 03/05/07 generations, a sample counts as truncated if its code fence never closes):**

| Model | Truncated at 384 tokens | Pass@1 when not truncated |
| :--- | :--- | :--- |
| base | 8.4% | 44% |
| SFT | 0.9% | 39% |
| GRPO from base | 8.5% | 64% |
| GRPO from SFT | 1.0% | 43% |

   - Truncation is **not an RL side effect**: the base model has the same rate. It concentrates on ~26 of 100 tasks (19 shared between base and GRPO), where the model reasons inside docstrings/comments about an ambiguous task statement and runs out of tokens before writing code (no repetition loops found). Truncated samples pass 1% of the time.
   - SFT removed truncation (0.9%) but not errors; **GRPO's gain comes from being right when it answers** (44% -> 64% on completed samples), not from shorter answers. Evaluation uses the same 384-token cap for every model, so the comparisons stay fair; a larger cap is untested.
   - Also this phase: the agent harness no longer lets `run_tests` reveal hidden-test expectations (visible tests for feedback, hidden tests only for final grading); NB 10/11 (agent 2B/4B) are written and not yet run.

11. **Hard-example mining (NB 08, base model, 256 MBPP training tasks x 4 samples, 0.90 GPU h):**

| Passes out of 4 | 0 | 1 | 2 | 3 | 4 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Tasks | 82 | 33 | 34 | 30 | 77 |

   - Classes: solved 107 (42%), borderline 34, hard 33, impossible 82 (32%). The old local "420/420 impossible" result is confirmed to have been a prompt artefact.
   - **62.1% of tasks have zero outcome variance** (0 or 4 passes), matching the ~61% of GRPO groups with zero reward spread measured in NB 06. Mining therefore predicts how much GRPO compute is wasted. Only 37.9% of tasks (1-3 passes, 97 tasks) can teach anything.
   - Caveat: `classify_task` labels 3/4 as "solved" although those groups still carry signal. The exported `hard`/`mixed` sets hold only 67 tasks, so the planned GRPO mining ablation (Step 19g) will use the 97 tasks with 1-3 passes versus 97 random tasks.

12. **Agent evaluation, 2B (NB 10; 60 held-out MBPP tasks, ReAct loop with read/write/run_tests/submit, max 5 steps; `run_tests` shows only the visible example, hidden tests grade at the end; 1.11 GPU h):**

| Model | Agent success | Single-shot Pass@1, same tasks | Agent - single-shot (95% CI) | Recovery | Parse-error steps | Truncated steps |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| base | 50.0% | 45.0% | +5.0 pt [-5.6, +15.0] | 36.8% | 6.2% | 17.0% |
| GRPO from base | 58.3% | 60.4% | -2.1 pt [-11.3, +6.7] | 33.3% | 11.5% | 11.5% |

   - **The agent loop did not beat single-shot generation at 2B**, either for the base or for the GRPO model (no CI excludes zero). GRPO-as-agent vs base-as-agent is +8.3 pt [-1.7, +20.0], also not significant. The only significant contrast (GRPO agent vs single-shot base, +13.3 pt) is the RL gain itself.
   - Compare the cost-effectiveness: choosing among 16 samples with the visible test (NB 09) gave about +18 pt; five agent steps gave about zero. For small models the format overhead (6-12% unparseable steps, 11-17% truncated steps, about 4 of 5 steps used) eats the benefit of feedback.
   - Hidden-test leak fixed in this run (see item 10): false friends are 1.7% (base) and 6.7% (GRPO).
13. **External OOD test: HumanEval (x02; 157 of 164 problems expressible as plain asserts, 0 canonical solutions failed the audit, n=8, 512-token cap; 1.50 GPU h):**

| Model | Pass@1 | Pass@4 | Pass@8 |
| :--- | :--- | :--- | :--- |
| base | 51.3% | 68.7% | 75.2% |
| GRPO from base | 53.7% | 69.3% | 75.8% |
| difference (95% CI) | +2.4 [-0.8, +5.5] | +0.7 [-3.5, +4.9] | +0.6 [-4.5, +5.7] |

   - **The MBPP gain (+17.9 pt) does not transfer to HumanEval** (+2.4 pt, CI includes 0). The CI is about +/-3 pt, so a transfer gain larger than ~5.5 pt can be ruled out. This is the most important caveat on the Phase 3 result: **what GRPO learned on 256 MBPP tasks is mostly specific to MBPP-style tasks** (task wording plus the example assert, naming and output conventions), not general coding ability.
   - Context: the base model is already stronger on HumanEval (51.3% vs 40.6% on MBPP), so there is less headroom, and HumanEval may be partly memorised by the pretrained model. Neither explanation is tested yet (plan Steps 19a, 19h).
   - No harm either: truncation (3.8% -> 4.9%) and syntax errors (1.1% -> 0.6%) did not get worse.

14. **Failure analysis (plan Step 19a; CPU only, `python kaggle/tools/analyze_failures.py` on the saved MBPP samples):**

| Share of all 800 samples | base | SFT | GRPO from base | GRPO from SFT |
| :--- | :--- | :--- | :--- | :--- |
| pass | 40.6% | 39.0% | 58.5% | 42.6% |
| wrong function name (`name 'x' is not defined`) | **9.2%** | 0.5% | 0.6% | 0.6% |
| wrong output | 34.9% | 46.3% | 29.7% | 43.8% |
| other exception | 9.4% | 12.9% | 4.8% | 11.3% |
| syntax error | 5.5% | 1.1% | 6.1% | 1.1% |
| Pass rate when the function exists | 47.7% | 39.7% | **62.9%** | 43.5% |

   - **About 4 of the 17.9 points are format-following**: 9.2% of base samples define a different function name than the example assert uses (for example `is_prime` instead of `prime_num`). GRPO removes this, but so does SFT (0.5%), and SFT did not improve, so fixing names alone does not produce the gain.
   - **The rest (~14 pt) is higher correctness once the function exists** (47.7% -> 62.9%). 26 of 100 tasks improved by 3 or more samples out of 8, none got worse by that much, and the gain splits evenly between tasks the base model sometimes solves (9.6 task-units) and never solves (8.6).
   - **Hypothesis (supported by examples, not proven):** the base model often ignores the output convention shown in the prompt's example assert. In `mbpp_773` the example expects `('python', 0, 6)` and the base returns `(0, 6, 'python')`; in `mbpp_625` it returns the list unchanged. GRPO learned to match the example. HumanEval states the signature and examples in a docstring that the base model already follows, which would explain why the gain does not transfer (item 13). A direct test (re-evaluating MBPP with the example assert removed from the prompt) was considered and deliberately skipped to save quota, so this hypothesis stays untested.

15. **Qwen3.5-4B baseline (x03; same 100 MBPP tasks, n=8, 384-token cap as NB 03/07; 1.45 GPU h):**

| Model | Pass@1 | Pass@4 | Pass@8 |
| :--- | :--- | :--- | :--- |
| 2B base | 40.6% | 61.4% | 69.0% |
| 4B base | 53.8% | 66.2% | 69.0% |
| 2B GRPO from base | 58.5% | 74.4% | 80.0% |
| 4B base vs 2B base (95% CI) | +13.1 [+6.4, +20.3] | +4.8 [-2.8, +12.8] | 0.0 [-9.0, +9.0] |
| 4B base vs 2B GRPO (95% CI) | -4.8 [-10.9, +1.5] | -8.1 [-15.4, -1.0] | -11.0 [-19.0, -3.0] |

   - **Doubling the parameters gave +13.1 pt Pass@1; GRPO on 256 tasks gave +17.9 pt.** At Pass@1 the 2B GRPO model is at least as good as the 4B base (difference n.s.); at Pass@4 and Pass@8 it is significantly better. Synthetic OOD Pass@1 is 99.5% (2B base 99.6%).
   - **Caveat: the 4B number is a lower bound.** 90.4% of 4B samples hit the 384-token cap (2B base: 8.4%) and 27.9% are syntax errors, because 4B starts with a long natural-language reasoning preamble ("The user wants me to ...") before the code. The 4B vs 2B comparison therefore mixes capability with the token budget. A fair test needs a larger cap (plan Step 19e).
   - Peak VRAM 8.7 GB, one T4.
16. **Agent evaluation, 4B (NB 11; same 60-task list as NB 10, but the 2400 s time budget ended after 39 tasks; 0.71 GPU h):**

| Comparison on the same 39 tasks | Reference | 4B agent | Difference (95% CI) |
| :--- | :--- | :--- | :--- |
| vs 4B single-shot (x03) | 51.3% | 74.4% | **+23.1 pt [+13.1, +34.0]** |
| vs 2B single-shot (NB 03) | 43.9% | 74.4% | +30.4 pt [+16.7, +44.2] |
| vs 2B agent (NB 10) | 53.8% | 74.4% | +20.5 pt [+7.7, +35.9] |

   - **Unlike 2B (item 12), the agent loop clearly helps at 4B**: +23 pt over its own single-shot, CI excludes zero. Recovery rate 28.6%, false friends 5.1%.
   - Format overhead is still large: 15.0% of steps unparseable, 27.2% truncated (the verbose 4B again), mean 3.8 steps. The loop works despite this, because feedback lets the model finish after truncation.
   - Only 39 of 60 tasks completed within the time budget, so the CI is wide and the tasks are the first 39 of the list, not a random subset. The run was not repeated.

17. **Token-cap sensitivity (x04, plan Step 19e; MBPP eval, cap 384 -> 768, n=4; the run hit its 7200 s cap, so the 4B arm is partial; ~2.0 GPU h billed):**

| Model | Pass@1 @384 (n=8) | Pass@1 @768 (n=4) | Change (95% CI) | Truncated @384 -> @768 | Pass@4 @768 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 2B base | 40.6% | 40.8% | +0.1 pt [-3.5, +3.7] | 8.4% -> 3.3% | 58.0% |
| 2B GRPO from base | 58.5% | 59.5% | +1.0 pt [-2.9, +5.0] | 8.5% -> 2.8% | 79.0% |
| 4B base (first 73 of 100 tasks only) | 53.4% | 68.5% | **+15.1 pt [+8.0, +22.3]** | 90.4% -> not logged | n/a |

   - **For the 2B models the 384 cap was not hiding anything.** Raising it to 768 cut truncation by about 5 pt but Pass@1 did not move (CIs include 0). The samples that were cut off at 384 are rambling, not nearly-correct code, which confirms the truncation analysis above. All 2B results in this README stand.
   - **For 4B the cap was the problem.** Its Pass@1 rises from 53.4% to 68.5% on the same 73 tasks. The x03 result (item 15) understated 4B by about 15 pt.
   - **Revised "2B + RL vs 2x parameters" (73 common tasks, cap 768, n=4):** 4B base 68.5% vs 2B GRPO 58.2%, +10.3 pt [+2.4, +18.5], so **the 4B model is significantly better than the GRPO-trained 2B model once it is allowed to finish**. 4B vs 2B base is +27.1 pt [+17.8, +36.3]; GRPO closes about 17 of those 27 points. The earlier finding "2B GRPO matches 4B" (item 15) was an artefact of the cap.
   - **Limits:** the 4B arm only covers the first 73 tasks (the run was cut off during the 4B arm, because 4B at 768 tokens is slow: about 1900 characters per sample versus 680 for 2B), so the 4B comparison uses 73 tasks and n=4. No `summary.json` exists for the 4B arm and the summary cell never ran; the numbers above are computed from the saved `generations.jsonl` files. The 4B truncation rate at 768 was not computed. Slow 4B decoding at 768 tokens also raises the cost of any further 4B experiment (plan Step 19i).

### GPU time used so far (Kaggle free tier: 30 h/week)
| Notebook | GPU hours |
| :--- | :--- |
| 01 env + dtype benchmark / 01b (several runs) | ~0.5 total |
| 03 baseline (synthetic + MBPP v1/v2, 100 tasks x 8 samples) | 1.06 |
| 04 SFT (96 steps, 7.3 GB peak VRAM) | 0.16 |
| 05 SFT evaluation | 1.35 |
| 06s GRPO smoke (incl. one OOM run) | ~0.2 |
| 06 GRPO training (2 arms in parallel, 200 steps) | 2.34 |
| 07 GRPO evaluation | 1.31 |
| 08 hard-example mining | 0.90 |
| 09 test-time scaling | 0.52 |
| 10 agent evaluation 2B | 1.11 |
| x02 HumanEval OOD | 1.50 |
| x03 4B baseline | 1.45 |
| 11 agent evaluation 4B | 0.71 |
| x04 length cap (cut off at the 7200 s limit) | ~2.0 |
| **Running total (estimate)** | **~15.1 of 30** (check Settings -> Quota) |

### Pending (to be filled in as notebooks complete)
- GRPO mining ablation, extra seeds, HumanEval, 4B baseline, failure analysis (plan Phase 6.5, Steps 19a-19i).
- Redo of the agent experiments (NB 10/11) with the corrected prompt, then the research dashboard (NB 12).
