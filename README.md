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

**Status:** Complete through Phase 6.5 (all notebooks 00-11 and extension runs x02-x08s done, results below; Step 19j and the full 4B GRPO run intentionally not run, see Limitations in *Final Results*). Phase 7 dashboard: see *Final Results* at the end of this file. Plan and the full change log: [docs/kaggle_migration_plan.md](docs/kaggle_migration_plan.md) (section 0).

**Tasks Finished (Based on the Migration Plan):**
- Shared library `kaggle/datasets/llm-lab-utils/llm_lab_utils/` (schema, loader, `SubprocessVerifier`, batched evaluator, miner, GRPO/SFT helpers, paired-bootstrap metrics), 23 unit tests in `kaggle/tests/`. Docker-free: Kaggle's ephemeral, internet-off container plus rlimits and timeouts replaces Docker.
- Three Kaggle datasets (`llm-lab-utils`, `llm-lab-data` with a SHA-256 manifest, `llm-lab-wheels`) and push/pull scripts in `kaggle/scripts/`.
- NB 00 offline wheels, NB 01/01b hardware + dtype benchmarks, NB 02 dataset audit (CPU), NB 03 baseline, NB 04 SFT, NB 05 SFT evaluation, NB 06s GRPO smoke test, NB 06/07 GRPO training and evaluation, NB 08 mining, NB 09/09b test-time scaling, NB 10/11 agents, x02-x08s extension experiments (items 12-22 below), `kaggle/tools/make_dashboard.py` (Final Results).
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

18. **Test-time scaling of the base model (09b, plan Step 19c; same protocol as NB 09, n=16, temperature 0.8; 0.62 GPU h):**

| k samples | base: oracle | base: visible-test selection | GRPO: oracle | GRPO: visible-test selection |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 37.4% | 37.5% | 55.4% | 55.5% |
| 2 | 49.7% | 48.0% | 65.6% | 64.0% |
| 4 | 59.7% | 55.8% | 72.0% | 68.7% |
| 8 | 67.7% | 61.7% | 77.0% | 71.8% |
| 16 | 75.0% | 66.6% | 81.0% | 73.7% |

   - **RL and "sample more and select" are partly interchangeable.** The GRPO model at k=1 (55.5%) equals the base model with 4 samples and visible-test selection (55.8%): one GRPO sample is worth about four base samples, so RL moved roughly 4x inference compute into the weights.
   - **They do not fully substitute.** At equal k, GRPO stays ahead in the deployable curve (k=16: 73.7% vs 66.6%, +7.1 pt) but the gap shrinks from +18 pt at k=1 to +7 pt at k=16. The oracle curves converge more (81.0% vs 75.0%).
   - **False friends** (pass the visible example but fail hidden tests): base 14.8%, GRPO 10.0% - RL also made the visible example a more reliable filter.
   - Base sampling at T=0.8 gives Pass@1 37.4% (vs 40.6% at T=0.7, NB 03): temperature costs the base model about 3 pt, so the k=1 point here is not the same operating point as the NB 03 table.

19. **Robustness to renaming (x07, plan Step 19h; the function name is replaced by a neutral name such as `compute_result` in the prompt, example assert, all tests and the reference; 100 tasks, all references still pass; n=4, same sampling; renamed-set sha256 `5a3df107...`; 0.72 GPU h):**

| | Original names (n=8) | Renamed (n=4) | Renamed - original (95% CI) |
| :--- | :--- | :--- | :--- |
| base Pass@1 | 40.6% | 42.0% | +1.4 pt [-6.0, +8.9] |
| GRPO from base Pass@1 | 58.5% | 55.0% | -3.5 pt [-7.7, +0.5] |
| GRPO - base | +17.9 pt [+12.4, +23.9] | **+13.0 pt [+7.2, +19.0]** | interaction -4.9 pt [-12.6, +3.1] |

   - **The RL gain survives the rename: +13.0 pt, CI excludes zero.** The apparent shrinkage (-4.9 pt) is not statistically significant, so we cannot claim that part of the gain was memorised names, but a loss of up to about 3 to 4 pt cannot be ruled out either (GRPO alone drops 3.5 pt, CI just touching zero).
   - `name 'x' is not defined` errors: base 7.3% (9.2% with the original names), GRPO 0.0% (0.6%). A neutral name does not make the base model's naming problem worse, so the base model's wrong names are not caused by informative function names.
   - Together with x02 (no transfer to HumanEval) and the 19a failure analysis: the MBPP gain is real and not name-specific, but it is tied to MBPP-style prompts (example assert in the prompt, MBPP conventions), not to general coding ability. The "GRPO learned to match the example assert's output format" hypothesis (item 14) stays untested.

20. **Mining ablation with GRPO (x06, plan Step 19g; `grpo_from_base` recipe of NB 06, seed 42, 200 steps, 97 training tasks per arm; evaluation as NB 07; session 3.37 GPU h):**

| Arm (97 train tasks) | Pass@1 | Pass@4 | Pass@8 | Zero-variance groups in training | Train GPU h |
| :--- | :--- | :--- | :--- | :--- | :--- |
| base (reference) | 40.6% | 61.4% | 69.0% | - | - |
| `learnable97` (base got 1-3 of 4) | 55.8% | 72.5% | 77.0% | **40.3%** | 2.38 |
| `random97` | 58.5% | 73.7% | 77.0% | 61.3% | 2.18 |
| full 256 tasks (NB 06/07) | 58.5% | 74.4% | 80.0% | 61% | 2.34 |

| Comparison (95% CI) | Pass@1 | Pass@4 | Pass@8 |
| :--- | :--- | :--- | :--- |
| `learnable97` - base | +15.1 [+9.6, +20.9] | +11.1 [+4.4, +18.3] | +8.0 [+1.0, +16.0] |
| `random97` - base | +17.9 [+11.1, +24.9] | +12.3 [+4.9, +20.1] | +8.0 [0.0, +17.0] |
| `learnable97` - `random97` | -2.8 [-7.9, +2.1] | -1.2 [-5.1, +2.7] | 0.0 [-5.0, +5.0] |
| `random97` - full 256 | 0.0 [-4.6, +4.9] | -0.7 [-5.7, +4.0] | -3.0 [-9.0, +3.0] |

   - **The Phase 4 hypothesis is not supported: training only on the "learnable" tasks was not better than random tasks** (-2.8 pt Pass@1, CI includes 0; point estimates favour random on Pass@1 and Pass@4, tied on Pass@8). Mining did what it was designed to do for the signal (zero-variance groups 61% -> 40%), but a cleaner signal did not buy accuracy.
   - **Training on 97 random tasks matched training on all 256** (58.5% vs 58.5% Pass@1, difference 0.0 [-4.6, +4.9]). So the RL gain on MBPP does not need many tasks; the first ~100 tasks carry essentially all of it. This also means the 256-task training set was not the limit, and raises the possibility that much of the gain is learning a format/behaviour that a handful of tasks teaches (consistent with x02 and the 19a analysis).
   - Caveat: single seed per arm. The seed spread (x05) tells how large a difference of 2.8 pt has to be before it means anything; the CIs above are over tasks, not over training runs. Gain per training GPU hour: 0.063 (learnable97) vs 0.082 (random97) Pass@1 points per hour.
   - Synthetic OOD Pass@1: 99.8% (learnable97) and 100.0% (random97): no regression.

21. **4B GRPO calibration (x08s, plan Step 19i; 6 steps per configuration, `grpo_from_base` recipe with Qwen3.5-4B, completion cap 768; 0.32 GPU h) - NO-GO:**

| Config (16 completions per step) | Fits a 16 GB T4 | s / step | Projected 200 steps | Peak VRAM | Mean reward (6 steps) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| per-device batch 1 x accum 16 | yes | 166.7 | **9.3 h** | 12.0 GB | 0.615 |
| per-device batch 2 x accum 8 | yes | 159.8 | **8.9 h** | 13.7 GB | 0.573 |

   - **4B GRPO fits in memory but not in the budget.** The pre-registered rule was "run only if the projected 200 steps take at most 5 GPU hours"; the best configuration projects 8.9 h (about 3.5x the 2B run), so the run was **not** launched. The cause is generation: 4B writes about 660 tokens per completion (the 768 cap is needed, see x04) and decodes at roughly half the 2B speed.
   - Reward is not degenerate (0.57 to 0.62), so the recipe itself is sound for 4B; only compute is the limit. A shorter run (for example 100 steps, about 4.4 h) would be a different hyperparameter set and would not be comparable with the 2B run, so it was not substituted. The question "does the RL gain persist at 4B" stays open.

22. **Seed variance of the GRPO gain (x05, plan Step 19f; `grpo_from_base` recipe of NB 06 with seeds 43 and 44 only changed; evaluation as NB 07; session 3.65 GPU h):**

| Seed | Pass@1 | Pass@4 | Pass@8 | Gain over base, Pass@1 (95% CI) | Truncated | Zero-variance groups |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 42 (NB 06/07) | 58.5% | 74.4% | 80.0% | +17.9 [+12.4, +23.9] | 8.5% | 61% |
| 43 | 51.9% | 70.3% | 75.0% | +11.3 [+5.5, +17.2] | 16.6% | 57.4% |
| 44 | 55.8% | 70.8% | 73.0% | +15.1 [+9.4, +21.1] | 5.5% | 59.3% |
| **mean +/- std (3 seeds)** | **55.4% +/- 3.3 pt** | 71.8% | 76.0% | **+14.8 pt**, range +11.3 to +17.9 | | |

   - **The GRPO gain is real and stable in sign: all three seeds are significantly above base at Pass@1 and Pass@4.** The base-relative gain is **+14.8 pt on average (range +11.3 to +17.9)**.
   - **Seed 42 was the luckiest of the three.** The +17.9 pt reported in NB 07 is the top of the range; the honest headline number is "about +15 pt, seed std 3.3 pt". Seed 42 vs seed 43 differs significantly (+6.6 pt [+2.9, +10.5]) at the level of a single training run on identical data and settings.
   - **Pass@8 gains are not robust:** for seeds 43 and 44 the Pass@8 gain over base (+6.0 and +4.0 pt) has a CI that includes 0. RL mostly improves single-sample reliability, less the ceiling of what the model can ever solve.
   - **Consequence for the other experiments:** single-run differences smaller than about 3 to 6 pt are within run-to-run noise. This includes the mining ablation (`learnable97` vs `random97`, -2.8 pt, item 20) and every "GRPO vs X" comparison with a single seed. Seed 43 also has a much higher truncation rate (16.6% vs 5.5 to 8.5%), so RL does not always shorten answers.

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
| 09b test-time scaling, base | 0.62 |
| x07 robustness (rename) | 0.72 |
| x06 GRPO mining ablation (2 arms + eval) | 3.37 |
| x05 GRPO seeds 43/44 (2 arms + eval) | 3.65 |
| x08s 4B GRPO calibration | 0.32 |
| **Total** | **~23.8 GPU hours over two quota weeks** (week 1: 13.1 through NB 11; week 2: 10.7 from x04 on; never above 15.1 in one week) |

## Final Results (Phase 6.5 and Phase 7: research dashboard)

Generated by `python kaggle/tools/make_dashboard.py` from the pulled Kaggle outputs (0 GPU hours); files: `reports/final_results.csv`, `reports/final_results.md`, `reports/final_results.json`, figures in `reports/figures_final/` (`f1_pass_at_1_models`, `f2_roi`, `f3_test_time_scaling`, `f4_transfer`, `f5_token_cap`, `f6_agent`, `f7_mining`, `f8_seeds`). All numbers are on the frozen held-out set (100 `mbpp_v2_eval` tasks, 8 samples, temperature 0.7, top-p 0.9, 384 tokens, seed 1234) unless stated. "OOD Pass@1" is the Pass@1 on the first 100 synthetic test tasks (regression check), HumanEval is reported separately.

| Model | Pass@1 | Pass@4 | Pass@8 | OOD Pass@1 | Peak VRAM GB | Peak RAM GB | Eval wall h | Train GPU h |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2B base | 40.6% | 61.4% | 69.0% | 99.6% | 3.9 | 2.6 | 0.47 | - |
| 2B SFT | 39.0% | 57.6% | 65.0% | 98.8% | 3.9 | 2.6 | 0.57 | 0.16 |
| 2B GRPO from SFT | 42.6% | 56.4% | 63.0% | 100.0% | 3.9 | 2.6 | 0.41 | - |
| 2B GRPO seed 42 | 58.5% | 74.4% | 80.0% | 98.5% | 3.9 | 2.8 | 0.39 | 2.34 |
| 2B GRPO seed 43 | 51.9% | 70.3% | 75.0% | 98.2% | 3.9 | 2.7 | 0.46 | 2.51 |
| 2B GRPO seed 44 | 55.8% | 70.8% | 73.0% | 100.0% | 3.9 | 2.8 | 0.38 | 2.56 |
| 2B GRPO 97 learnable tasks | 55.8% | 72.5% | 77.0% | 99.8% | 3.9 | 2.6 | 0.42 | 2.38 |
| 2B GRPO 97 random tasks | 58.5% | 73.7% | 77.0% | 100.0% | 3.9 | 2.7 | 0.36 | 2.18 |
| 4B base (384 cap; 768 cap gives about +15 pt, item 17) | 53.8% | 66.2% | 69.0% | 99.5% | 8.7 | 2.7 | 0.97 | - |

GRPO from base over 3 seeds: **Pass@1 55.4% +/- 3.3 pt, gain over base +14.8 pt**. Training GPU hours are session times for two arms running in parallel; the "wall time" and "GPU hours" of every individual experiment are in `reports/final_results.csv` and in each experiment's `summary.json`.

### Answers to the research questions
1. **Does RL with unit-test rewards improve a 2B model on held-out MBPP?** Yes. About **+15 pt Pass@1** (range +11 to +18 over three seeds), significant on every seed at Pass@1 and Pass@4, with no loss on the synthetic regression set. The +17.9 pt first reported was the best of three seeds.
2. **Does SFT help first?** No. SFT alone did not help (-1.6 pt) and GRPO starting from SFT was far worse (+2.0 pt) than GRPO from the base model (+17.9 pt, seed 42).
3. **Is it better to train or to sample more at inference?** They overlap. One GRPO sample is worth about four base samples chosen with the visible test (item 18), but the GRPO model stays ahead at equal k. Training costs 2.3 GPU h once; sampling costs a multiple on every query.
4. **Does bigger beat trained-smaller?** Doubling parameters gives about +27 pt Pass@1 over 2B base (73 tasks, 768 cap), GRPO about +15 to +18 pt. The 4B model is better; GRPO recovers more than half of the gap. 4B GRPO itself was not run (8.9 projected hours, over the 5 h rule).
5. **Does hard-example mining help GRPO?** Not measurably. Mining cut wasted groups from 61% to 40% but accuracy was not better than for random tasks (-2.8 pt, within seed noise), and 97 random tasks matched all 256 tasks.
6. **Does the gain generalise?** Partly. It survives renaming the function (+13 pt) but **does not transfer to HumanEval (+2.4 pt, CI [-0.8, +5.5])**. The honest claim is "the model got better at MBPP-style tasks", not "better at coding".
7. **Do agents help?** Only at 4B (+23 pt over its single-shot on 39 tasks); at 2B the tool loop gave no significant gain.

### Which method is worth its compute? (Pass@1 gain over 2B base per training GPU hour; `f2_roi`)
GRPO from base: +14.8 pt / 2.5 h of training = about 6 pt per hour (seed range 4.5 to 7.7); SFT: negative; mining: no better than random. Inference-time options cost no training but multiply per-query cost (k=4 selection: +18 pt for 4x generation).

### Limitations (read before quoting any number)
- 100 evaluation tasks: a single comparison has a 95% CI of about +/-6 pt; differences below that are not claimed. Run-to-run (seed) std of a GRPO model is 3.3 pt, also larger than most ablation differences.
- One model family (Qwen3.5), one training source (256 MBPP tasks), one benchmark family. HumanEval conversion covers 157 of 164 problems (the others are not expressible as plain asserts).
- The 4B results are lower bounds under the 384 cap and, at 768, cover only 73 tasks (the run hit its time cap); the 4B agent run completed 39 of 60 tasks.
- The "GRPO learned to match the example assert's format" hypothesis (item 14) was deliberately not tested (plan Step 19k dropped to save quota) and stays a hypothesis.
- Step 19j (a second training source to test transfer to HumanEval) was not run: it needs a new decontaminated dataset and about 3.5 GPU h, and the seed variance (item 22) means one more single-run result would not settle the question. It is the first thing to do next.
- All code execution happened inside Kaggle's isolated container with `SubprocessVerifier` limits; nothing generated was run on the Windows host (AGENTS.md rules 8 and 9).

### Blog drafts
Nine draft posts in `docs/blog/` (index: `docs/blog/README.md`) tell the story with the numbers above.

### Reproduction
`kaggle/scripts/sync_all.ps1` uploads the shared library and the frozen data; `kaggle/scripts/push_kernel.ps1 <kernel> -Accelerator NvidiaTeslaT4 -TimeoutSeconds N` runs a notebook; `python kaggle/tools/build_*.py` regenerate the notebooks; `kaggle/scripts/pull_results.ps1` and `kaggle kernels output` download results; `python kaggle/tools/analyze_failures.py` and `python kaggle/tools/make_dashboard.py` regenerate the CPU-side analyses. Unit tests: `python -m pytest kaggle/tests`. The data files are listed with their sha256 in `kaggle/datasets/llm-lab-data/MANIFEST.sha256`; the frozen test set was never modified.
