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