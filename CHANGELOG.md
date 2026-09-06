# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Phase 2: Supervised Fine-Tuning (SFT) Pipeline**
  - Added training configurations and datasets for LoRA fine-tuning using Hugging Face `trl` and `bitsandbytes` (4-bit NF4).
  - Created `src/llm_lab/training/sft.py` supporting gradient checkpointing, tracking peak VRAM usage, and handling chat template alignment.
  - Added `notebooks/sft_evaluation.ipynb` demonstrating the end-to-end dataset generation, training, and evaluation lifecycle.
- **Phase 1: Dataset, Verifier & Evaluation Pipeline**
  - Dataset Layer: `src/llm_lab/data` implementing Pydantic `TaskSchema`, JSONL loaders, and synthetic `generator.py`.
  - Verifier Layer: `src/llm_lab/verifier` implementing secure `DockerVerifier` to execute and test generated candidate code in an ephemeral `public.ecr.aws/lambda/python:3.12-rapid-x86_64` container without networking.
  - Evaluation Layer: `src/llm_lab/evaluation` calculating Pass@K metrics (`metrics.py`) and running the full inference-verification loop (`evaluator.py`).
  - Reporting Layer: `src/llm_lab/reporting/experiment.py` outputting `summary.json` and per-task `generations.jsonl`.
  - Baseline Config: `configs/eval.yaml` defining parameters for the initial evaluation loop.
- **Phase 0: Project Scaffolding & Environment Setup**
  - Project directory structure (`src/llm_lab`, `configs`, `experiments`, `data`, `tests`, `scripts`).
  - `pyproject.toml` containing dependencies for small-scale LLM lab (`transformers`, `trl`, `peft`, `bitsandbytes`, `pytest`, etc.).
  - `AGENTS.md` establishing AI development rules, reproducibility principles, and confirming Qwen3.5-2B as the primary model.
  - `README.md` defining the core research question and primary metrics (Pass@k, Capability Gain / GPU Hour).
  - `.gitignore`, `.env.example`, and `src/llm_lab/__init__.py`.
  - Diagnostics tools: `src/llm_lab/hardware.py` and `scripts/doctor.py` to monitor VRAM, RAM, and CUDA availability.
  - Smoke tests in `tests/test_smoke.py`.
  - Inference config `configs/baseline.yaml` using 4-bit NF4 quantization.
  - Model caching script `scripts/download_model.py` and inference script `src/llm_lab/inference/generate.py`.

### Changed
- Configured PyTorch for CUDA 12.4 local installation via `uv pip install`.
- Configured Hugging Face downloads to route through `hf-mirror.com` via `HF_ENDPOINT` to prevent connection reset errors.
- Relocated model cache to `D:\huggingface_cache` and removed cache from C: drive (~4.26 GB freed).
- Added `src/llm_lab/constants.py` defining `HF_HOME`, `DEFAULT_HF_HOME`, and model IDs.
- Configured Windows User environment variable `HF_HOME=D:\huggingface_cache` and updated `.env` and `.env.example`.
- Modified `src/llm_lab/inference/generate.py` to support passing a direct `--prompt` argument, custom `cache_dir`, and `local_files_only: true` for offline zero-latency execution.

### Fixed
- Handled Hugging Face connection drops (`[WinError 10054]`) by implementing the official mirror.
- Resolved conflict between Ollama model format (`GGUF`) and TRL training pipeline by ensuring the project downloads Hugging Face `safetensors` for future SFT/RLVR training phases.

### Verified
- **Milestone M0**: `Qwen/Qwen3.5-2B` successfully loads onto an RTX 2060 SUPER (8GB VRAM) in 4-bit precision, utilizing just `1.71 GB` of Peak VRAM and returning a successful inference pass for standard Python code generation.
