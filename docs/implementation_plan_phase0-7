# Local LLM Lab — Consolidated Implementation Plan

> Merged from proposal.md sections 1–59 (lines 1547–3764). All original content preserved, reorganized into 7 sequential phases. Each phase groups **manual setup** (things you do yourself) and **AI code-gen** (prompts for Cursor/agent) so you can batch work efficiently.

---

## Phase 0: Project Scaffolding + Environment + Model Loading

**Goal**: Get from zero to "model loads, GPU works, inference runs" on your Windows machine.

**Milestone M0**: `python scripts/doctor.py` prints `STATUS: READY`, and a single Qwen3.5-2B inference completes successfully.

---

### 0A — Manual Setup (Do This First)

#### Project Structure (§1)

Create the directory skeleton:

```text
local-llm-lab/
│
├── README.md
├── pyproject.toml
├── uv.lock
├── .gitignore
├── .env.example
│
├── configs/
│   ├── baseline.yaml
│   ├── sft_v1.yaml
│   ├── grpo_v1.yaml
│   └── eval.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── train.jsonl
│   ├── dev.jsonl
│   └── test.jsonl
│
├── scripts/
│   ├── download_model.py
│   ├── validate_dataset.py
│   └── run_experiment.py
│
├── src/
│   └── llm_lab/
│       ├── __init__.py
│       │
│       ├── config.py
│       ├── logging_utils.py
│       ├── hardware.py
│       │
│       ├── inference/
│       │   ├── generate.py
│       │   └── sampling.py
│       │
│       ├── evaluation/
│       │   ├── evaluator.py
│       │   ├── metrics.py
│       │   └── splits.py
│       │
│       ├── verifier/
│       │   ├── interface.py
│       │   ├── python_verifier.py
│       │   └── docker_verifier.py
│       │
│       ├── data/
│       │   ├── schema.py
│       │   ├── loader.py
│       │   ├── generator.py
│       │   └── miner.py
│       │
│       ├── training/
│       │   ├── sft.py
│       │   └── grpo.py
│       │
│       └── reporting/
│           ├── experiment.py
│           └── plots.py
│
├── tests/
│   ├── test_dataset.py
│   ├── test_metrics.py
│   ├── test_verifier.py
│   └── test_smoke.py
│
├── experiments/
│   ├── exp00_baseline/
│   ├── exp01_sft/
│   ├── exp02_grpo/
│   ├── exp03_mining/
│   └── exp04_search/
│
└── reports/
    ├── experiment_log.csv
    └── figures/
```

Each experiment produces an independent directory (§1):

```text
experiments/exp02_grpo/
├── config.yaml
├── metrics.json
├── train.log
├── generations.jsonl
├── checkpoint/
└── summary.md
```

#### Tech Stack (§2)

```text
Python 3.11
PyTorch, Transformers, TRL, PEFT, bitsandbytes, datasets
accelerate, safetensors, PyYAML, pydantic
pandas, numpy, matplotlib
pytest, psutil, GPUtil / pynvml
```

**Windows 上建议 Python 3.11。** 依赖由 `uv` 管理；PyTorch 根据 NVIDIA 驱动支持的 CUDA wheel 安装。

#### `pyproject.toml` (§3)

不要让 AI 把所有版本写死成某个旧版本。ML stack 变化很快。

```toml
[project]
name = "local-llm-lab"
version = "0.1.0"
description = "Small-scale LLM capability research on a consumer GPU"
readme = "README.md"
requires-python = ">=3.11,<3.13"

dependencies = [
    "transformers>=4.55",
    "trl>=0.20",
    "peft>=0.17",
    "bitsandbytes>=0.46",
    "accelerate>=1.10",
    "datasets>=4.0",
    "safetensors>=0.5",
    "pyyaml>=6.0",
    "pydantic>=2.10",
    "numpy>=2.0",
    "pandas>=2.2",
    "matplotlib>=3.9",
    "psutil>=6.0",
    "pynvml>=12.0",
    "pytest>=8.0",
    "orjson>=3.10",
    "tqdm>=4.67",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "ruff>=0.12",
    "mypy>=1.16",
]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]

[build-system]
requires = ["setuptools>=75"]
build-backend = "setuptools.build_meta"
```

> [!IMPORTANT]
> **PyTorch 不放 pyproject.toml 里面**，因为它需要根据 NVIDIA/CUDA 环境单独安装。
> bitsandbytes 官方目前要求 Python >=3.10、PyTorch >=2.4，并提供 NVIDIA CUDA 支持。

#### Windows Environment Install (§4–§5)

PowerShell:

```powershell
mkdir local-llm-lab
cd local-llm-lab

# Option A: venv
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# Option B: uv
pip install uv
uv venv --python 3.11
.venv\Scripts\Activate.ps1
```

**第一步单独安装 PyTorch** — **不要让 AI 猜 CUDA 版本。** 打开 PyTorch 官方安装页面，选择 Windows / Pip / Python / CUDA = your supported version，然后执行官方给你的安装命令。

验证:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_properties(0).total_memory / 1024**3)"
```

你应该看到 `True`、`<你的 NVIDIA GPU>`、`~8 GB`。

安装其余依赖:

```powershell
pip install -e ".[dev]"
python -c "import transformers, trl, peft, bitsandbytes; print('OK')"
pytest
```

#### AGENTS.md — AI 开发规则 (§40)

把下面这段放进项目根目录 `AGENTS.md`:

```markdown
# Local LLM Lab Development Rules

## Goal

Build a reproducible small-scale LLM capability research framework
for an 8GB NVIDIA GPU and 48GB system RAM.

Primary model:
Qwen/Qwen3.5-2B

Secondary model:
Qwen/Qwen3.5-4B

## Principles

1. Reproducibility is more important than convenience.
2. Never modify the frozen test set.
3. Every experiment must have a unique experiment ID.
4. Every experiment must save its exact config.
5. Every model checkpoint must record its parent checkpoint.
6. Never silently change hyperparameters.
7. Never mix train/dev/test examples.
8. Never execute generated code directly on the Windows host.
9. Python candidate code must run inside a sandbox/container.
10. Every experiment must report:
   - Pass@1
   - Pass@4
   - Pass@8
   - OOD Pass@1
   - peak VRAM
   - peak RAM
   - wall time
   - GPU hours
11. Do not introduce additional reward terms unless explicitly requested.
12. Keep evaluation deterministic where possible.
13. Add tests before refactoring core infrastructure.
14. Avoid unnecessary dependencies.
15. Do not redesign the model architecture.
16. Prefer small experiments that can finish in hours.
17. Never overwrite previous experiment artifacts.
```

#### Git Strategy (§39)

```text
main
 │
 ├── exp/baseline
 ├── exp/sft-v1
 ├── exp/grpo-v1
 ├── exp/mining-v1
 └── exp/agent-v1
```

每个 experiment 的 code + config 都能复现。**不要把 hyperparameter 改来改去然后继续用同一个 checkpoint。**

#### Model 选择 (§7, updated)

第一阶段：**Qwen/Qwen3.5-2B** (Apache-2.0 license)

> [!TIP]
> **为什么从 Qwen3-1.7B 升级到 Qwen3.5-2B？**
> - 更新的架构 (Gated Delta Networks + Sparse MoE)，推理效率更高
> - 原生 262K context（vs 32K），为后续 agent/tool-use 留出空间
> - VRAM 需求相当（4-bit ~2–3 GB，bf16 LoRA ~5 GB），8GB 显卡完全够用
> - HuggingFace transformers / TRL / QLoRA 全部支持
> - 注意：bf16 LoRA 可能优于 4-bit QLoRA（Qwen3.5 的量化噪声略高），先试 4-bit，如不稳定切 bf16 LoRA

**不要直接默认开启它的 thinking 模式** 做第一轮 coding baseline。你的第一个研究问题是：

> RLVR 能否提升可验证 coding capability？

不是 "thinking mode + sampling + RLVR 混在一起会不会更好？" 变量一次只改一个。

#### README 科学问题 (§59)

README 开头:

> **Research Question**
>
> Can small, consumer-hardware-scale post-training methods improve the verifiable coding capability of a 2B language model, and which intervention gives the highest capability gain per unit compute?

Interventions: SFT, RLVR, Hard-example mining, Test-time compute, Tool-use

核心研究指标: $\text{Capability Gain / GPU Hour}$

#### 第一版配置锁死 (§58, §59)

| 项目             | 第一版                                         |
| -------------- | ------------------------------------------- |
| GPU            | 8GB NVIDIA                                  |
| RAM            | 48GB                                        |
| Model          | **Qwen3.5-2B**                              |
| Quantization   | 4-bit NF4 (fallback: bf16 LoRA)             |
| Fine-tuning    | QLoRA                                       |
| SFT data       | 500 → 5K                                    |
| Context        | 4096                                        |
| RL             | GRPO                                        |
| GRPO group     | **4**                                       |
| Completion     | 384 tokens                                  |
| Reward         | **hidden pytest pass/fail**                 |
| Test           | 1,000 frozen tasks                          |
| Primary metric | **Pass@1**                                  |
| Secondary      | Pass@4 / Pass@8                             |
| Generalization | OOD Pass@1                                  |
| Efficiency     | Gain / GPU-hour                             |
| Later          | Hard mining → test-time search → agent → 4B (Qwen3.5-4B) |

> [!NOTE]
> 你的 8GB 显存会限制实验规模，但不会阻止你研究真正重要的东西。

---

### 0B — AI Code-Gen (Two Prompts, Run Sequentially)

#### Prompt #1 — Environment + Doctor (§6, §41)

```text
Implement Phase 0 of this repository.

Goal:
Create a reproducible Windows environment diagnostic for an 8GB NVIDIA GPU
and 48GB system RAM.

Create:
- scripts/doctor.py
- src/llm_lab/hardware.py
- tests/test_smoke.py

The doctor must report:
- Python version
- PyTorch version
- CUDA availability
- CUDA version
- GPU name
- VRAM
- system RAM
- Transformers version
- TRL version
- PEFT version
- bitsandbytes version
- Docker availability

Do not install packages automatically.

Add clear error messages.

Run the tests and fix all failures.

Do not implement training yet.
```

Expected doctor output (§6):

```text
=== Local LLM Lab Environment ===

Python              3.11.x        OK
PyTorch             x.x.x         OK
CUDA available      True          OK
GPU                 RTX xxxx
VRAM                7.9 GB
System RAM          47.8 GB

Transformers        x.x.x         OK
TRL                 x.x.x         OK
PEFT                x.x.x         OK
bitsandbytes        x.x.x         OK

Docker              available     OK

STATUS: READY
```

#### Prompt #2 — Model Loading + Inference (§42)

```text
Implement local inference for Qwen/Qwen3.5-2B.

Requirements:
- 4-bit quantized loading
- device_map=auto
- configurable generation parameters
- no hardcoded absolute paths
- support model ID from YAML config
- save generation metadata
- record peak GPU memory

Create:
- src/llm_lab/inference/generate.py
- configs/baseline.yaml
- scripts/download_model.py
- tests/test_smoke.py

The implementation must work on an 8GB NVIDIA GPU.

Add a CLI:
python -m llm_lab.inference.generate --config configs/baseline.yaml

Do not implement training.
```

Model download code reference (§7):

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
MODEL_ID = "Qwen/Qwen3.5-2B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto")
```

#### `baseline.yaml` Reference (§20)

```yaml
experiment:
  id: exp00_baseline
  description: "Qwen3.5-2B baseline"

model:
  id: Qwen/Qwen3.5-2B
  load_in_4bit: true
  device_map: auto

generation:
  max_new_tokens: 384
  temperature: 0.7
  top_p: 0.9
  do_sample: true

evaluation:
  dataset: data/test.jsonl
  num_samples: [1, 4, 8]

logging:
  output_dir: experiments/exp00_baseline
```

建议**固定 random seed**，同时另外做少量 seed robustness test。

### 0C — Manual Validation

- [ ] `python scripts/doctor.py` → `STATUS: READY`
- [ ] `python -m llm_lab.inference.generate --config configs/baseline.yaml` → generates output
- [ ] `pytest` → all green

---

## Phase 1: Dataset + Verifier + Baseline Evaluation

**Goal**: Build the full experiment loop: prompt → model → code → verify → reward → metric → log. Run it on 100 toy tasks.

**Milestone M1 (§53)**: The complete closed loop runs stably on **100 tasks**:

```text
prompt → Qwen3.5-2B → candidate code → Docker verifier → reward → metric → experiment log
```

> [!IMPORTANT]
> **这才是第一大里程碑。** 不是"模型开始训练"。

---

### 1A — Manual Setup

#### Docker Desktop

Install Docker Desktop with WSL2 backend on Windows. Verify with `docker --version`.

#### Data Schema Design (§8)

每道任务的 schema:

```json
{
  "id": "py_000001",
  "version": 1,
  "category": "array",
  "difficulty": 0.42,
  "prompt": "Write a Python function called merge_intervals(...)",
  "starter_code": "",
  "visible_tests": [
    { "input": "[[1,3],[2,6]]", "expected": "[[1,6]]" }
  ],
  "hidden_tests": [
    { "input": "[]", "expected": "[]" },
    { "input": "[[1,1]]", "expected": "[[1,1]]" }
  ],
  "reference_solution": "def merge_intervals(...): ...",
  "metadata": {
    "source": "synthetic",
    "generator": "teacher_model_v1"
  }
}
```

> [!WARNING]
> `reference_solution` 不进入 prompt，也不进入 test set 的训练 artifact。

#### Train / Dev / Test Split (§9)

Fixed:

```text
train = 5,000
dev   = 500
test  = 1,000
```

`test.jsonl` **freeze**。以后 Base / SFT / RL / Mining / Agent / 4B 全部跑同一个 `test.jsonl`。

#### Dataset v0 — 100 Tasks (§52)

不要马上准备 5K 数据！实际执行顺序：

| Version | Size | 目的 |
|---------|------|------|
| v0 | 100 tasks | 验证 schema, verifier, evaluator |
| v1 | 500 tasks | baseline, debug, 第一次 SFT |
| v2 | 5,000 tasks | 正式实验 |

这会节省你大量时间。因为最容易发生的事情是：

> 你花 10 小时生成 5,000 道题，然后发现 verifier 有 bug。 😅

#### Dataset 设计 (§10)

第一版目标分布 (5,000 total):

| Category         | Count |
| ---------------- | ----: |
| Basic Python     | 1,000 |
| Arrays / strings | 1,000 |
| Data structures  | 1,000 |
| Algorithms       | 1,250 |
| Edge cases       |   750 |

第二阶段再加入: debugging, multi-function, repo-level, tool use

#### Data 来源 (§11)

不要只下载一个 benchmark 然后 fine-tune:
- **20–30%**: 公开数据 / benchmark-inspired
- **70–80%**: synthetic + verified

生成流程:

```text
Teacher model → generate problem → generate reference solution
→ generate hidden tests → run verifier → pass? → keep/discard
```

#### Verifier 安全 (§12)

**不要直接让模型生成的 Python 在你的 Windows host 上运行。** 模型输出就是不可信代码。

正确结构:

```text
Model → candidate.py → Docker container → network disabled
→ CPU limit → RAM limit → timeout → pytest → reward
```

Docker Desktop 在 Windows 上可以使用 WSL2 backend。**第一阶段就先把 verifier 和容器隔离做好。**

#### Reward 设计 — 第一版 (§14)

**就这么简单：**

```python
reward = 1.0 if all_tests_pass else 0.0
```

不要一上来就加 correctness + elegance + token length + style + complexity + comments。否则你以后无法知道：

> 是模型真的变强了，还是 reward hacking。

#### Reward 第二版 (§15)

等基础 RL 跑通以后再:

```python
reward = (
    1.0 * correctness
    + 0.05 * format_valid
)
```

之后再实验 efficiency, test coverage, tool usage, steps。每次只增加一个 reward component。

---

### 1B — AI Code-Gen (Three Prompts, Run Sequentially)

#### Prompt #3 — Dataset Layer (§43)

```text
Implement the dataset layer.

Create:
- src/llm_lab/data/schema.py
- src/llm_lab/data/loader.py
- src/llm_lab/data/generator.py
- scripts/validate_dataset.py
- tests/test_dataset.py

Define a versioned JSONL schema for Python coding tasks.

Each task must contain:
- id
- version
- category
- difficulty
- prompt
- visible_tests
- hidden_tests
- optional starter_code
- metadata

Support train/dev/test split validation.

The test set must be treated as immutable.

Add checks for:
- duplicate IDs
- duplicate prompts
- train/test leakage
- malformed JSON
- missing hidden tests
- invalid difficulty values

Do not implement model training.
```

#### Prompt #4 — Verifier (§13, §44)

```text
Implement a secure Python coding verifier.

IMPORTANT:
Never execute model-generated Python directly on the Windows host.

Implement:
- src/llm_lab/verifier/interface.py
- src/llm_lab/verifier/docker_verifier.py
- tests/test_verifier.py

Use Docker for execution.

The verifier must:
- disable network access
- use an ephemeral container
- mount only the candidate workspace
- enforce CPU limits
- enforce memory limits
- enforce timeout
- capture stdout/stderr
- return structured results
- never mutate the source dataset
- clean up containers after execution

Return:
reward
passed
tests_passed
tests_total
timeout
syntax_error
runtime_error
stderr
duration_seconds

Implement unit tests using trusted toy examples.
```

Verifier interface (§13):

```python
class Verifier:
    def verify(self, task: dict, candidate_code: str) -> dict:
        ...
```

统一返回:

```json
{
  "reward": 1.0,
  "passed": true,
  "tests_passed": 12,
  "tests_total": 12,
  "timeout": false,
  "syntax_error": false,
  "runtime_error": false,
  "stderr": "",
  "duration_seconds": 0.83
}
```

#### Prompt #5 — Baseline Evaluation + Metrics + Reporting (§16, §17, §18, §19, §45)

```text
Implement the benchmark evaluator.

Create:
- src/llm_lab/evaluation/evaluator.py
- src/llm_lab/evaluation/metrics.py
- src/llm_lab/reporting/experiment.py

Requirements:
- evaluate Pass@1, Pass@4, Pass@8
- run hidden tests
- support deterministic seed
- record generation latency
- record token counts
- record peak VRAM
- record peak RAM
- save per-task results as JSONL
- save experiment summary as JSON

CLI:

python -m llm_lab.evaluation.evaluator \
  --config configs/eval.yaml

Do not modify the test dataset.

Do not implement SFT or RL.
```

Evaluator output (§16):

```text
=== Evaluation ===

Tasks:              1000

Pass@1:             0.XX
Pass@4:             0.XX
Pass@8:             0.XX

Mean completion:    XXX tokens
Median completion:  XXX tokens

Timeout rate:       X.X%
Syntax error rate:  X.X%

Mean latency:       X.XX sec
Peak VRAM:          X.XX GB
Peak RAM:           XX.X GB
```

Metrics 定义 (§17):
- **Pass@1**: 一次生成就正确 = successful tasks / tasks
- **Pass@K**: K 次生成至少有一个成功 (K = 1, 4, 8)

Additional metrics to record (§18):

| Category | Metrics |
|----------|---------|
| Capability | Pass@1, Pass@4, Pass@8, OOD Pass@1 |
| Efficiency | generation tokens, seconds/task, GPU hours, tokens/second |
| Training health | loss, reward, reward_std, entropy, KL, grad_norm, completion length |
| Hardware | peak VRAM, peak RAM, GPU utilization, temperature |

Experiment tracking file (§19):

```json
{
  "experiment_id": "exp02_grpo_v1",
  "model": "Qwen/Qwen3.5-2B",
  "base_checkpoint": "...",
  "dataset": "train_v1",
  "test_dataset": "test_v1",
  "method": "GRPO",
  "hyperparameters": { "num_generations": 4, "learning_rate": 1e-5, ... },
  "metrics": { "pass_at_1": 0.0, "pass_at_4": 0.0, "pass_at_8": 0.0 },
  "hardware": { "peak_vram_gb": 0.0, "peak_ram_gb": 0.0 },
  "training": { "wall_hours": 0.0, "gpu_hours": 0.0, "tokens": 0 }
}
```

### 1C — Manual Validation

- [ ] Generate 100 toy tasks (Dataset v0)
- [ ] `python scripts/validate_dataset.py` → passes
- [ ] Docker verifier runs trusted toy examples → correct rewards
- [ ] Full pipeline: Qwen3.5-2B → generate → verify → metrics on 100 tasks
- [ ] Experiment log written to `experiments/exp00_baseline/`

---

## Phase 2: SFT (Supervised Fine-Tuning)

**Goal**: QLoRA SFT on Qwen3.5-2B, prove SFT helps without hurting OOD.

**Milestone M2 (§54)**: `OOD Pass@1` 没有明显恶化, SFT Pass@1 > Base Pass@1.

---

### 2A — Manual Setup

#### Scale Dataset to v1 (500 tasks) then v2 (5,000 tasks) (§52)

First SFT: use 500 tasks. Only scale to 5K after verifier + pipeline are proven stable.

#### SFT Experiment Design (§23, §24)

第一轮: 5K data, 1 epoch. 第二轮: 10K data. 比较 5K → 10K.

每个 checkpoint 记录:

```text
train loss, eval loss, Pass@1, Pass@4, Pass@8, OOD Pass@1, avg output tokens, GPU hours, VRAM
```

最终表:

| Model   | Pass@1 | Pass@4 | Pass@8 | OOD |
| ------- | -----: | -----: | -----: | --: |
| Base    |        |        |        |     |
| SFT 5K  |        |        |        |     |
| SFT 10K |        |        |        |     |

---

### 2B — AI Code-Gen (One Prompt)

#### Prompt #6 — SFT (§22, §46)

```text
Implement QLoRA SFT for Qwen/Qwen3.5-2B using Hugging Face TRL.

Requirements:
- 4-bit NF4 quantization
- PEFT LoRA
- gradient checkpointing
- micro batch size 1
- configurable gradient accumulation
- YAML configuration
- checkpoint saving
- resume support
- experiment manifest
- deterministic seed

Start with:
max_seq_length=4096
per_device_train_batch_size=1
gradient_accumulation_steps=8
learning_rate=1e-5
num_train_epochs=1
LoRA r=16
LoRA alpha=32

Do not modify evaluation code.
Do not modify test.jsonl.
Log:
- train loss
- learning rate
- GPU memory
- training time

Add a smoke test that does not perform a full training run.
```

SFT config reference (`configs/sft_v1.yaml`, §22):

```yaml
experiment:
  id: exp01_sft_v1

model:
  id: Qwen/Qwen3.5-2B

quantization:
  enabled: true
  bits: 4
  quant_type: nf4
  compute_dtype: bfloat16

lora:
  enabled: true
  r: 16
  alpha: 32
  dropout: 0.05

training:
  dataset: data/train.jsonl
  eval_dataset: data/dev.jsonl
  max_seq_length: 4096
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 0.00001
  num_train_epochs: 1
  gradient_checkpointing: true
  logging_steps: 10
  save_steps: 250

output:
  dir: experiments/exp01_sft_v1
```

> [!NOTE]
> QLoRA uses 4-bit base model + LoRA. NF4 is the Hugging Face bitsandbytes recommended 4-bit training configuration.

### 2C — Manual Validation

- [ ] SFT training completes (5K data, 1 epoch)
- [ ] Evaluate SFT checkpoint on frozen `test.jsonl`
- [ ] Compare Pass@1/4/8 vs baseline
- [ ] Verify OOD Pass@1 has not significantly degraded
- [ ] Experiment logged to `experiments/exp01_sft/`

---

## Phase 3: GRPO / RLVR

**Goal**: Apply GRPO reinforcement learning with verifiable rewards. Prove RLVR Pass@1 > SFT Pass@1.

**Milestone M3 (§55)**: `RLVR Pass@1 > SFT Pass@1`. Otherwise don't add any fancy tricks.

---

### 3A — Manual Setup

#### GRPO Design Decisions (§25, §26, §27, §28)

**Why only 4 generations?** (§26) Your 8GB VRAM is a hard constraint. TRL docs confirm that GRPO's generation count, context length significantly affect VRAM. Start with `num_generations=4` as baseline. Later experiment `4 vs 8`, not 16.

**Save all rollouts** (§27):

```json
{
  "task_id": "py_00234",
  "responses": [
    { "text": "...", "reward": 0.0 },
    { "text": "...", "reward": 1.0 },
    { "text": "...", "reward": 0.0 },
    { "text": "...", "reward": 1.0 }
  ]
}
```

This file is crucial for later analysis of "为什么 RL 有效？".

**Key training metrics to monitor** (§28):

每 N steps: `reward_mean`, `reward_std`, `completion_length_mean`, `entropy`, `KL`, `grad_norm`

- `reward_mean`: 应该逐渐改善
- `reward_std`: 不能长期直接塌成 0, 否则 GRPO group 内没有学习信号

---

### 3B — AI Code-Gen (One Prompt)

#### Prompt #7 — GRPO (§47)

```text
Implement GRPO/RLVR using the current TRL GRPOTrainer API.

Starting setup:
- base model = SFT checkpoint
- QLoRA
- num_generations=4
- max_prompt_length=1024
- max_completion_length=384
- micro batch size=1
- gradient accumulation=4

Reward:
1.0 if all hidden pytest tests pass
0.0 otherwise

Do NOT add reward shaping.

Requirements:
- record all rollouts
- record reward mean/std
- record completion length
- record training time
- record peak VRAM/RAM
- support checkpoint resume
- save YAML config with experiment
- never modify test.jsonl

Build a reward adapter around the existing verifier.

Use the current TRL API rather than inventing a custom GRPO implementation.
```

GRPO config reference (`grpo_v1.yaml`, §25):

```yaml
experiment:
  id: exp02_grpo_v1

model:
  base_checkpoint: experiments/exp01_sft_v1/checkpoint-final

quantization:
  enabled: true
  bits: 4
  quant_type: nf4

lora:
  enabled: true
  r: 16
  alpha: 32

grpo:
  num_generations: 4
  max_prompt_length: 1024
  max_completion_length: 384
  learning_rate: 0.00001
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 4
  gradient_checkpointing: true

reward:
  type: pytest
  pass_reward: 1.0
  fail_reward: 0.0

dataset:
  train: data/train.jsonl
  eval: data/dev.jsonl

logging:
  output_dir: experiments/exp02_grpo_v1
  log_completions: true
```

> [!IMPORTANT]
> TRL 官方当前已提供 `GRPOTrainer` 和自定义 reward function 接口。**让 AI 遵循当前 API，而不是自己实现 GRPO** 很重要。

### 3C — Manual Validation

- [ ] GRPO training runs without OOM
- [ ] `reward_mean` trends upward, `reward_std` doesn't collapse
- [ ] Evaluate GRPO checkpoint on frozen `test.jsonl`
- [ ] Verify RLVR Pass@1 > SFT Pass@1
- [ ] Rollout files saved for later analysis
- [ ] Experiment logged to `experiments/exp02_grpo/`

---

## Phase 4: Hard Example Mining + RLVR v2

**Goal**: Mine hard examples from the current model, run ablation studies on data composition, prove hard mining adds value.

**Milestone M4 (§56)**: RLVR + Hard Mining > RLVR alone.

---

### 4A — Manual Setup

#### Mining Classification (§29, §30)

Run current model on 10K tasks, sample 4 per task, verify, calculate per-task Pass@4:

```text
Solved:     Pass@4 >= 0.75
Borderline: 0.25 <= Pass@4 < 0.75
Hard:       0 < Pass@4 < 0.25
Impossible: Pass@4 == 0
```

重点研究 **Borderline**, 不是 Impossible, 因为 borderline 最容易产生有效学习信号。

#### Mining Ablation Design (§31)

Three experiments:

| Strategy | Composition |
|----------|-------------|
| A — Random | 100% random |
| B — Hard | 100% hard/borderline |
| C — Mixed | 70% hard + 30% random |

最终:

| Strategy | Pass@1 | OOD | GPU-hours |
| -------- | -----: | --: | --------: |
| Random   |        |     |           |
| Hard     |        |     |           |
| Mixed    |        |     |           |

这样你才能真正回答: **hard-example mining 是否值得？**

---

### 4B — AI Code-Gen (One Prompt)

#### Prompt #8 — Hard Mining (§48)

```text
Implement hard-example mining.

For each task:
- sample 4 model responses
- verify all responses
- calculate per-task Pass@4

Classify:
solved: Pass@4 >= 0.75
borderline: 0.25 <= Pass@4 < 0.75
hard: 0 < Pass@4 < 0.25
impossible: Pass@4 == 0

Create:
- src/llm_lab/data/miner.py
- tests/test_miner.py

Output:
- mining_report.json
- hard.jsonl
- borderline.jsonl
- mixed.jsonl

Never include test.jsonl examples.

Add experiment IDs and dataset versions to every output.
```

### 4C — Manual Validation

- [ ] Mining runs on full training pool
- [ ] Three ablation experiments (Random/Hard/Mixed) completed
- [ ] Results fill the ablation comparison table
- [ ] RLVR v2 (with best mining strategy) evaluated on `test.jsonl`
- [ ] Experiment logged to `experiments/exp03_mining/`

---

## Phase 5: Test-Time Compute Scaling

**Goal**: Measure test-time scaling by varying K without any training. Determine if the model has scaling potential.

**Milestone M5 (§57)**: 验证 `1 → 2 → 4 → 8 → 16 samples` 是否出现 `Pass@K ↑`, 然后研究: 模型到底是"不会"，还是"知道但不会稳定找到"。

---

### 5A — AI Code-Gen (One Prompt)

#### Prompt #9 — Test-Time Search (§32, §49)

```text
Implement test-time sampling evaluation.

Evaluate:
K = 1, 2, 4, 8, 16

For each K:
- generate K candidates
- verify each candidate
- record whether at least one succeeds
- record total tokens
- record latency

Produce:
- Pass@K table
- tokens per task
- seconds per task
- capability-vs-compute plot

Do not train the model.
```

Expected output — capability-vs-compute curve (§32):

```text
Pass@K
   │
   │                  ●
   │             ●
   │         ●
   │      ●
   │   ●
   └────────────────────────
         inference compute
```

This tells you: 你的模型是否具有 test-time scaling potential.

### 5B — Manual Validation

- [ ] Pass@K results for K = 1, 2, 4, 8, 16
- [ ] Capability-vs-compute plot generated
- [ ] Experiment logged to `experiments/exp04_search/`

---

## Phase 6: Tool-Use Agent + 4B Scaling

**Goal**: Add tool-use capability, then replicate core experiments on Qwen3.5-4B.

---

### 6A — Manual Setup

#### Tool-Use Design (§33, §34)

Tools:

```text
read_file()
write_file()
run_tests()
```

Environment:

```text
workspace/
    main.py
    tests/
```

Agent loop:

```text
inspect → edit → run_tests → observe → fix → run_tests
```

TRL `GRPOTrainer` already provides environment/tool training interface.

#### Agent Metrics (§34)

| Metric | Definition |
|--------|-----------|
| Task completion rate | completed tasks / total tasks |
| Steps to success | 平均多少 tool calls |
| Recovery rate | $\frac{failure \rightarrow success}{initial\ failures}$ |

Recovery rate 这个指标特别有价值。

#### 4B Scaling (§35)

当 2B 全流程跑完 (Base → SFT → RLVR → Hard mining)，换 Qwen3.5-4B 只重复最核心实验 (Base, SFT, RLVR)。不用从头把所有 agent experiment 再做一遍。

---

### 6B — AI Code-Gen

This phase is primarily extension work — use the existing codebase patterns. No dedicated prompt template provided in the original proposal.

### 6C — Manual Validation

- [ ] Agent completes tool-use tasks
- [ ] Recovery rate measured
- [ ] 4B base/SFT/RLVR results compared to 1.7B
- [ ] All experiments logged

---

## Phase 7: Research Dashboard + Final Report

**Goal**: Aggregate all experiments into a unified reporting pipeline.

---

### 7A — AI Code-Gen (One Prompt)

#### Prompt #10 — Research Dashboard (§36, §37, §38, §50)

```text
Build a research reporting pipeline.

Read all experiment manifests and metrics.

Produce:
1. experiment_log.csv
2. comparison table
3. Pass@1 vs experiment plot
4. OOD Pass@1 vs experiment plot
5. capability gain per GPU-hour
6. Pass@K vs inference compute
7. reward vs training step
8. hardware utilization summary

Never silently discard failed experiments.

A failed experiment must appear in the report with status=failed
and an error summary.
```

#### Experiment Log Fields (§36)

```text
reports/experiment_log.csv
```

Fields:

```text
experiment_id, date, model, method, dataset_version, train_examples,
pass_at_1, pass_at_4, pass_at_8, ood_pass_at_1,
mean_reward, reward_std, eval_loss,
mean_completion_tokens, mean_latency, timeout_rate,
gpu_hours, tokens_trained, tokens_generated,
peak_vram_gb, peak_ram_gb,
learning_rate, batch_size, grad_accum, num_generations,
context_length, max_completion_length, lora_rank
```

#### Required Figures (§37)

| Figure | Content |
|--------|---------|
| 1 | Pass@1 vs Experiment |
| 2 | Pass@4 vs Experiment |
| 3 | OOD Pass@1 vs Experiment |
| 4 | Capability gain / GPU-hour |
| 5 | Pass@K vs inference compute |
| 6 | Reward vs training step |

#### ROI — The Most Important Metric (§38)

$$ROI = \frac{Pass@1_{new} - Pass@1_{base}}{GPUHours}$$

Example:

```text
SFT:         +4.2 points / 3 hours  = 1.4 points/hour
RLVR:        +9.5 points / 14 hours = 0.68 points/hour
Hard Mining:  +6.1 points / 4 hours = 1.53 points/hour
```

> **Hard Mining 反而比 RLVR 更高 ROI.** 这就是你真正要发现的东西。

#### Benchmark Design (§58)

核心实验使用**自己生成、自己冻结的 benchmark** (`test_v1`, 1,000 tasks, hidden tests, randomized inputs)。这样你能更可靠地判断模型能力是否真的增加，避免:

```text
benchmark ↑
real capability ?
```

### 7B — Manual Validation

- [ ] `experiment_log.csv` contains all experiments
- [ ] All 6 figures generated
- [ ] ROI calculated for each method
- [ ] Failed experiments appear with `status=failed`

---

## Quick Reference: Timeline (§51)

| Time | Phase | Work |
|------|-------|------|
| Day 1 | Phase 0 | Environment + doctor.py + Qwen inference |
| Day 2 | Phase 1 | Dataset schema + 100 toy tasks + verifier |
| Day 3 | Phase 1 | 1K baseline evaluation + metrics/reporting |
| Day 4–5 | Phase 2 | 5K dataset + QLoRA SFT |
| Day 6–7 | Phase 2 | SFT evaluation |
| Week 2 | Phase 3 | GRPO/RLVR |
| Week 3 | Phase 4 | Hard mining + RLVR v2 + ablation |
| Week 4 | Phase 5–6 | Test-time scaling + tool-use prototype |
| After | Phase 6–7 | 4B scaling + final dashboard |

---

## Quick Reference: All AI Code-Gen Prompts

| # | Prompt | Phase | Creates |
|---|--------|-------|---------|
| 1 | Environment + Doctor | 0 | `scripts/doctor.py`, `src/llm_lab/hardware.py`, `tests/test_smoke.py` |
| 2 | Model Loading + Inference | 0 | `src/llm_lab/inference/generate.py`, `configs/baseline.yaml`, `scripts/download_model.py` |
| 3 | Dataset Layer | 1 | `src/llm_lab/data/schema.py`, `loader.py`, `generator.py`, `scripts/validate_dataset.py`, `tests/test_dataset.py` |
| 4 | Secure Verifier | 1 | `src/llm_lab/verifier/interface.py`, `docker_verifier.py`, `tests/test_verifier.py` |
| 5 | Baseline Evaluation | 1 | `src/llm_lab/evaluation/evaluator.py`, `metrics.py`, `src/llm_lab/reporting/experiment.py` |
| 6 | QLoRA SFT | 2 | `src/llm_lab/training/sft.py`, `configs/sft_v1.yaml` |
| 7 | GRPO / RLVR | 3 | `src/llm_lab/training/grpo.py`, `configs/grpo_v1.yaml` |
| 8 | Hard Mining | 4 | `src/llm_lab/data/miner.py`, `tests/test_miner.py` |
| 9 | Test-Time Search | 5 | Evaluation extension |
| 10 | Research Dashboard | 7 | `src/llm_lab/reporting/plots.py`, `reports/` |

---

## Quick Reference: Milestone Checklist

| Milestone | Gate Condition |
|-----------|----------------|
| M0 | doctor.py → READY, inference works |
| M1 | Full loop (prompt → model → verify → reward → log) stable on 100 tasks |
| M2 | SFT Pass@1 > Base Pass@1, OOD not degraded |
| M3 | RLVR Pass@1 > SFT Pass@1 |
| M4 | RLVR + Hard Mining > RLVR alone |
| M5 | Pass@K scaling curve produced |

> [!CAUTION]
> **没做到当前阶段的 milestone，不要进入下一阶段。** (§21)
