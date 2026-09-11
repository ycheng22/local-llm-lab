# Phase 6: Tool-Use Agent & 4B Scaling

The goal of Phase 6 is two-fold:
1. Build a basic Agent loop (`inspect -> edit -> run_tests -> fix`) to evaluate if the model can use tools to recover from coding failures.
2. Prepare configurations for scaling the experiments from `Qwen3.5-2B` to `Qwen3.5-4B`.

## User Review Required

> [!NOTE]
> Please review the architecture for the `AgentWorkspace`. Instead of keeping a long-lived container running (which is complex to manage and prone to hanging), I propose using a **local temporary directory** for the workspace that is mounted into an ephemeral Docker container *only* when the `run_tests()` tool is called. This perfectly satisfies the security requirement while being extremely robust and easy to implement.

## Open Questions

> [!IMPORTANT]
> **Prompt Formatting for Tool-Use:** Qwen3.5 does not natively support OpenAI-style JSON tool calling perfectly out-of-the-box via `transformers.generate` without custom formatting. I plan to use a simple text-based action parser (e.g. `Action: read_file\nPath: main.py`) in the system prompt. Are you comfortable with this text-based ReAct style, or do you want me to attempt using strict JSON schema enforcement?

## Proposed Changes

### 1. Tool-Use Agent Framework

#### [NEW] [src/llm_lab/agent/workspace.py](file:///d:/Github_Clones/local-llm-lab/src/llm_lab/agent/workspace.py)
A secure workspace manager.
- Initializes a temporary directory with `main.py`.
- **Tools**:
  - `read_file(filename)`: Reads from the temp directory.
  - `write_file(filename, content)`: Writes to the temp directory.
  - `run_tests(test_cases)`: Mounts the temp directory to a Docker container (`python:3.11-slim`), runs `pytest` or a test script, and returns the output/errors.

#### [NEW] [src/llm_lab/agent/loop.py](file:///d:/Github_Clones/local-llm-lab/src/llm_lab/agent/loop.py)
The core agent reasoning loop.
- Feeds the system prompt and task to the model.
- Parses actions (`read_file`, `write_file`, `run_tests`, `submit`).
- Executes the action via `AgentWorkspace`.
- Appends the `Observation` back to the model prompt.
- Limits to N max steps (e.g., 5) to prevent infinite loops.

#### [NEW] [src/llm_lab/agent/evaluator.py](file:///d:/Github_Clones/local-llm-lab/src/llm_lab/agent/evaluator.py)
- Runs the agent loop over the evaluation dataset.
- Tracks custom Agent metrics:
  - `task_completion_rate`: Overall success rate.
  - `steps_to_success`: Average iterations taken to succeed.
  - `recovery_rate`: Percentage of tasks where the first `run_tests()` failed, but a subsequent one passed.

---

### 2. Qwen3.5-4B Scaling Configurations

We will duplicate the core configurations for the 4B model so we can compare its capabilities against the 2B baseline.

#### [NEW] [configs/baseline_4b.yaml](file:///d:/Github_Clones/local-llm-lab/configs/baseline_4b.yaml)
- Baseline evaluation config for `Qwen/Qwen3.5-4B`.

#### [NEW] [configs/sft_4b.yaml](file:///d:/Github_Clones/local-llm-lab/configs/sft_4b.yaml)
- QLoRA SFT training config for `Qwen/Qwen3.5-4B`.

#### [NEW] [configs/grpo_4b.yaml](file:///d:/Github_Clones/local-llm-lab/configs/grpo_4b.yaml)
- GRPO/RLVR training config for `Qwen/Qwen3.5-4B`.

### 3. Notebooks for Evaluation

#### [NEW] [notebooks/agent_evaluation.ipynb](file:///d:/Github_Clones/local-llm-lab/notebooks/agent_evaluation.ipynb)
- A Jupyter Notebook that allows you to:
  1. Download or load the local `Qwen3.5-4B` model (or the 2B model) directly from the huggingface cache.
  2. Run the agent evaluation on a small subset of the dataset.
  3. Visualize the `task_completion_rate`, `steps_to_success`, and `recovery_rate` metrics.

---

## Verification Plan

### Automated Tests
- I will write `tests/test_agent_workspace.py` to verify that `read_file`, `write_file`, and `run_tests` operate securely within the temporary directory and don't leak into the host filesystem.

### Manual Verification
- I will execute a smoke test of the agent loop on a single task to ensure it can successfully call tools, observe test failures, and fix its code.
- You can manually run the `download_model.py` script for `Qwen3.5-4B` and kick off the 4B baseline evaluation.
