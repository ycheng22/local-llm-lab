import re
from typing import Dict, Any, List, Tuple
from .agent_workspace import AgentWorkspace

SYSTEM_PROMPT = """You are an autonomous AI coding agent. You have access to a workspace where you can read, write, and test Python code.

You must solve the user's task using the following tools:
1. read_file
   Usage:
   Action: read_file
   Action Input: main.py

2. write_file
   Usage:
   Action: write_file
   Action Input: main.py
   Action Content:
   ```python
   def solve(x):
       return x + 1
   ```

3. run_tests
   Usage:
   Action: run_tests
   Action Input: none

4. submit
   Usage:
   Action: submit
   Action Input: none

INSTRUCTIONS:
- You must always format your response with exactly ONE Action per turn.
- After calling `write_file`, always call `run_tests` to test your solution.
- If the tests fail, inspect the failure trace, update `main.py` using `write_file`, and run `run_tests` again.
- When `run_tests` passes, call `submit` to finish the task.

EXAMPLE WORKFLOW:
User: Task Prompt: Write a function `add(a, b)` that returns the sum of two numbers.
Assistant:
Thought: I will implement the add function in main.py.
Action: write_file
Action Input: main.py
Action Content:
```python
def add(a, b):
    return a + b
```
User: Observation: Successfully wrote 38 bytes to main.py. Please call 'run_tests' next.
Assistant:
Thought: I need to verify my solution by running the tests.
Action: run_tests
Action Input: none
User: Observation: PASSED: All 3 tests passed!
Assistant:
Thought: All tests passed. I will submit the solution.
Action: submit
Action Input: none
"""

def extract_action(text: str) -> Tuple[str, str, str]:
    """Parses ReAct style action from text with robust fallbacks for small code LLMs."""
    # 1. Check for standard ReAct Action syntax
    action_match = re.search(r"(?:^|\n)\s*(?:\*\*)?Action(?:\*\*)?:\s*([a-zA-Z_]+)", text, re.IGNORECASE)
    input_match = re.search(r"(?:^|\n)\s*(?:\*\*)?Action Input(?:\*\*)?:\s*(.*?)(?:\n|$)", text, re.IGNORECASE)
    
    action = action_match.group(1).lower().strip() if action_match else None
    action_input = input_match.group(1).strip() if input_match else ""
    action_content = ""
    
    if action == "write_file":
        content_match = re.search(r"(?:Action Content:)?\s*```(?:python)?\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if content_match:
            action_content = content_match.group(1)
        return "write_file", (action_input or "main.py"), action_content
    elif action in ("read_file", "run_tests", "submit"):
        return action, action_input, ""

    # 2. Fallback parsing for natural / zero-shot LLM output
    # Check for explicit submit
    if re.search(r"\b(?:Action:\s*)?submit(?:\(\))?\b", text, re.IGNORECASE) and not re.search(r"\bdef\s+", text):
        return "submit", "none", ""

    # Check for python function definitions inside code blocks -> treat as write_file
    code_blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    def_blocks = [c for c in code_blocks if "def " in c]
    if def_blocks:
        return "write_file", "main.py", def_blocks[0]

    # Check for run_tests (e.g. `run_tests()` or `run_tests`)
    if re.search(r"\b(?:run_tests|test_cases|pytest|unittests)\b", text, re.IGNORECASE):
        return "run_tests", "none", ""

    # Check for read_file
    if re.search(r"\bread_file\b", text, re.IGNORECASE):
        return "read_file", "main.py", ""

    # Check for raw function definition without markdown backticks
    if "def " in text and ("return " in text or "pass" in text):
        raw_match = re.search(r"(def\s+[a-zA-Z0-9_]+\s*\(.*?\):[\s\S]*)", text)
        if raw_match:
            return "write_file", "main.py", raw_match.group(1).strip()

    return None, "", ""

def run_agent_loop(task: Dict[str, Any], model, tokenizer, max_steps: int = 5,
                   max_new_tokens: int = 512, temperature: float = 0.2) -> Dict[str, Any]:
    """
    Executes the autonomous agent loop for a given task.
    The run_tests tool shows only the VISIBLE tests; `passed` is decided at the end by grade() (visible + hidden),
    so tool feedback cannot leak hidden tests.
    """
    workspace = AgentWorkspace()
    workspace.setup_task(task, initial_code=task.get("starter_code", ""))
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Task Prompt:\n{task['prompt']}\n\nThe starter code is already written to main.py. Your goal is to write a complete solution in main.py and pass the hidden tests."}
    ]
    
    result = {
        "task_id": task["id"],
        "passed": False,
        "steps": 0,
        "history": [],
        "recovery": False,
        "parse_errors": 0,
        "tokens_generated": 0,
        "truncated_steps": 0,
        "visible_passed": False,
    }
    
    first_test_passed = None
    
    try:
        for step in range(max_steps):
            result["steps"] += 1
            
            # Generate response
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature, # Low temperature for more reliable formatting
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
            gen_tokens = outputs[0][inputs.input_ids.shape[1]:]
            eos = tokenizer.eos_token_id
            n_new = int((gen_tokens != eos).sum())          # pad == eos, so this excludes padding/stop tokens
            result["tokens_generated"] += n_new
            if n_new >= max_new_tokens:
                result["truncated_steps"] += 1
            completion = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            
            messages.append({"role": "assistant", "content": completion})
            result["history"].append({"role": "assistant", "content": completion})
            
            action, action_input, action_content = extract_action(completion)
            
            observation = ""
            if not action:
                result["parse_errors"] += 1
                observation = "Error: Could not parse Action. Please provide your solution in a ```python code block or use 'Action: run_tests' / 'Action: submit'."
            elif action == "read_file":
                observation = workspace.read_file(action_input or "main.py")
                observation += "\nYou can now edit the file with 'Action: write_file'."
            elif action == "write_file":
                observation = workspace.write_file(action_input or "main.py", action_content)
                observation += "\nFile updated. Please call 'Action: run_tests' to verify your solution."
            elif action == "run_tests":
                passed, obs = workspace.run_tests()
                observation = obs
                
                # Tracking recovery metrics (on the visible tests the agent can see)
                if first_test_passed is None:
                    first_test_passed = passed
                elif first_test_passed is False and passed is True:
                    result["recovery"] = True
                
                if passed:
                    observation += "\nAll tests passed! Please call 'Action: submit' to complete the task."
                else:
                    observation += "\nTests failed. Please inspect the error and update main.py using write_file."
                    
            elif action == "submit":
                observation = "Task submitted."
                break
            else:
                observation = f"Error: Unknown action {action}."
                
            messages.append({"role": "user", "content": f"Observation:\n{observation}"})
            result["history"].append({"role": "user", "content": observation})
            
        # Final grading on visible + hidden tests, whether the agent submitted or ran out of steps.
        result["passed"] = workspace.grade()
        result["visible_passed"] = workspace.run_tests()[0]
        result["first_test_passed"] = first_test_passed
        if first_test_passed is False and result["visible_passed"]:
            result["recovery"] = True
                
    finally:
        workspace.cleanup()
        
    return result


def evaluate_agent(model, tokenizer, tasks, max_steps: int = 5, progress: bool = True,
                   time_budget_s: float = None, seed: int = None, **loop_kwargs):
    """
    Runs the agent over tasks. Returns (metrics, traces).
    seed: if given, torch is re-seeded to seed+task_index before every task (deterministic sampling per task).
    time_budget_s: stop starting new tasks after this many seconds (metrics then cover the finished tasks only
    and report n_tasks_run, so a truncated run is never mistaken for a full one).
    """
    import time

    iterator = tasks
    if progress:
        from tqdm.auto import tqdm
        iterator = tqdm(tasks)

    traces = []
    t0 = time.time()
    for idx, task_obj in enumerate(iterator):
        if time_budget_s is not None and time.time() - t0 > time_budget_s:
            break
        if seed is not None:
            import torch
            torch.manual_seed(seed + idx)
        task = task_obj.model_dump() if hasattr(task_obj, "model_dump") else task_obj
        traces.append(run_agent_loop(task, model, tokenizer, max_steps=max_steps, **loop_kwargs))

    n = len(traces)
    passed = [t for t in traces if t["passed"]]
    # Recovery: of the runs whose first visible test failed, the share that ended with passing visible tests.
    first_failed = [t for t in traces if t.get("first_test_passed") is False]
    steps_total = sum(t["steps"] for t in traces)
    metrics = {
        "n_tasks_run": n,
        "task_completion_rate": len(passed) / n if n else 0,
        "steps_to_success": sum(t["steps"] for t in passed) / len(passed) if passed else 0,
        "recovery_rate": sum(t["recovery"] for t in traces) / len(first_failed) if first_failed else 0,
        "mean_steps": steps_total / n if n else 0,
        "parse_error_rate": sum(t["parse_errors"] for t in traces) / steps_total if steps_total else 0,
        "truncated_step_rate": sum(t["truncated_steps"] for t in traces) / steps_total if steps_total else 0,
        "tokens_generated": sum(t["tokens_generated"] for t in traces),
        # passed the visible example the agent could test against, failed the hidden tests
        "false_friend_rate": sum(1 for t in traces if t["visible_passed"] and not t["passed"]) / n if n else 0,
    }
    return metrics, traces
