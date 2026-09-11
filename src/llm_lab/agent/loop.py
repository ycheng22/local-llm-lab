import re
import torch
from typing import Dict, Any, List, Tuple
from llm_lab.agent.workspace import AgentWorkspace

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

def run_agent_loop(task: Dict[str, Any], model, tokenizer, max_steps: int = 5) -> Dict[str, Any]:
    """Executes the autonomous agent loop for a given task."""
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
        "recovery": False
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
                max_new_tokens=512,
                temperature=0.2, # Low temperature for more reliable formatting
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
            gen_tokens = outputs[0][inputs.input_ids.shape[1]:]
            completion = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            
            messages.append({"role": "assistant", "content": completion})
            result["history"].append({"role": "assistant", "content": completion})
            
            action, action_input, action_content = extract_action(completion)
            
            observation = ""
            if not action:
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
                
                # Tracking recovery metrics
                if first_test_passed is None:
                    first_test_passed = passed
                elif first_test_passed is False and passed is True:
                    result["recovery"] = True
                
                if passed:
                    result["passed"] = True
                    observation += "\nAll tests passed! Please call 'Action: submit' to complete the task."
                else:
                    observation += "\nTests failed. Please inspect the error and update main.py using write_file."
                    
            elif action == "submit":
                passed, obs = workspace.run_tests()
                result["passed"] = passed
                observation = "Task submitted. " + ("All tests passed!" if passed else f"Tests failed: {obs}")
                break
            else:
                observation = f"Error: Unknown action {action}."
                
            messages.append({"role": "user", "content": f"Observation:\n{observation}"})
            result["history"].append({"role": "user", "content": observation})
            
        # Final pass verification if not already explicitly marked
        if not result["passed"]:
            final_passed, _ = workspace.run_tests()
            result["passed"] = final_passed
            if first_test_passed is False and final_passed:
                result["recovery"] = True
                
    finally:
        workspace.cleanup()
        
    return result
