import pytest
from llm_lab.agent.loop import extract_action

def test_extract_action_react_write_file():
    text = """Thought: I will write the code.
Action: write_file
Action Input: main.py
Action Content:
```python
def add(a, b):
    return a + b
```"""
    action, action_input, content = extract_action(text)
    assert action == "write_file"
    assert action_input == "main.py"
    assert "def add(a, b):" in content

def test_extract_action_react_run_tests():
    text = """Thought: Running tests.
Action: run_tests
Action Input: none"""
    action, action_input, content = extract_action(text)
    assert action == "run_tests"

def test_extract_action_react_submit():
    text = """Thought: Tests passed.
Action: submit
Action Input: none"""
    action, action_input, content = extract_action(text)
    assert action == "submit"

def test_extract_action_fallback_code_block():
    text = """I'll implement the function:

```python
def solve(s):
    return s[::-1]
```
Let me know if this works."""
    action, action_input, content = extract_action(text)
    assert action == "write_file"
    assert action_input == "main.py"
    assert "def solve(s):" in content

def test_extract_action_fallback_run_tests():
    text = """Let me run the tests now:
```python
run_tests()
```"""
    action, action_input, content = extract_action(text)
    assert action == "run_tests"

def test_extract_action_fallback_read_file():
    text = """Let me read the file first:
read_file
"""
    action, action_input, content = extract_action(text)
    assert action == "read_file"
