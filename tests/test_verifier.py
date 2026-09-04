import pytest
from llm_lab.verifier import DockerVerifier
from llm_lab.data import TaskSchema, TestCase

@pytest.fixture
def dummy_task():
    return {
        "id": "test_001",
        "version": 1,
        "category": "math",
        "difficulty": 0.5,
        "prompt": "Add two numbers",
        "visible_tests": [{"input": "add(1, 1)", "expected": "2"}],
        "hidden_tests": [{"input": "add(2, 2)", "expected": "4"}]
    }

def test_verifier_pass(dummy_task):
    verifier = DockerVerifier()
    code = "def add(a, b): return a + b"
    result = verifier.verify(dummy_task, code)
    assert result["passed"] is True
    assert result["reward"] == 1.0
    assert result["tests_passed"] == 2

def test_verifier_syntax_error(dummy_task):
    verifier = DockerVerifier()
    code = "def add(a, b) return a + b" # Missing colon
    result = verifier.verify(dummy_task, code)
    assert result["passed"] is False
    assert result["syntax_error"] is True

def test_verifier_runtime_error(dummy_task):
    verifier = DockerVerifier()
    code = "def add(a, b): return a / 0" # Div by zero
    result = verifier.verify(dummy_task, code)
    assert result["passed"] is False
    assert result["runtime_error"] is True

def test_verifier_timeout(dummy_task):
    verifier = DockerVerifier(timeout=2.0)
    code = "def add(a, b):\n    while True: pass" # Infinite loop
    result = verifier.verify(dummy_task, code)
    assert result["passed"] is False
    assert result["timeout"] is True
