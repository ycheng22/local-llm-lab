import json
import pytest
from llm_lab_utils import (
    TaskSchema, TestCase, SubprocessVerifier, load_dataset, save_dataset,
    pass_at_k, aggregate_pass_at_k,
)
from llm_lab_utils.miner import classify_task
from llm_lab_utils.evaluator import extract_code
from llm_lab_utils.agent_loop import extract_action
from llm_lab_utils.agent_workspace import AgentWorkspace
from llm_lab_utils.generator import generate_toy_tasks


@pytest.fixture
def dummy_task():
    return {
        "id": "test_001", "version": 1, "category": "math", "difficulty": 0.5,
        "prompt": "Add two numbers",
        "visible_tests": [{"input": "add(1, 1)", "expected": "2"}],
        "hidden_tests": [{"input": "add(2, 2)", "expected": "4"}],
    }


def test_verifier_pass(dummy_task):
    r = SubprocessVerifier().verify(dummy_task, "def add(a, b): return a + b")
    assert r["passed"] is True and r["reward"] == 1.0 and r["tests_passed"] == 2


def test_verifier_wrong_answer(dummy_task):
    r = SubprocessVerifier().verify(dummy_task, "def add(a, b): return a - b")
    assert r["passed"] is False and r["runtime_error"] is True


def test_verifier_syntax_error(dummy_task):
    r = SubprocessVerifier().verify(dummy_task, "def add(a, b) return a + b")
    assert r["passed"] is False and r["syntax_error"] is True


def test_verifier_runtime_error(dummy_task):
    r = SubprocessVerifier().verify(dummy_task, "def add(a, b): return a / 0")
    assert r["passed"] is False and r["runtime_error"] is True


def test_verifier_timeout(dummy_task):
    r = SubprocessVerifier(timeout=2.0).verify(dummy_task, "def add(a, b):\n    while True: pass")
    assert r["passed"] is False and r["timeout"] is True


def test_verifier_all_reference_solutions_pass():
    v = SubprocessVerifier()
    for task in generate_toy_tasks(8):
        assert v.verify(task.model_dump(), task.reference_solution)["passed"], task.id


def test_dataset_roundtrip(tmp_path):
    tasks = generate_toy_tasks(4)
    save_dataset(tasks, tmp_path / "d.jsonl")
    loaded = load_dataset(tmp_path / "d.jsonl")
    assert [t.id for t in loaded] == [t.id for t in tasks]


def test_pass_at_k():
    assert pass_at_k(8, 0, 1) == 0.0
    assert pass_at_k(8, 8, 4) == 1.0
    assert pass_at_k(4, 1, 1) == pytest.approx(0.25)
    m = aggregate_pass_at_k([{"n": 8, "c": 2}], [1, 8])
    assert m["pass_at_1"] == pytest.approx(0.25) and m["pass_at_8"] == 1.0


def test_classify_task():
    assert [classify_task(c, 4) for c in range(5)] == ["impossible", "hard", "borderline", "solved", "solved"]


def test_extract_code():
    assert extract_code("x\n```python\nprint(1)\n```\ny") == "print(1)"
    assert extract_code("```\nprint(2)\n```") == "print(2)"
    assert extract_code("print(3)") == "print(3)"


def test_extract_action():
    assert extract_action("Action: run_tests\nAction Input: none")[0] == "run_tests"
    a, inp, content = extract_action("Action: write_file\nAction Input: main.py\n```python\ndef f(): pass\n```")
    assert (a, inp) == ("write_file", "main.py") and "def f()" in content


def test_agent_workspace_roundtrip(dummy_task):
    ws = AgentWorkspace()
    try:
        ws.setup_task(dummy_task, initial_code="")
        assert ws.run_tests()[0] is False
        ws.write_file("main.py", "def add(a, b):\n    return a + b\n")
        ok, msg = ws.run_tests()
        assert ok, msg
    finally:
        ws.cleanup()


def test_paired_bootstrap_diff():
    from llm_lab_utils import paired_bootstrap_diff
    a = {f"t{i}": 0 for i in range(50)}
    b = {f"t{i}": 8 for i in range(50)}          # b strictly better on every task
    r = paired_bootstrap_diff(a, b, n=8, k=1)
    assert r["diff"] == 1.0 and r["ci95_low"] > 0.99 and r["p_better"] == 1.0
    same = paired_bootstrap_diff(a, a, n=8, k=1)
    assert same["diff"] == 0.0 and same["p_better"] == 0.0
