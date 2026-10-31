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


def test_agent_feedback_does_not_leak_hidden_tests():
    task = {"id": "t", "version": 1, "category": "math", "difficulty": 0.5, "prompt": "p",
            "visible_tests": [{"input": "add(1, 1)", "expected": "2"}],
            "hidden_tests": [{"input": "add(1000, 337)", "expected": "1337"}]}
    ws = AgentWorkspace()
    try:
        ws.setup_task(task)
        ws.write_file("main.py", "def add(a, b):\n    return 2 if (a, b) == (1, 1) else 0\n")   # passes visible only
        ok, msg = ws.run_tests()
        assert ok and "1337" not in msg          # the tool the agent sees uses visible tests only
        assert ws.grade() is False               # final grading still includes the hidden tests
        ws.write_file("main.py", "def add(a, b):\n    return a + b\n")
        assert ws.grade() is True
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


def test_selection_accuracy():
    from llm_lab_utils.metrics import selection_accuracy
    # 1 of 4 passes visible and full; the other 3 fail both -> choosing by visible test is perfect when k covers it
    s = [{"visible": True, "full": True}] + [{"visible": False, "full": False}] * 3
    assert selection_accuracy(s, 4) == 1.0
    assert abs(selection_accuracy(s, 1, draws=4000) - 0.25) < 0.03
    # visible test passes but hidden fails (a "false friend") caps accuracy
    f = [{"visible": True, "full": False}] * 4
    assert selection_accuracy(f, 4) == 0.0


# ---- HumanEval conversion ---------------------------------------------------------------
_HE_OK = {
    "task_id": "HumanEval/900", "entry_point": "double_all",
    "prompt": "from typing import List\n\n\ndef helper(x):\n    return x * 2\n\n\ndef double_all(xs: List[int]) -> List[int]:\n    \"\"\" double each element \"\"\"\n",
    "canonical_solution": "    return [helper(x) for x in xs]\n",
    "test": "METADATA = {}\n\n\ndef check(candidate):\n    import math\n    assert candidate([1, 2]) == [2, 4]\n    assert True, 'noise'\n    assert math.isclose(candidate([1])[0], 2.0)\n    assert not candidate([]) == [1]\n",
}


def test_humaneval_convert_and_verify():
    from llm_lab_utils.humaneval import convert_problem, PreludeVerifier
    task, reason = convert_problem(_HE_OK)
    assert task and not reason and task["id"] == "humaneval_900"
    assert [t["expected"] for t in task["hidden_tests"]] == ["[2, 4]", "True", "True"]
    assert "def helper" in task["metadata"]["prelude"] and "import math" in task["metadata"]["prelude"]
    v = PreludeVerifier()
    # reference solution passes; a candidate that omits the prompt's imports/helpers still works through the prelude
    assert v.verify(task, task["reference_solution"])["passed"]
    assert v.verify(task, "def double_all(xs):\n    return [helper(x) for x in xs]")["passed"]
    assert not v.verify(task, "def double_all(xs):\n    return xs")["passed"]


def test_humaneval_unsupported_problems_are_reported():
    from llm_lab_utils.humaneval import convert_problem
    loop = dict(_HE_OK, test="def check(candidate):\n    for i in range(3):\n        assert candidate([i]) == [2 * i]\n")
    task, reason = convert_problem(loop)
    assert task is None and "non-assert" in reason
    helper = dict(_HE_OK, test="def enc(x):\n    return x\n\ndef check(candidate):\n    assert candidate([1]) == [2]\n")
    assert convert_problem(helper)[0] is None
