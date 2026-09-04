import pytest
import json
from pathlib import Path
from llm_lab.data import TaskSchema, TestCase, load_dataset
from pydantic import ValidationError

def test_schema_valid():
    task = TaskSchema(
        id="test_001",
        version=1,
        category="math",
        difficulty=0.5,
        prompt="Add two numbers",
        visible_tests=[TestCase(input="add(1, 1)", expected="2")],
        hidden_tests=[TestCase(input="add(2, 2)", expected="4")]
    )
    assert task.id == "test_001"
    assert len(task.visible_tests) == 1
    assert task.starter_code == "" # Default

def test_schema_invalid():
    with pytest.raises(ValidationError):
        TaskSchema(id="test_002") # Missing required fields

def test_load_dataset(tmp_path: Path):
    file_path = tmp_path / "test.jsonl"
    task_dict = {
        "id": "test_003",
        "version": 1,
        "category": "string",
        "difficulty": 0.1,
        "prompt": "Reverse a string",
        "visible_tests": [],
        "hidden_tests": []
    }
    with open(file_path, "w") as f:
        f.write(json.dumps(task_dict) + "\n")
        
    tasks = load_dataset(file_path)
    assert len(tasks) == 1
    assert tasks[0].id == "test_003"
