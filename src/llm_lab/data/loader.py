import json
from pathlib import Path
from typing import List, Generator
from .schema import TaskSchema

def load_dataset(file_path: str | Path) -> List[TaskSchema]:
    """
    Load a JSONL dataset file and validate it against the TaskSchema.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    
    tasks = []
    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                task = TaskSchema(**data)
                tasks.append(task)
            except Exception as e:
                raise ValueError(f"Error parsing line {line_num} in {path}: {e}")
                
    return tasks

def save_dataset(tasks: List[TaskSchema], file_path: str | Path) -> None:
    """
    Save a list of TaskSchema objects to a JSONL file.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        for task in tasks:
            f.write(task.model_dump_json() + "\n")
