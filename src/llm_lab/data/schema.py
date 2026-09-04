from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class TestCase(BaseModel):
    input: str
    expected: str

class TaskSchema(BaseModel):
    """
    Schema for a Python coding task in the Local LLM Lab.
    """
    id: str
    version: int
    category: str
    difficulty: float
    prompt: str
    starter_code: Optional[str] = ""
    visible_tests: List[TestCase] = Field(default_factory=list)
    hidden_tests: List[TestCase] = Field(default_factory=list)
    reference_solution: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
