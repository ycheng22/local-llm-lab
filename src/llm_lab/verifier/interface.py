from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

class Verifier(ABC):
    """
    Base interface for code execution verifiers.
    """
    
    @abstractmethod
    def verify(self, task: Dict[str, Any], candidate_code: str) -> Dict[str, Any]:
        """
        Executes candidate code against the task's visible and hidden tests.
        
        Returns a dictionary with the following schema:
        {
            "reward": float,
            "passed": bool,
            "tests_passed": int,
            "tests_total": int,
            "timeout": bool,
            "syntax_error": bool,
            "runtime_error": bool,
            "stderr": str,
            "duration_seconds": float
        }
        """
        pass
