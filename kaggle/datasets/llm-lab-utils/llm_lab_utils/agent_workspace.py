import os
import sys
import shutil
import tempfile
import subprocess
import time
from typing import Dict, Any, Tuple

from .subprocess_verifier import limit_prelude

class AgentWorkspace:
    """
    Provides a safe, sandboxed environment for the agent to inspect and edit files,
    and run tests against a generated solution in a subprocess.
    """
    
    def __init__(self, timeout: float = 5.0, max_memory_mb: int = 512):
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb
        self.workspace_dir = tempfile.mkdtemp(prefix="agent_workspace_")
        
    def setup_task(self, task_dict: Dict[str, Any], initial_code: str = ""):
        """Sets up the workspace with the initial code and a hidden test script."""
        main_path = os.path.join(self.workspace_dir, "main.py")
        
        code = initial_code or task_dict.get("starter_code", "")
        with open(main_path, "w", encoding="utf-8") as f:
            f.write(code)
            
        self._build_test_script(task_dict)
        
    def _build_test_script(self, task_dict: Dict[str, Any]):
        """Builds a test script that imports main.py and runs all tests."""
        test_path = os.path.join(self.workspace_dir, "test_runner.py")
        
        script = limit_prelude(self.max_memory_mb * 1024 * 1024, int(self.timeout) + 1)
        script += "import sys\n"
        script += "try:\n"
        script += "    from main import *\n"
        script += "except Exception as e:\n"
        script += "    print(f'Import error: {e}', file=sys.stderr)\n"
        script += "    sys.exit(1)\n\n"
        script += "tests_passed = 0\n"
        
        all_tests = task_dict.get("visible_tests", []) + task_dict.get("hidden_tests", [])
        
        for i, test in enumerate(all_tests):
            script += f"try:\n"
            script += f"    actual = {test['input']}\n"
            script += f"    expected = {test['expected']}\n"
            script += f"    if actual != expected:\n"
            script += f"        print(f'Test {i} failed: expected {{expected}}, got {{actual}}', file=sys.stderr)\n"
            script += f"        sys.exit(1)\n"
            script += f"    tests_passed += 1\n"
            script += f"except Exception as e:\n"
            script += f"    print(f'Test {i} error: {{e}}', file=sys.stderr)\n"
            script += f"    sys.exit(1)\n"
            
        script += f"print('ALL_PASS:{len(all_tests)}')\n"
        script += "sys.exit(0)\n"
        
        with open(test_path, "w", encoding="utf-8") as f:
            f.write(script)

    def read_file(self, filename: str) -> str:
        """Tool: read a file from the workspace."""
        path = os.path.join(self.workspace_dir, filename)
        if not os.path.exists(path) or not os.path.abspath(path).startswith(os.path.abspath(self.workspace_dir)):
            return f"Error: File {filename} does not exist."
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"Error reading file: {e}"

    def write_file(self, filename: str, content: str) -> str:
        """Tool: write content to a file in the workspace."""
        path = os.path.join(self.workspace_dir, filename)
        if not os.path.abspath(path).startswith(os.path.abspath(self.workspace_dir)):
            return f"Error: Invalid path {filename}."
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Successfully wrote to {filename}"
        except Exception as e:
            return f"Error writing file: {e}"

    def run_tests(self) -> Tuple[bool, str]:
        """Tool: run the hidden tests in a throwaway subprocess (see SubprocessVerifier for the isolation model)."""
        try:
            process = subprocess.run(
                [sys.executable, "-E", "test_runner.py"],  # not -I: tests must import main.py from cwd
                text=True,
                capture_output=True,
                timeout=self.timeout,
                cwd=self.workspace_dir,
                env={"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0"},
            )

            stdout = process.stdout
            stderr = process.stderr

            output = ""
            if stdout:
                output += f"STDOUT:\n{stdout}\n"
            if stderr:
                output += f"STDERR:\n{stderr}\n"

            passed = process.returncode == 0 and "ALL_PASS" in stdout

            if passed:
                return True, "All tests passed successfully!"
            return False, f"Tests failed.\n{output}"

        except subprocess.TimeoutExpired:
            return False, "Execution timed out."
        except Exception as e:
            return False, f"Internal verification error: {e}"

    def cleanup(self):
        """Removes the temporary workspace directory."""
        if os.path.exists(self.workspace_dir):
            shutil.rmtree(self.workspace_dir, ignore_errors=True)
