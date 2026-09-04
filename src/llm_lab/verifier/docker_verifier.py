import os
import tempfile
import subprocess
import time
from typing import Dict, Any
from .interface import Verifier
from llm_lab.data import TaskSchema

class DockerVerifier(Verifier):
    """
    Safely executes candidate code against task tests using Docker.
    """
    
    def __init__(self, timeout: float = 3.0, memory_limit: str = "512m"):
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.image = "python:3.11-slim"
        
    def _build_test_script(self, task_dict: Dict[str, Any], candidate_code: str) -> str:
        """
        Combines candidate code with assertion logic for visible and hidden tests.
        """
        task = TaskSchema(**task_dict)
        script = candidate_code + "\n\n"
        
        script += "if __name__ == '__main__':\n"
        script += "    import sys\n"
        script += "    tests_passed = 0\n"
        
        all_tests = task.visible_tests + task.hidden_tests
        
        for i, test in enumerate(all_tests):
            # Evaluate the expected output to compare properly
            script += f"    try:\n"
            script += f"        actual = {test.input}\n"
            script += f"        expected = {test.expected}\n"
            script += f"        if actual != expected:\n"
            script += f"            print(f'Test {i} failed: expected {{expected}}, got {{actual}}', file=sys.stderr)\n"
            script += f"            sys.exit(1)\n"
            script += f"        tests_passed += 1\n"
            script += f"    except Exception as e:\n"
            script += f"        print(f'Test {i} error: {{e}}', file=sys.stderr)\n"
            script += f"        sys.exit(1)\n"
            
        script += f"    print('ALL_PASS:{len(all_tests)}')\n"
        script += "    sys.exit(0)\n"
        
        return script

    def verify(self, task: Dict[str, Any], candidate_code: str) -> Dict[str, Any]:
        start_time = time.time()
        
        result = {
            "reward": 0.0,
            "passed": False,
            "tests_passed": 0,
            "tests_total": len(task.get("visible_tests", [])) + len(task.get("hidden_tests", [])),
            "timeout": False,
            "syntax_error": False,
            "runtime_error": False,
            "stderr": "",
            "duration_seconds": 0.0
        }
        
        test_script = self._build_test_script(task, candidate_code)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            script_path = os.path.join(temp_dir, "candidate.py")
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(test_script)
                
            cmd = [
                "docker", "run", "--rm", "-i",
                "--network", "none",
                "--memory", self.memory_limit,
                "--entrypoint", "python",
                "public.ecr.aws/lambda/python:3.12-rapid-x86_64",
                "-"
            ]
            
            try:
                process = subprocess.run(
                    cmd,
                    input=test_script,
                    text=True,
                    capture_output=True,
                    timeout=self.timeout
                )
                
                stdout = process.stdout
                stderr = process.stderr
                
                result["stderr"] = stderr
                
                if process.returncode == 0 and f"ALL_PASS:{result['tests_total']}" in stdout:
                    result["passed"] = True
                    result["reward"] = 1.0
                    result["tests_passed"] = result["tests_total"]
                else:
                    if "SyntaxError" in stderr:
                        result["syntax_error"] = True
                    else:
                        result["runtime_error"] = True
                    
            except subprocess.TimeoutExpired as e:
                result["timeout"] = True
                result["stderr"] = "Execution timed out"
                if e.stderr:
                    result["stderr"] += "\n" + str(e.stderr)
            except Exception as e:
                result["stderr"] = f"Internal verification error: {e}"

        result["duration_seconds"] = time.time() - start_time
        return result
