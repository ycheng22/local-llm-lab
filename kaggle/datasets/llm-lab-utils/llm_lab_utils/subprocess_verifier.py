import os
import sys
import tempfile
import subprocess
import time
from abc import ABC, abstractmethod
from typing import Dict, Any

from .schema import TaskSchema


class Verifier(ABC):
    """Base interface for code execution verifiers."""

    @abstractmethod
    def verify(self, task: Dict[str, Any], candidate_code: str) -> Dict[str, Any]:
        """
        Returns {reward, passed, tests_passed, tests_total, timeout,
                 syntax_error, runtime_error, stderr, duration_seconds}.
        """


def build_test_script(task_dict: Dict[str, Any], candidate_code: str) -> str:
    """
    Combines candidate code with assertion logic for visible and hidden tests.
    Identical logic to the local DockerVerifier so results stay comparable.
    """
    task = TaskSchema(**task_dict)
    script = candidate_code + "\n\n"

    script += "if __name__ == '__main__':\n"
    script += "    import sys\n"
    script += "    tests_passed = 0\n"

    all_tests = task.visible_tests + task.hidden_tests

    for i, test in enumerate(all_tests):
        script += "    try:\n"
        script += f"        actual = {test.input}\n"
        script += f"        expected = {test.expected}\n"
        script += "        if actual != expected:\n"
        script += f"            print(f'Test {i} failed: expected {{expected}}, got {{actual}}', file=sys.stderr)\n"
        script += "            sys.exit(1)\n"
        script += "        tests_passed += 1\n"
        script += "    except Exception as e:\n"
        script += f"        print(f'Test {i} error: {{e}}', file=sys.stderr)\n"
        script += "        sys.exit(1)\n"

    script += f"    print('ALL_PASS:{len(all_tests)}')\n"
    script += "    sys.exit(0)\n"

    return script


def limit_prelude(max_memory_bytes: int, cpu_seconds: int) -> str:
    """
    Source prepended to the child script: applies POSIX rlimits inside the child itself.
    (Deliberately not subprocess' preexec_fn, which can deadlock when called from threads.)
    On platforms without `resource` (Windows) it is a no-op and only the wall-clock timeout applies.
    """
    return (
        "try:\n"
        "    import resource as _r\n"
        f"    _r.setrlimit(_r.RLIMIT_AS, ({max_memory_bytes}, {max_memory_bytes}))\n"
        f"    _r.setrlimit(_r.RLIMIT_CPU, ({cpu_seconds}, {cpu_seconds}))\n"
        "except Exception:\n"
        "    pass\n"
    )


class SubprocessVerifier(Verifier):
    """
    Executes candidate code in a throwaway subprocess.

    Isolation model: the Kaggle notebook container is itself ephemeral and, with
    Internet turned off, network-isolated. On top of that each candidate runs in
    its own temp directory with a wall-clock timeout and (on Linux) rlimits for
    memory and CPU time.

    NEVER use this on a personal machine with untrusted code; use the Docker
    verifier locally (see AGENTS.md rule 8/9).
    """

    def __init__(self, timeout: float = 5.0, max_memory_mb: int = 512):
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb

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
            "duration_seconds": 0.0,
        }

        test_script = limit_prelude(self.max_memory_mb * 1024 * 1024, int(self.timeout) + 1) + \
            build_test_script(task, candidate_code)

        with tempfile.TemporaryDirectory() as temp_dir:
            script_path = os.path.join(temp_dir, "candidate.py")
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(test_script)

            try:
                process = subprocess.run(
                    [sys.executable, "-I", script_path],
                    text=True,
                    capture_output=True,
                    timeout=self.timeout,
                    cwd=temp_dir,
                    env={"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0"},
                )

                result["stderr"] = process.stderr

                if process.returncode == 0 and f"ALL_PASS:{result['tests_total']}" in process.stdout:
                    result["passed"] = True
                    result["reward"] = 1.0
                    result["tests_passed"] = result["tests_total"]
                elif "SyntaxError" in process.stderr:
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
