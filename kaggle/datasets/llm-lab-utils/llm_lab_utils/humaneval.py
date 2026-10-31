"""
HumanEval -> TaskSchema converter (plan Step 19b, external out-of-distribution benchmark).

HumanEval tests are a `check(candidate)` function full of `assert candidate(...) == expected` lines. Our verifier
takes (input expression, expected expression) pairs, so each plain top-level assert in `check` is converted into
one pair: `a == b` becomes (a, b); any other boolean assert becomes (expression, True). Problems whose check()
contains anything else (loops, random inputs, helper definitions) are dropped and listed in the audit, so the
benchmark we report is "the N HumanEval problems expressible as plain asserts", never silently "HumanEval".

Imports and helper functions from the original prompt (and imports used inside check) go into
metadata["prelude"]; PreludeVerifier prepends it to the candidate code, mirroring how the original harness
appends a completion to the full prompt.
"""
import ast
from typing import Any, Dict, List, Optional, Tuple

from .subprocess_verifier import SubprocessVerifier

INSTRUCTION = "Complete the following Python function. Return the full function definition in a ```python code block.\n\n"


class _RenameCandidate(ast.NodeTransformer):
    def __init__(self, entry_point: str):
        self.entry_point = entry_point
        self.used = False

    def visit_Name(self, node):
        if node.id == "candidate":
            self.used = True
            return ast.copy_location(ast.Name(id=self.entry_point, ctx=node.ctx), node)
        return node


def split_prelude(prompt: str, entry_point: str) -> str:
    """Everything in the prompt before the target function (imports and fully defined helper functions)."""
    idx = prompt.rfind(f"def {entry_point}")
    return prompt[:idx] if idx >= 0 else ""


def convert_tests(test_src: str, entry_point: str) -> Tuple[Optional[List[Dict[str, str]]], str, str]:
    """
    Returns (tests, extra_imports, reason). tests is None when the problem is not expressible as plain asserts,
    and reason says why. extra_imports are `import ...` lines found inside check() that the asserts rely on.
    """
    tree = ast.parse(test_src)
    check = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "check":
            check = node
        elif isinstance(node, ast.FunctionDef):
            return None, "", f"helper function '{node.name}' defined next to check()"
    if check is None:
        return None, "", "no check() function"

    tests, imports = [], []
    for st in check.body:
        if isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant):
            continue                                                    # docstring / bare constant
        if isinstance(st, (ast.Import, ast.ImportFrom)):
            imports.append(ast.unparse(st))
            continue
        if not isinstance(st, ast.Assert):
            return None, "", f"non-assert statement in check(): {type(st).__name__}"
        renamer = _RenameCandidate(entry_point)
        test = renamer.visit(st.test)
        if not renamer.used:
            continue                                                    # e.g. `assert True, "msg"`
        if isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq):
            tests.append({"input": ast.unparse(test.left), "expected": ast.unparse(test.comparators[0])})
        else:
            tests.append({"input": ast.unparse(test), "expected": "True"})
    if not tests:
        return None, "", "no assert uses candidate"
    return tests, "\n".join(imports), ""


def convert_problem(row: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
    """One HumanEval row (task_id, prompt, canonical_solution, test, entry_point) -> (task dict | None, reason)."""
    entry = row["entry_point"]
    tests, imports, reason = convert_tests(row["test"], entry)
    if tests is None:
        return None, reason
    prelude = (split_prelude(row["prompt"], entry) + "\n" + imports).strip("\n") + "\n"
    tid = row["task_id"].replace("HumanEval/", "humaneval_")
    return {
        "id": tid, "version": 1, "category": "coding", "difficulty": 0.5,
        "prompt": INSTRUCTION + row["prompt"].rstrip() + "\n",
        "starter_code": "",
        "visible_tests": [],                                           # the docstring examples are the only visible ones
        "hidden_tests": tests,
        "reference_solution": row["prompt"] + row["canonical_solution"],
        "metadata": {"source": "openai_humaneval", "entry_point": entry, "prelude": prelude, "split": "humaneval_v1"},
    }, ""


class PreludeVerifier(SubprocessVerifier):
    """SubprocessVerifier that first prepends metadata['prelude'] (imports/helpers the original prompt provided)."""

    def verify(self, task: Dict[str, Any], candidate_code: str) -> Dict[str, Any]:
        prelude = (task.get("metadata") or {}).get("prelude", "")
        return super().verify(task, prelude + "\n" + candidate_code)
