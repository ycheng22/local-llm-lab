"""
llm_lab_utils: shared library for the Local LLM Lab Kaggle notebooks.

Imports here are kept light (no torch/transformers) so CPU notebooks and unit
tests can `import llm_lab_utils` without a GPU stack. Heavy modules
(model_utils, evaluator.run_evaluation, plots) import their dependencies lazily
or must be imported explicitly.
"""

__version__ = "0.1.0"

PRIMARY_MODEL_ID = "Qwen/Qwen3.5-2B"
SECONDARY_MODEL_ID = "Qwen/Qwen3.5-4B"

from .schema import TaskSchema, TestCase
from .loader import load_dataset, save_dataset
from .subprocess_verifier import SubprocessVerifier, Verifier
from .metrics import pass_at_k, aggregate_pass_at_k, paired_bootstrap_diff
from .experiment_tracker import ExperimentTracker

__all__ = [
    "PRIMARY_MODEL_ID", "SECONDARY_MODEL_ID",
    "TaskSchema", "TestCase", "load_dataset", "save_dataset",
    "SubprocessVerifier", "Verifier",
    "pass_at_k", "aggregate_pass_at_k", "paired_bootstrap_diff", "ExperimentTracker",
]
