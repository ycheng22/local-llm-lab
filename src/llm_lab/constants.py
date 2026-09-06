import os
from pathlib import Path

# Hugging Face Cache & Endpoints
DEFAULT_HF_HOME = Path("D:/huggingface_cache")
HF_HOME = Path(os.environ.get("HF_HOME", str(DEFAULT_HF_HOME)))
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")

# Ensure environment variables are set in the runtime environment
os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("HF_ENDPOINT", HF_ENDPOINT)

# Models
PRIMARY_MODEL_ID = "Qwen/Qwen3.5-2B"
SECONDARY_MODEL_ID = "Qwen/Qwen3.5-4B"

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def resolve_path(path: str | Path) -> Path:
    """
    Resolve a path relative to the project root unless it is already absolute.
    """
    p = Path(path)
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()
