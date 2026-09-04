import pytest
from llm_lab.hardware import get_system_ram_gb

def test_system_ram():
    ram_gb = get_system_ram_gb()
    assert ram_gb > 0, "System RAM should be greater than 0"

def test_imports():
    """Ensure basic imports work, catching missing dependencies early."""
    try:
        import transformers
        import trl
        import peft
        import bitsandbytes
    except ImportError as e:
        pytest.fail(f"Failed to import essential ML packages: {e}")
