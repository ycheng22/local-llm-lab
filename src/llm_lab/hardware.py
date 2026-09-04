import psutil
import subprocess
from typing import Dict, Any, Optional

def get_system_ram_gb() -> float:
    """Returns total system RAM in GB."""
    mem = psutil.virtual_memory()
    return mem.total / (1024 ** 3)

def get_gpu_info() -> Dict[str, Any]:
    """Returns GPU information using PyTorch if available, fallback to pynvml."""
    info = {
        "available": False,
        "name": "N/A",
        "vram_gb": 0.0,
        "cuda_version": "N/A",
    }
    
    try:
        import torch
        info["available"] = torch.cuda.is_available()
        if info["available"]:
            info["name"] = torch.cuda.get_device_name(0)
            info["vram_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            info["cuda_version"] = torch.version.cuda
    except ImportError:
        pass
        
    return info

def check_docker_available() -> bool:
    """Checks if Docker is available and running in the system."""
    try:
        result = subprocess.run(["docker", "info"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError):
        return False
