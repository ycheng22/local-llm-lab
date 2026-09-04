#!/usr/bin/env python3
import sys
import platform
from llm_lab.hardware import get_system_ram_gb, get_gpu_info, check_docker_available

def check_package(package_name: str) -> str:
    try:
        module = __import__(package_name)
        return getattr(module, '__version__', 'Installed (no __version__)')
    except ImportError:
        return 'Not Installed'

def main():
    print("=== Local LLM Lab Environment ===\n")
    
    py_version = platform.python_version()
    print(f"Python              {py_version:<13} {'OK' if py_version.startswith('3.1') else 'WARN'}")
    
    torch_ver = check_package('torch')
    print(f"PyTorch             {torch_ver:<13} {'OK' if torch_ver != 'Not Installed' else 'FAIL'}")
    
    gpu_info = get_gpu_info()
    print(f"CUDA available      {str(gpu_info['available']):<13} {'OK' if gpu_info['available'] else 'FAIL'}")
    
    if gpu_info['available']:
        print(f"CUDA Version        {gpu_info['cuda_version']}")
        print(f"GPU                 {gpu_info['name']}")
        print(f"VRAM                {gpu_info['vram_gb']:.1f} GB")
    
    sys_ram = get_system_ram_gb()
    print(f"System RAM          {sys_ram:.1f} GB")
    print()
    
    for pkg in ['transformers', 'trl', 'peft', 'bitsandbytes']:
        ver = check_package(pkg)
        print(f"{pkg:<19} {ver:<13} {'OK' if ver != 'Not Installed' else 'FAIL'}")
        
    print()
    docker_ok = check_docker_available()
    print(f"Docker              {'available' if docker_ok else 'unavailable':<13} {'OK' if docker_ok else 'FAIL'}")
    print()
    
    from llm_lab.constants import HF_HOME, PRIMARY_MODEL_ID
    print(f"HF_HOME             {str(HF_HOME):<35}")
    model_cached = (HF_HOME / "hub" / f"models--{PRIMARY_MODEL_ID.replace('/', '--')}").exists()
    print(f"Primary Model Cache ({PRIMARY_MODEL_ID}) {'CACHED' if model_cached else 'NOT CACHED':<10} {'OK' if model_cached else 'WARN'}")
    print()
    
    # Assess overall status
    is_ready = True
    if torch_ver == 'Not Installed' or not gpu_info['available']:
        is_ready = False
    for pkg in ['transformers', 'trl', 'peft', 'bitsandbytes']:
        if check_package(pkg) == 'Not Installed':
            is_ready = False
    
    if is_ready:
        print("STATUS: READY")
    else:
        print("STATUS: MISSING DEPENDENCIES. See README/Plan for installation instructions.")

if __name__ == "__main__":
    main()
