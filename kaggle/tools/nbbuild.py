"""Tiny helpers shared by the notebook generators (build_nbXX.py). Run generators from the repo root."""
import ast
import json
import os

DATASETS = ["ycheng22/llm-lab-utils", "ycheng22/llm-lab-data", "ycheng22/llm-lab-wheels"]

# Every notebook starts with this cell. Kaggle mounts datasets under /kaggle/input/datasets/<user>/<slug> and
# flattens llm-lab-utils so __init__.py sits at the mount root; the symlink restores the package name.
SETUP_CELL = '''
import glob, os, subprocess, sys, time
T0 = time.time()

def find_input(name):
    hits = [p for p in glob.glob(f"/kaggle/input/**/{name}", recursive=True) if os.path.isdir(p)]
    assert hits, f"dataset '{name}' not attached; found: {os.listdir('/kaggle/input')}"
    return hits[0]

UTILS, DATA = find_input("llm-lab-utils"), find_input("llm-lab-data")
WHEELS = find_input("llm-lab-wheels") if INSTALL_WHEELS else None

# Kaggle's image ships torchao 0.10, which peft>=0.20 refuses to import. We do not use it.
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], capture_output=True)

if INSTALL_WHEELS:
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps",
                        *sorted(glob.glob(f"{WHEELS}/*.whl"))], capture_output=True, text=True)
    print("offline pip install rc:", r.returncode, r.stderr[-500:])

if os.path.exists(f"{UTILS}/__init__.py"):
    os.makedirs("/tmp/_pkg", exist_ok=True)
    if not os.path.exists("/tmp/_pkg/llm_lab_utils"):
        os.symlink(UTILS, "/tmp/_pkg/llm_lab_utils")
    sys.path.insert(0, "/tmp/_pkg")
else:
    found = glob.glob(f"{UTILS}/**/llm_lab_utils/__init__.py", recursive=True)
    assert found, "llm_lab_utils package not found in dataset"
    sys.path.insert(0, os.path.dirname(os.path.dirname(found[0])))
import llm_lab_utils
print("llm_lab_utils", llm_lab_utils.__version__, "from", llm_lab_utils.__file__)
'''


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.strip("\n").splitlines(True)}


def setup_cell(install_wheels: bool, single_gpu: bool = False):
    """single_gpu=True hides GPU 1 BEFORE torch loads: HF Trainer otherwise wraps the model in DataParallel
    and crashes on a model pinned to cuda:0 (seen in NB 04)."""
    pre = 'import os\nos.environ["CUDA_VISIBLE_DEVICES"] = "0"\n' if single_gpu else ""
    return code(pre + f"INSTALL_WHEELS = {install_wheels}\n" + SETUP_CELL)


def write_kernel(folder, ipynb_name, kernel_id, title, cells, gpu, internet, kernel_sources=None):
    """title must slugify to the id's last part, or Kaggle silently uses the title's slug."""
    for c in cells:
        if c["cell_type"] == "code":
            ast.parse("".join(c["source"]))
    os.makedirs(folder, exist_ok=True)
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    with open(os.path.join(folder, ipynb_name), "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    meta = {"id": kernel_id, "title": title, "code_file": ipynb_name, "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": gpu, "enable_tpu": False,
            "enable_internet": internet, "dataset_sources": DATASETS, "competition_sources": [],
            "kernel_sources": kernel_sources or [], "model_sources": []}
    with open(os.path.join(folder, "kernel-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("written", folder)
