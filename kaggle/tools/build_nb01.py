import json, os


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.strip("\n").splitlines(True)}


cells = [
    md("""
# 01 - Environment, Inference & dtype Benchmark (Phase 0)

**Mode:** GPU (T4 x2), run as a background **commit** (stops the moment the last cell ends). Budget ~0.5 h.

## Learning goals
1. Know exactly what hardware Kaggle gave us (GPU model, VRAM, compute capability, RAM).
2. Load Qwen3.5-2B and generate code, recording VRAM and latency.
3. **Decide the training dtype for the whole project**: a T4 has no native bf16, so we measure bf16 vs fp16 vs 4-bit NF4 (speed, VRAM, and whether output quality survives).
4. Confirm the `SubprocessVerifier` resource limits actually work on Linux.

Every config is wrapped in try/except, so one failure never wastes the GPU session; results are written to `/kaggle/working/exp00_env_benchmark/summary.json`.
"""),
    md("## 1. Setup: offline wheels + shared library"),
    code("""
import glob, os, subprocess, sys, time
T0 = time.time()

def find_input(name):
    hits = [p for p in glob.glob(f"/kaggle/input/**/{name}", recursive=True) if os.path.isdir(p)]
    assert hits, f"dataset '{name}' not attached; found: {os.listdir('/kaggle/input')}"
    return hits[0]

WHEELS, UTILS, DATA = find_input("llm-lab-wheels"), find_input("llm-lab-utils"), find_input("llm-lab-data")
print(WHEELS, UTILS, DATA, sep="\\n")

wheels = sorted(glob.glob(f"{WHEELS}/*.whl"))
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", *wheels],
                   capture_output=True, text=True)
print("offline pip install rc:", r.returncode, r.stderr[-500:])

# Locate the llm_lab_utils package whether Kaggle extracted it as a folder or left it as a .zip
print("--- tree of", UTILS)
for root, dirs, files in os.walk(UTILS):
    print(root, dirs[:5], files[:8])
# Kaggle mounts this dataset with the package files at the dataset ROOT (the llm_lab_utils/ folder is flattened),
# so expose the mount under the package name via a symlink; relative imports keep working.
if os.path.exists(f"{UTILS}/__init__.py"):
    os.makedirs("/tmp/_pkg", exist_ok=True)
    link = "/tmp/_pkg/llm_lab_utils"
    if not os.path.exists(link):
        os.symlink(UTILS, link)
    sys.path.insert(0, "/tmp/_pkg")
else:
    found = glob.glob(f"{UTILS}/**/llm_lab_utils/__init__.py", recursive=True)
    assert found, "llm_lab_utils package not found in dataset"
    sys.path.insert(0, os.path.dirname(os.path.dirname(found[0])))
import llm_lab_utils
print("llm_lab_utils", llm_lab_utils.__version__, "from", llm_lab_utils.__file__)
"""),
    md("## 2. Hardware diagnosis"),
    code("""
import torch, psutil
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
hw = {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu_count": torch.cuda.device_count(),
      "ram_total_gb": round(psutil.virtual_memory().total / 1e9, 1)}
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    hw[f"gpu{i}"] = {"name": p.name, "vram_gb": round(p.total_memory / 1e9, 1), "capability": f"{p.major}.{p.minor}"}
hw["bf16_supported_flag"] = torch.cuda.is_bf16_supported()
print(hw)
"""),
    md("""
## 3. Verifier sanity on Linux
`SubprocessVerifier` relies on `resource.setrlimit` (Linux only, untested on our Windows dev box). The memory bomb and the infinite loop must both be stopped quickly; the good solution must pass.
"""),
    code("""
from llm_lab_utils import SubprocessVerifier
v = SubprocessVerifier(timeout=3.0, max_memory_mb=512)
task = {"id": "t", "version": 1, "category": "x", "difficulty": 0.1, "prompt": "",
        "visible_tests": [{"input": "f(2)", "expected": "4"}], "hidden_tests": [{"input": "f(3)", "expected": "6"}]}
cases = {"good": "def f(x): return 2*x",
         "memory_bomb": "def f(x):\\n    return len([0]*(10**10))",
         "infinite_loop": "def f(x):\\n    while True: pass",
         "syntax_error": "def f(x) return x"}
verifier_check = {}
for name, src in cases.items():
    res = v.verify(task, src)
    verifier_check[name] = {k: res[k] for k in ("passed", "timeout", "runtime_error", "syntax_error", "duration_seconds")}
    print(name, verifier_check[name])
assert verifier_check["good"]["passed"] and not verifier_check["memory_bomb"]["passed"] and verifier_check["infinite_loop"]["timeout"]
"""),
    md("""
## 4. dtype benchmark
Same 8 dev tasks, **greedy decoding** (deterministic), single GPU, for each of: fp16, bf16, 4-bit NF4 (fp16 compute). We record load time, VRAM, tokens/s and the pass rate of the generated code.
"""),
    code("""
import gc, json
from llm_lab_utils import PRIMARY_MODEL_ID, load_dataset
from llm_lab_utils.model_utils import setup_model_and_tokenizer
from llm_lab_utils.evaluator import build_messages, extract_code

tasks = load_dataset(f"{DATA}/dev.jsonl")[:8]
MAX_NEW = 384

def bench(label, dtype, nf4=False):
    out = {"label": label, "dtype": dtype, "nf4": nf4}
    try:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(0)
        t = time.time()
        model, tok = setup_model_and_tokenizer(PRIMARY_MODEL_ID, load_in_4bit=nf4, dtype=dtype, device_map={"": 0})
        out["load_s"] = round(time.time() - t, 1)
        out["vram_after_load_gb"] = round(torch.cuda.memory_allocated(0) / 1e9, 2)

        gen_s = n_tok = passed = empty = 0
        sample = ""
        for task in tasks:
            td = task.model_dump()
            text = tok.apply_chat_template(build_messages(td), tokenize=False, add_generation_prompt=True)
            inp = tok(text, return_tensors="pt").to(model.device)
            torch.cuda.synchronize(); t = time.time()
            with torch.no_grad():
                o = model.generate(**inp, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tok.eos_token_id)
            torch.cuda.synchronize(); gen_s += time.time() - t
            new = o[0][inp.input_ids.shape[1]:]
            n_tok += len(new)
            comp = tok.decode(new, skip_special_tokens=True)
            empty += (len(comp.strip()) == 0)
            passed += bool(v.verify(td, extract_code(comp))["passed"])
            sample = sample or comp[:400]
        out.update(peak_vram_gb=round(torch.cuda.max_memory_allocated(0) / 1e9, 2),
                   tokens_per_s=round(n_tok / gen_s, 1), pass_rate=passed / len(tasks),
                   empty_outputs=empty, new_tokens=n_tok, gen_seconds=round(gen_s, 1), sample=sample)
        del model, tok
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"[:500]
    gc.collect(); torch.cuda.empty_cache()
    print({k: x for k, x in out.items() if k != "sample"})
    return out

results = [bench("fp16", "float16"), bench("bf16", "bfloat16"), bench("nf4_fp16compute", "float16", nf4=True)]
"""),
    md("## 5. Compare and decide"),
    code("""
import pandas as pd
df = pd.DataFrame([{k: x for k, x in r.items() if k != "sample"} for r in results])
print(df.to_string())
ok = [r for r in results if "error" not in r]
best = max(ok, key=lambda r: (r["pass_rate"], r["tokens_per_s"])) if ok else None
print("RECOMMENDED:", best["label"] if best else "none succeeded")
for r in ok:
    print("\\n---", r["label"], "sample:\\n", r["sample"])
"""),
    md("## 6. Save results (picked up as the version's Output)"),
    code("""
from llm_lab_utils import ExperimentTracker
tracker = ExperimentTracker("exp00_env_benchmark", "/kaggle/working")
tracker.set_config(model=PRIMARY_MODEL_ID, dataset="dev.jsonl[:8]",
                   hyperparameters={"max_new_tokens": MAX_NEW, "decoding": "greedy", "device_map": "single GPU 0"})
tracker.update_metrics({"benchmark": results, "recommended": best["label"] if best else None,
                        "hardware": hw, "verifier_check": verifier_check})
tracker.update_hardware(max([r.get("peak_vram_gb", 0) for r in results]), psutil.Process().memory_info().rss / 1e9)
wall_h = (time.time() - T0) / 3600
tracker.data["training"].update(wall_hours=round(wall_h, 4), gpu_hours=round(wall_h, 4))
tracker.save()
print(open("/kaggle/working/exp00_env_benchmark/summary.json").read()[:1500])
print(f"wall time: {wall_h * 60:.1f} min")
"""),
    md("""
## Takeaways (fill in after reading the table)
- Which dtype is fastest on T4, and did bf16 emulation cost speed or quality?
- How much VRAM headroom does the 2B model leave for LoRA + GRPO rollouts?
- Next: `02_dataset_and_verification` (CPU), then `03_baseline_evaluation`.
"""),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
d = "kaggle/kernels/01-env-inference-benchmark/"
os.makedirs(d, exist_ok=True)
json.dump(nb, open(d + "01_environment_and_inference.ipynb", "w"), indent=1)
json.dump({"id": "ycheng22/01-env-inference-benchmark", "title": "01 Env Inference Benchmark",
           "code_file": "01_environment_and_inference.ipynb", "language": "python", "kernel_type": "notebook",
           "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
           "dataset_sources": ["ycheng22/llm-lab-utils", "ycheng22/llm-lab-data", "ycheng22/llm-lab-wheels"],
           "competition_sources": [], "kernel_sources": [], "model_sources": []},
          open(d + "kernel-metadata.json", "w"), indent=2)

import ast
for c in cells:
    if c["cell_type"] == "code":
        ast.parse("".join(c["source"]))
print("written; all code cells parse")
