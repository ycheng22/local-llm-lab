"""Generates kaggle/kernels/05-sft-evaluation. Run from the repo root: python kaggle/tools/build_nb05.py"""
import sys
sys.path.insert(0, "kaggle/tools")
from nbbuild import md, code, setup_cell, write_kernel

cells = [
    md("""
# 05 - SFT Evaluation (Phase 2)

**Mode:** GPU (T4 x2 session, GPU 0), background commit, 1.5 h hard cap. Estimated ~0.6 GPU hours.

## Learning goals
1. Evaluate a LoRA adapter produced by an earlier notebook (mounted with `kernel_sources`, no upload needed).
2. Compare against the **same-tasks, same-sampling** baseline from NB 03.
3. Quantify uncertainty: with 100 tasks, a few points of Pass@1 difference can be noise, so we report a **paired bootstrap 95% CI**.
4. Check for regressions on the in-distribution synthetic set.

Required inputs (attached via `kernel_sources`): `04-sft-training` (adapter) and `03-baseline-evaluation` (baseline generations).
"""),
    setup_cell(install_wheels=False),
    code("""
import json, torch, pandas as pd
from peft import PeftModel
from llm_lab_utils import (PRIMARY_MODEL_ID, SubprocessVerifier, ExperimentTracker, load_dataset,
                           paired_bootstrap_diff)
from llm_lab_utils.model_utils import setup_model_and_tokenizer
from llm_lab_utils.evaluator import run_evaluation

SFT_DIR = find_input("04-sft-training")
BASE_DIR = find_input("03-baseline-evaluation")
ADAPTER = glob.glob(f"{SFT_DIR}/**/exp01_sft_mbpp_v1/checkpoint-final", recursive=True)[0]
print("adapter:", ADAPTER); print(json.load(open(f"{ADAPTER}/experiment_meta.json")))

GEN_CFG = {"max_new_tokens": 384, "temperature": 0.7, "top_p": 0.9, "do_sample": True}   # identical to NB 03
K_VALUES, SEED = [1, 4, 8], 1234
"""),
    md("## 1. Load base model + SFT adapter"),
    code("""
model, tokenizer = setup_model_and_tokenizer(PRIMARY_MODEL_ID, dtype="float16", device_map={"": 0}, adapter_path=ADAPTER)
model.eval()
print("VRAM GB:", round(torch.cuda.memory_allocated(0) / 1e9, 2))
verifier = SubprocessVerifier(timeout=5.0, max_memory_mb=512)
eval_tasks = load_dataset(f"{DATA}/mbpp_v2_eval.jsonl")
synth = load_dataset(f"{DATA}/test.jsonl")[:100]
"""),
    md("## 2. Evaluate on the 100 held-out MBPP tasks and the synthetic test set"),
    code("""
RUNS, TASK_RESULTS = {}, {}
def evaluate(exp_id, tasks, dataset_name):
    t0 = time.time(); torch.cuda.reset_peak_memory_stats(0)
    tracker = ExperimentTracker(exp_id, "/kaggle/working")
    tracker.set_config(model=PRIMARY_MODEL_ID + " + exp01_sft_mbpp_v1 adapter", dataset=dataset_name,
                       hyperparameters={**GEN_CFG, "k_values": K_VALUES, "seed": SEED, "dtype": "float16", "n_tasks": len(tasks)})
    metrics, task_res = run_evaluation(model, tokenizer, tasks, verifier, GEN_CFG, K_VALUES, tracker=tracker, seed=SEED)
    wall_h = (time.time() - t0) / 3600
    tracker.data["training"].update(wall_hours=round(wall_h, 4), gpu_hours=round(wall_h, 4)); tracker.save()
    RUNS[exp_id] = {**metrics, "peak_vram_gb": tracker.data["hardware"]["peak_vram_gb"],
                    "peak_ram_gb": tracker.data["hardware"]["peak_ram_gb"], "wall_hours": wall_h}
    TASK_RESULTS[exp_id] = {r["task_id"]: r["c"] for r in task_res}
    print(exp_id, {k: round(v, 4) for k, v in RUNS[exp_id].items()})

evaluate("exp01_sft_mbpp_eval", eval_tasks, "mbpp_v2_eval.jsonl")
evaluate("exp01_sft_synth_eval", synth, "test.jsonl")
"""),
    md("## 3. Compare with the NB 03 baseline (paired, same tasks)"),
    code("""
base_summary = json.load(open(glob.glob(f"{BASE_DIR}/**/exp00_baseline/summary.json", recursive=True)[0]))["metrics"]["runs"]
base_counts = {}
for line in open(glob.glob(f"{BASE_DIR}/**/exp00_baseline_mbpp_v2/generations.jsonl", recursive=True)[0]):
    g = json.loads(line); base_counts[g["task_id"]] = base_counts.get(g["task_id"], 0) + int(g["verification"]["passed"])
assert len(base_counts) == 100

rows = []
for k in K_VALUES:
    ci = paired_bootstrap_diff(base_counts, TASK_RESULTS["exp01_sft_mbpp_eval"], n=8, k=k)
    rows.append({"metric": f"Pass@{k}", "base": round(ci["mean_a"], 4), "sft": round(ci["mean_b"], 4),
                 "diff": round(ci["diff"], 4), "95% CI": f"[{ci['ci95_low']:+.3f}, {ci['ci95_high']:+.3f}]",
                 "P(sft better)": ci["p_better"]})
cmp_df = pd.DataFrame(rows); print(cmp_df.to_string(index=False))
print("\\nbaseline (NB 03) Pass@1/4/8:", [round(base_summary['exp00_baseline_mbpp_v2'][f'pass_at_{k}'], 4) for k in K_VALUES])
print("synthetic Pass@1 base vs sft:", round(base_summary['exp00_baseline_synth']['pass_at_1'], 4), "vs", round(RUNS['exp01_sft_synth_eval']['pass_at_1'], 4))
"""),
    md("## 4. Save"),
    code("""
tracker = ExperimentTracker("exp01_sft_evaluation", "/kaggle/working")
tracker.set_config(model=PRIMARY_MODEL_ID + " + exp01_sft_mbpp_v1 adapter", dataset="mbpp_v2_eval.jsonl + test.jsonl",
                   hyperparameters={**GEN_CFG, "k_values": K_VALUES, "seed": SEED, "dtype": "float16"})
wall_h = (time.time() - T0) / 3600
tracker.update_metrics({"runs": RUNS, "comparison_vs_baseline": rows,
                        "ood_pass_at_1": RUNS["exp01_sft_mbpp_eval"]["pass_at_1"],
                        "in_dist_pass_at_1": RUNS["exp01_sft_synth_eval"]["pass_at_1"]})
tracker.update_hardware(max(r["peak_vram_gb"] for r in RUNS.values()), max(r["peak_ram_gb"] for r in RUNS.values()))
tracker.data["training"].update(wall_hours=round(wall_h, 4), gpu_hours=round(wall_h, 4)); tracker.save()
print(f"saved. wall = GPU hours = {wall_h:.2f}")
"""),
    md("""
## Takeaways
- Does the CI of the Pass@1 difference exclude zero? If not, say "no detectable change", not "SFT helped".
- Did Pass@8 move more or less than Pass@1? (SFT often sharpens Pass@1 but can reduce diversity.)
- Any regression on the synthetic set?
- Next: `06_grpo_training`.
"""),
]

write_kernel("kaggle/kernels/05-sft-evaluation", "05_sft_evaluation.ipynb",
             "ycheng22/05-sft-evaluation", "05 SFT Evaluation", cells, gpu=True, internet=True,
             kernel_sources=["ycheng22/04-sft-training", "ycheng22/03-baseline-evaluation"])
