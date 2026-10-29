"""Generates kaggle/kernels/03-baseline-evaluation. Run from the repo root: python kaggle/tools/build_nb03.py"""
import sys
sys.path.insert(0, "kaggle/tools")
from nbbuild import md, code, setup_cell, write_kernel

cells = [
    md("""
# 03 - Baseline Evaluation (Phase 1)

**Mode:** GPU (T4 x2 session, uses GPU 0), background commit, 1.5 h hard cap. Estimated ~1 GPU hour.

## Learning goals
1. Run the full evaluation loop: prompt -> sample n completions -> extract code -> verify in the sandbox -> Pass@K.
2. Report the metrics every experiment must report (AGENTS.md rule 10): Pass@1/4/8, OOD Pass@1, peak VRAM, peak RAM, wall time, GPU hours.
3. **Test a hypothesis from the data audit (NB 02):** the original MBPP prompts do not name the function the tests call. We evaluate the *same 100 tasks* with the v1 prompt and the v2 prompt (which adds one example test) and compare.

## Three evaluations (same base model Qwen3.5-2B, fp16, same sampling settings)
| Experiment | Data | Role |
|---|---|---|
| `exp00_baseline_synth` | `test.jsonl` (100 frozen synthetic tasks) | in-distribution |
| `exp00_baseline_mbpp_v1` | first 100 MBPP-v2 task ids, ORIGINAL prompt | OOD, old prompt |
| `exp00_baseline_mbpp_v2` | same 100 tasks, prompt + example test | OOD, fixed prompt |

**OOD Pass@1** in this project = Pass@1 on MBPP (v2 prompt is the headline; v1 shown for contrast).
"""),
    setup_cell(install_wheels=False),
    code("""
import gc, json, torch, psutil, pandas as pd
from llm_lab_utils import PRIMARY_MODEL_ID, SubprocessVerifier, ExperimentTracker, load_dataset
from llm_lab_utils.model_utils import setup_model_and_tokenizer
from llm_lab_utils.evaluator import run_evaluation

GEN_CFG = {"max_new_tokens": 384, "temperature": 0.7, "top_p": 0.9, "do_sample": True}   # identical to local configs/baseline.yaml
K_VALUES = [1, 4, 8]
SEED, N_TASKS = 1234, 100

model, tokenizer = setup_model_and_tokenizer(PRIMARY_MODEL_ID, dtype="float16", device_map={"": 0})
model.eval()
print("loaded; VRAM GB:", round(torch.cuda.memory_allocated(0) / 1e9, 2))
"""),
    md("""
## Datasets
The v1 and v2 MBPP sets contain the same tasks and tests; only the prompt differs. We use identical task ids for both.
"""),
    code("""
synth = load_dataset(f"{DATA}/test.jsonl")[:N_TASKS]
mbpp_v2 = load_dataset(f"{DATA}/train_complex_v2.jsonl")[:N_TASKS]
ids = {t.id for t in mbpp_v2}
mbpp_v1 = [t for t in load_dataset(f"{DATA}/train_complex_500.jsonl") if t.id in ids]
assert len(mbpp_v1) == len(mbpp_v2) == N_TASKS
print(len(synth), len(mbpp_v1), len(mbpp_v2))
print("\\nv1 prompt:\\n", mbpp_v1[0].prompt, "\\n\\nv2 prompt:\\n", mbpp_v2[0].prompt)
"""),
    md("## Evaluate"),
    code("""
verifier = SubprocessVerifier(timeout=5.0, max_memory_mb=512)
RUNS = {}

def evaluate(exp_id, tasks, dataset_name):
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats(0)
    tracker = ExperimentTracker(exp_id, "/kaggle/working")
    tracker.set_config(model=PRIMARY_MODEL_ID, dataset=dataset_name,
                       hyperparameters={**GEN_CFG, "k_values": K_VALUES, "seed": SEED, "dtype": "float16",
                                        "n_tasks": len(tasks), "verifier": "SubprocessVerifier(5s,512MB)"})
    metrics, _ = run_evaluation(model, tokenizer, tasks, verifier, GEN_CFG, K_VALUES, tracker=tracker, seed=SEED)
    wall_h = (time.time() - t0) / 3600
    tracker.data["training"].update(wall_hours=round(wall_h, 4), gpu_hours=round(wall_h, 4))
    tracker.save()
    RUNS[exp_id] = {**metrics, "peak_vram_gb": tracker.data["hardware"]["peak_vram_gb"],
                    "peak_ram_gb": tracker.data["hardware"]["peak_ram_gb"], "wall_hours": wall_h}
    print(exp_id, {k: round(v, 4) for k, v in RUNS[exp_id].items()})

evaluate("exp00_baseline_synth", synth, "test.jsonl")
evaluate("exp00_baseline_mbpp_v2", mbpp_v2, "train_complex_v2.jsonl[:100]")
evaluate("exp00_baseline_mbpp_v1", mbpp_v1, "train_complex_500.jsonl (same ids as v2[:100])")
"""),
    md("## Results table"),
    code("""
df = pd.DataFrame(RUNS).T[["pass_at_1", "pass_at_4", "pass_at_8", "timeout_rate", "syntax_error_rate",
                           "mean_latency", "peak_vram_gb", "peak_ram_gb", "wall_hours"]]
print(df.round(4).to_string())
ood = RUNS["exp00_baseline_mbpp_v2"]["pass_at_1"]
print(f"\\nIn-distribution Pass@1 (synthetic): {RUNS['exp00_baseline_synth']['pass_at_1']:.3f}")
print(f"OOD Pass@1 (MBPP, v2 prompt):        {ood:.3f}")
print(f"OOD Pass@1 (MBPP, v1 prompt):        {RUNS['exp00_baseline_mbpp_v1']['pass_at_1']:.3f}")
"""),
    md("""
## Why do they differ? Failure analysis
If the v1 prompt hides the required function name, failures should be dominated by "name ... is not defined" (the test calls a function the model never defined). We count error types over all failed samples.
"""),
    code("""
import re, collections
def failure_profile(exp_id):
    c = collections.Counter(); total = 0
    for line in open(f"/kaggle/working/{exp_id}/generations.jsonl"):
        v = json.loads(line)["verification"]
        total += 1
        if v["passed"]: c["passed"] += 1
        elif v["timeout"]: c["timeout"] += 1
        elif v["syntax_error"]: c["syntax_error"] += 1
        elif "is not defined" in v["stderr"]: c["name not defined"] += 1
        elif "failed: expected" in v["stderr"]: c["wrong answer"] += 1
        else: c["other runtime error"] += 1
    return {k: round(n / total, 3) for k, n in c.most_common()}

profiles = {e: failure_profile(e) for e in ["exp00_baseline_mbpp_v1", "exp00_baseline_mbpp_v2"]}
prof_df = pd.DataFrame(profiles).fillna(0.0)
print(prof_df.to_string())
"""),
    md("## Save combined summary"),
    code("""
tracker = ExperimentTracker("exp00_baseline", "/kaggle/working")
tracker.set_config(model=PRIMARY_MODEL_ID, dataset="synthetic test + MBPP v1/v2 (100 tasks each)",
                   hyperparameters={**GEN_CFG, "k_values": K_VALUES, "seed": SEED, "dtype": "float16"})
wall_h = (time.time() - T0) / 3600
tracker.update_metrics({"runs": RUNS, "failure_profiles": profiles,
                        "in_dist_pass_at_1": RUNS["exp00_baseline_synth"]["pass_at_1"],
                        "ood_pass_at_1": ood, "ood_pass_at_1_v1_prompt": RUNS["exp00_baseline_mbpp_v1"]["pass_at_1"]})
tracker.update_hardware(max(r["peak_vram_gb"] for r in RUNS.values()), max(r["peak_ram_gb"] for r in RUNS.values()))
tracker.data["training"].update(wall_hours=round(wall_h, 4), gpu_hours=round(wall_h, 4))
tracker.save()
print(f"saved. total wall = GPU hours = {wall_h:.2f}")
"""),
    md("""
## Takeaways (fill in from the table)
- Did the example test in the prompt change MBPP Pass@1, and did NameError failures drop?
- If yes: the earlier "0% / impossible" results measured prompt format, not capability.
- Pass@4/8 vs Pass@1 shows how much latent ability sampling uncovers.
- Next: `04_sft_training`.
"""),
]

write_kernel("kaggle/kernels/03-baseline-evaluation", "03_baseline_evaluation.ipynb",
             "ycheng22/03-baseline-evaluation", "03 Baseline Evaluation", cells, gpu=True, internet=True)
