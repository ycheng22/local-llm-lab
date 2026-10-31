"""Generates kaggle/kernels/x03-4b-baseline (plan Step 19d). Run from the repo root: python kaggle/tools/build_x03.py"""
import sys
sys.path.insert(0, "kaggle/tools")
from nbbuild import md, code, setup_cell, write_kernel

cells = [
    md("""
# x03 - Qwen3.5-4B single-shot baseline (Phase 6.5, Step 19d)

**Mode:** GPU (T4 x2 session, GPU 0), background commit, 2 h hard cap. Estimated ~1.2 GPU hours (4B decodes about half as fast as 2B).

## Question
GRPO took the 2B model from 40.6% to 58.5% Pass@1 on held-out MBPP. **Is that more or less than simply using a model with twice the parameters?** AGENTS.md names Qwen3.5-4B as the secondary model, so this is also the first 4B number in the project.

## Setup (identical to NB 03 / NB 07 except the model)
- Qwen/Qwen3.5-4B in fp16 (about 8.4 GB of weights, fits one 16 GB T4), no adapters.
- `mbpp_v2_eval.jsonl` (100 tasks), n=8 samples, temperature 0.7, top-p 0.9, 384 new tokens, seed 1234. Plus the first 100 synthetic test tasks for the OOD / regression Pass@1.
- Comparison with the 2B base (NB 03) and 2B GRPO (NB 07) per-task counts: paired bootstrap 95% CI.

Inputs via `kernel_sources`: `03-baseline-evaluation`, `07-grpo-evaluation`.
"""),
    setup_cell(install_wheels=False, single_gpu=True),
    code("""
import json, gc, torch, psutil, pandas as pd
from llm_lab_utils import SubprocessVerifier, ExperimentTracker, load_dataset, paired_bootstrap_diff, pass_at_k
from llm_lab_utils.model_utils import setup_model_and_tokenizer
from llm_lab_utils.evaluator import run_evaluation

MODEL_ID = "Qwen/Qwen3.5-4B"
GEN_CFG = {"max_new_tokens": 384, "temperature": 0.7, "top_p": 0.9, "do_sample": True}   # identical to NB 03 / NB 07
K_VALUES, SEED = [1, 4, 8], 1234

def one(pattern):
    hits = glob.glob(pattern, recursive=True); assert hits, f"not found: {pattern}"; return hits[0]
def counts_from(pattern):
    c = {}
    for line in open(one(pattern)):
        g = json.loads(line); c[g["task_id"]] = c.get(g["task_id"], 0) + int(g["verification"]["passed"])
    assert len(c) == 100; return c

REF = {"2B base (NB 03)": counts_from(f"{find_input('03-baseline-evaluation')}/**/exp00_baseline_mbpp_v2/generations.jsonl"),
       "2B GRPO (NB 07)": counts_from(f"{find_input('07-grpo-evaluation')}/**/exp02_grpo_from_base_mbpp_eval/generations.jsonl")}
print({k: round(sum(v.values()) / 800, 3) for k, v in REF.items()}, "<- Pass@1 of the 2B references")
verifier = SubprocessVerifier(timeout=5.0, max_memory_mb=512)
eval_tasks = load_dataset(f"{DATA}/mbpp_v2_eval.jsonl")
synth = load_dataset(f"{DATA}/test.jsonl")[:100]
"""),
    md("## 1. Load the 4B model and evaluate"),
    code("""
model, tok = setup_model_and_tokenizer(MODEL_ID, dtype="float16", device_map={"": 0})
model.eval()
print("VRAM GB:", round(torch.cuda.memory_allocated(0) / 1e9, 2))

RUNS, COUNTS = {}, {}
for suffix, tasks, ks, ds in [("mbpp", eval_tasks, K_VALUES, "mbpp_v2_eval.jsonl"), ("synth", synth, [1, 4], "test.jsonl")]:
    exp_id = f"x03_4b_baseline_{suffix}"
    t0 = time.time(); torch.cuda.reset_peak_memory_stats(0)
    tracker = ExperimentTracker(exp_id, "/kaggle/working")
    tracker.set_config(model=MODEL_ID, dataset=ds, hyperparameters={**GEN_CFG, "k_values": ks, "seed": SEED, "dtype": "float16", "n_tasks": len(tasks)})
    metrics, task_res = run_evaluation(model, tok, tasks, verifier, GEN_CFG, ks, tracker=tracker, seed=SEED)
    wall_h = (time.time() - t0) / 3600
    tracker.data["training"].update(wall_hours=round(wall_h, 4), gpu_hours=round(wall_h, 4)); tracker.save()
    RUNS[suffix] = {**metrics, "peak_vram_gb": tracker.data["hardware"]["peak_vram_gb"], "wall_hours": wall_h}
    if suffix == "mbpp": COUNTS["4B base"] = {r["task_id"]: r["c"] for r in task_res}
    print(exp_id, {k: round(v, 4) for k, v in RUNS[suffix].items()})
"""),
    md("## 2. Is 2B + RL better than 2x the parameters?"),
    code("""
ALL = {**REF, **COUNTS}
rows = []
for new, old in [("4B base", "2B base (NB 03)"), ("4B base", "2B GRPO (NB 07)")]:
    for k in K_VALUES:
        r = paired_bootstrap_diff(ALL[old], ALL[new], n=8, k=k)
        rows.append({"comparison": f"{new} vs {old}", "metric": f"Pass@{k}", "old": round(r["mean_a"], 3), "new": round(r["mean_b"], 3),
                     "diff": round(r["diff"], 3), "95% CI": f"[{r['ci95_low']:+.3f}, {r['ci95_high']:+.3f}]",
                     "significant": r["ci95_low"] > 0 or r["ci95_high"] < 0})
cmp_df = pd.DataFrame(rows); print(cmp_df.to_string(index=False))
def agg(counts, k): return sum(pass_at_k(8, c, k) for c in counts.values()) / len(counts)
print(pd.DataFrame({m: {f"Pass@{k}": round(agg(ALL[m], k), 3) for k in K_VALUES} for m in ALL}).T.to_string())
m = RUNS["mbpp"]
print("\\n4B behaviour:", {x: round(m[x], 3) for x in ["truncated_rate", "syntax_error_rate", "timeout_rate", "gen_tokens_per_s"]},
      "| tokens/sample", round(m["tokens_generated"] / 800, 1), "| synthetic Pass@1", round(RUNS["synth"]["pass_at_1"], 4), "(2B base 0.9962)")
"""),
    md("## 3. Save"),
    code("""
tracker = ExperimentTracker("x03_4b_baseline_summary", "/kaggle/working")
tracker.set_config(model=MODEL_ID, dataset="mbpp_v2_eval.jsonl + test.jsonl", hyperparameters={**GEN_CFG, "k_values": K_VALUES, "seed": SEED, "dtype": "float16"})
wall_h = (time.time() - T0) / 3600
tracker.update_metrics({"runs": RUNS, "comparisons": rows, "ood_pass_at_1": RUNS["synth"]["pass_at_1"]})
tracker.update_hardware(max(r["peak_vram_gb"] for r in RUNS.values()), psutil.virtual_memory().used / 1e9)
tracker.data["training"].update(wall_hours=round(wall_h, 4), gpu_hours=round(wall_h, 4)); tracker.save()
print(f"saved. wall = GPU hours = {wall_h:.2f}")
"""),
    md("""
## Takeaways (fill in)
- 4B base vs 2B base: how much does doubling parameters give (CI)?
- 4B base vs 2B GRPO: if the CI includes zero or favours GRPO, RL on 256 tasks matched or beat 2x parameters on MBPP (remember x02: the RL gain did not transfer to HumanEval, the 4B gain probably will).
"""),
]

write_kernel("kaggle/kernels/x03-4b-baseline", "x03_4b_baseline.ipynb", "ycheng22/x03-4b-baseline", "x03 4B Baseline", cells,
             gpu=True, internet=True, kernel_sources=["ycheng22/03-baseline-evaluation", "ycheng22/07-grpo-evaluation"])
