"""Generates kaggle/kernels/x04-length-cap (plan Step 19e). Run from the repo root: python kaggle/tools/build_x04.py"""
import sys
sys.path.insert(0, "kaggle/tools")
from nbbuild import md, code, setup_cell, write_kernel

cells = [
    md("""
# x04 - Does the 384-token cap hide capability? (Phase 6.5, Step 19e)

**Mode:** GPU (T4 x2 session, GPU 0), background commit, 2 h hard cap. Estimated ~1.3 GPU hours (2B base, 2B GRPO, 4B base; 100 tasks x 4 samples each; 4B last).

## Why
All earlier MBPP numbers use `max_new_tokens=384`. Two findings make that cap suspect:
- 2B models: 8.4% of samples are truncated and nearly all of them fail (truncation analysis, plan row 21).
- **Qwen3.5-4B (x03): 90.4% of samples hit the cap** because it writes a long reasoning preamble before the code. Its 53.8% Pass@1 is therefore a lower bound, and "2B + GRPO vs 4B" is not a fair comparison yet.

## Design (only the cap changes)
- Same 100 `mbpp_v2_eval` tasks, temperature 0.7, top-p 0.9, seed 1234, fp16. **Only `max_new_tokens` changes: 384 -> 768.** No length penalty, no new reward.
- n=4 samples per task (half of the 384 runs) to keep cost near 1.3 GPU h. Pass@1 is unbiased for any n, so the 384 (n=8) and 768 (n=4) Pass@1 values are comparable; Pass@4 is reported for 768 only.
- Arms: 2B base, 2B GRPO-from-base (adapter merged), 4B base. 4B runs last so a time-out costs only the least essential arm.
- Comparison: paired bootstrap over tasks of per-task Pass@1 (768 minus 384). The 384 per-task counts come from the saved NB 03 / NB 07 / x03 generations.

## Pre-registered reading
- If 768 raises 2B Pass@1 by more than the CI, the 384 results understate every model and the README must say so.
- The question for 4B: does it overtake 2B GRPO once it can finish its reasoning?

Inputs via `kernel_sources`: `03-baseline-evaluation`, `06-grpo-training`, `07-grpo-evaluation`, `x03-4b-baseline`.
"""),
    setup_cell(install_wheels=False, single_gpu=True),
    code("""
import json, gc, random, torch, psutil, pandas as pd
from llm_lab_utils import PRIMARY_MODEL_ID, SubprocessVerifier, ExperimentTracker, load_dataset
from llm_lab_utils.model_utils import setup_model_and_tokenizer
from llm_lab_utils.evaluator import run_evaluation

def one(pattern):
    hits = glob.glob(pattern, recursive=True); assert hits, f"not found: {pattern}"; return hits[0]
def counts_from(pattern):
    c = {}
    for line in open(one(pattern)):
        g = json.loads(line); c[g["task_id"]] = c.get(g["task_id"], 0) + int(g["verification"]["passed"])
    assert len(c) == 100; return c

GEN_CFG = {"max_new_tokens": 768, "temperature": 0.7, "top_p": 0.9, "do_sample": True}   # only the cap differs from NB 03/07/x03
K_VALUES, SEED = [1, 4], 1234          # run_evaluation draws max(K_VALUES) = 4 samples per task
N_SAMPLES = max(K_VALUES)
REF384 = {   # per-task passes out of 8 at the 384 cap
    "base": counts_from(f"{find_input('03-baseline-evaluation')}/**/exp00_baseline_mbpp_v2/generations.jsonl"),
    "grpo_from_base": counts_from(f"{find_input('07-grpo-evaluation')}/**/exp02_grpo_from_base_mbpp_eval/generations.jsonl"),
    "4b_base": counts_from(f"{find_input('x03-4b-baseline')}/**/x03_4b_baseline_mbpp/generations.jsonl"),
}
print({k: round(sum(v.values()) / 800, 3) for k, v in REF384.items()}, "<- Pass@1 at 384 tokens")
verifier = SubprocessVerifier(timeout=5.0, max_memory_mb=512)
tasks = load_dataset(f"{DATA}/mbpp_v2_eval.jsonl")
G_BASE = one(f"{find_input('06-grpo-training')}/**/grpo_from_base/checkpoint-final")
"""),
    md("## 1. Evaluate at 768 tokens (4B last)"),
    code("""
ARMS = [("base", PRIMARY_MODEL_ID, None), ("grpo_from_base", PRIMARY_MODEL_ID, [G_BASE]), ("4b_base", "Qwen/Qwen3.5-4B", None)]
RUNS, COUNTS = {}, {}
for name, model_id, adapters in ARMS:
    model, tok = setup_model_and_tokenizer(model_id, dtype="float16", device_map={"": 0}, adapter_path=adapters, merge=bool(adapters))
    model.eval()
    exp_id = f"x04_cap768_{name}"
    t0 = time.time(); torch.cuda.reset_peak_memory_stats(0)
    tracker = ExperimentTracker(exp_id, "/kaggle/working")
    tracker.set_config(model=f"{model_id} ({name})", dataset="mbpp_v2_eval.jsonl",
                       hyperparameters={**GEN_CFG, "k_values": K_VALUES, "seed": SEED, "dtype": "float16", "merged": bool(adapters),
                                        "parent_checkpoint": G_BASE if adapters else None, "n_tasks": len(tasks)})
    metrics, task_res = run_evaluation(model, tok, tasks, verifier, GEN_CFG, K_VALUES, tracker=tracker, seed=SEED)
    wall_h = (time.time() - t0) / 3600
    tracker.data["training"].update(wall_hours=round(wall_h, 4), gpu_hours=round(wall_h, 4)); tracker.save()
    RUNS[name] = {**metrics, "peak_vram_gb": tracker.data["hardware"]["peak_vram_gb"], "wall_hours": wall_h}
    COUNTS[name] = {r["task_id"]: r["c"] for r in task_res}
    print(exp_id, {k: round(v, 4) for k, v in RUNS[name].items()})
    del model, tok; gc.collect(); torch.cuda.empty_cache()
"""),
    md("## 2. 768 vs 384: paired bootstrap over tasks of per-task Pass@1"),
    code("""
def boot(a, b, n_boot=10000, seed=0):
    rng = random.Random(seed); d = [y - x for x, y in zip(a, b)]
    m = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(n_boot))
    return sum(d) / len(d), m[int(0.025 * n_boot)], m[int(0.975 * n_boot)]
def p1(counts, n, ids): return [counts[i] / n for i in ids]
ids = sorted(REF384["base"])
rows = []
for name in COUNTS:
    a, b = p1(REF384[name], 8, ids), p1(COUNTS[name], N_SAMPLES, ids)
    d, lo, hi = boot(a, b)
    rows.append({"model": name, "pass@1 @384": round(sum(a) / len(a), 3), "pass@1 @768": round(sum(b) / len(b), 3), "diff": round(d, 3),
                 "95% CI": f"[{lo:+.3f}, {hi:+.3f}]", "significant": lo > 0 or hi < 0,
                 "truncated @768": round(RUNS[name]["truncated_rate"], 3), "syntax_err @768": round(RUNS[name]["syntax_error_rate"], 3),
                 "pass@4 @768": round(RUNS[name]["pass_at_4"], 3)})
print(pd.DataFrame(rows).to_string(index=False))

print("\\nDoes 4B overtake 2B GRPO once the cap is 768?  (per-task Pass@1, paired)")
cmp_rows = []
for a_name, b_name in [("grpo_from_base", "4b_base"), ("base", "4b_base"), ("base", "grpo_from_base")]:
    if b_name in COUNTS and a_name in COUNTS:
        d, lo, hi = boot(p1(COUNTS[a_name], N_SAMPLES, ids), p1(COUNTS[b_name], N_SAMPLES, ids))
        cmp_rows.append({"comparison": f"{b_name} - {a_name} @768", "diff": round(d, 3), "95% CI": f"[{lo:+.3f}, {hi:+.3f}]", "significant": lo > 0 or hi < 0})
print(pd.DataFrame(cmp_rows).to_string(index=False))
"""),
    md("## 3. Save"),
    code("""
tracker = ExperimentTracker("x04_cap768_summary", "/kaggle/working")
tracker.set_config(model=f"{PRIMARY_MODEL_ID} + Qwen/Qwen3.5-4B", dataset="mbpp_v2_eval.jsonl", hyperparameters={**GEN_CFG, "k_values": K_VALUES, "seed": SEED})
wall_h = (time.time() - T0) / 3600
tracker.update_metrics({"runs": RUNS, "cap_effect": rows, "model_comparisons": cmp_rows})
tracker.update_hardware(max(r["peak_vram_gb"] for r in RUNS.values()), psutil.virtual_memory().used / 1e9)
tracker.data["training"].update(wall_hours=round(wall_h, 4), gpu_hours=round(wall_h, 4)); tracker.save()
print(f"saved. wall = GPU hours = {wall_h:.2f}")
"""),
    md("""
## Takeaways (fill in)
- Does 768 change 2B Pass@1 beyond the CI? What is the truncated rate now?
- Does 4B Pass@1 rise a lot (cap-limited) and does it overtake 2B GRPO?
"""),
]

write_kernel("kaggle/kernels/x04-length-cap", "x04_length_cap.ipynb", "ycheng22/x04-length-cap", "x04 Length Cap", cells,
             gpu=True, internet=True,
             kernel_sources=["ycheng22/03-baseline-evaluation", "ycheng22/06-grpo-training", "ycheng22/07-grpo-evaluation", "ycheng22/x03-4b-baseline"])
