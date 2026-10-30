"""Generates kaggle/kernels/07-grpo-evaluation. Run from the repo root: python kaggle/tools/build_nb07.py"""
import sys
sys.path.insert(0, "kaggle/tools")
from nbbuild import md, code, setup_cell, write_kernel

cells = [
    md("""
# 07 - GRPO Evaluation (Phase 3)

**Mode:** GPU (T4 x2 session, GPU 0), background commit, 2 h hard cap. Estimated ~1.0 GPU hour (adapters are **merged** into the weights, so decoding runs at base speed - NB 05 showed ~2.4x slowdown when unmerged).

## Learning goals
1. Evaluate both GRPO arms on the same 100 held-out MBPP tasks with the same seed and sampling as the baseline (NB 03) and SFT (NB 05).
2. Answer with **paired bootstrap 95% CIs**, not eyeballed differences: with 100 tasks only effects larger than ~7 points are detectable.
3. Four questions:
   - Did GRPO improve on its starting point? (`grpo_from_base` vs base; `grpo_from_sft` vs SFT)
   - Does SFT first help or hurt RL? (`grpo_from_sft` vs `grpo_from_base`)
   - Does RL turn latent ability into reliable ability? (compare GRPO Pass@1 with the base model's Pass@8 = 69%)
   - What happened to answer length and truncation?

Inputs via `kernel_sources`: `04-sft-training` (SFT adapter), `06-grpo-training` (both GRPO adapters), `03-baseline-evaluation` and `05-sft-evaluation` (earlier per-sample results).
"""),
    setup_cell(install_wheels=False),
    code("""
import json, gc, torch, psutil, pandas as pd
from llm_lab_utils import (PRIMARY_MODEL_ID, SubprocessVerifier, ExperimentTracker, load_dataset, paired_bootstrap_diff)
from llm_lab_utils.model_utils import setup_model_and_tokenizer
from llm_lab_utils.evaluator import run_evaluation

def mount(name): return find_input(name)
def one(pattern):
    hits = glob.glob(pattern, recursive=True); assert hits, f"not found: {pattern}"; return hits[0]

SFT = one(f"{mount('04-sft-training')}/**/exp01_sft_mbpp_v1/checkpoint-final")
G_SFT = one(f"{mount('06-grpo-training')}/**/grpo_from_sft/checkpoint-final")
G_BASE = one(f"{mount('06-grpo-training')}/**/grpo_from_base/checkpoint-final")
for p in (SFT, G_SFT, G_BASE): print(p)
# provenance (rule 5): each GRPO adapter records its parent
print(json.load(open(f"{G_SFT}/experiment_meta.json"))["parent_checkpoint"], "|", json.load(open(f"{G_BASE}/experiment_meta.json"))["parent_checkpoint"])

GEN_CFG = {"max_new_tokens": 384, "temperature": 0.7, "top_p": 0.9, "do_sample": True}   # identical to NB 03 / NB 05
K_VALUES, SEED = [1, 4, 8], 1234
verifier = SubprocessVerifier(timeout=5.0, max_memory_mb=512)
eval_tasks = load_dataset(f"{DATA}/mbpp_v2_eval.jsonl")
synth = load_dataset(f"{DATA}/test.jsonl")[:100]
"""),
    md("## 1. Earlier results to compare against (per-task pass counts)"),
    code("""
def counts_from(pattern):
    c = {}
    for line in open(one(pattern)):
        g = json.loads(line); c[g["task_id"]] = c.get(g["task_id"], 0) + int(g["verification"]["passed"])
    assert len(c) == 100; return c

REF = {"base": counts_from(f"{mount('03-baseline-evaluation')}/**/exp00_baseline_mbpp_v2/generations.jsonl"),
       "sft": counts_from(f"{mount('05-sft-evaluation')}/**/exp01_sft_mbpp_eval/generations.jsonl")}
print({k: round(sum(v.values()) / 800, 4) for k, v in REF.items()}, "<- Pass@1 of base and SFT from the saved samples")
"""),
    md("""
## 2. Evaluate each GRPO arm (adapters merged)
`grpo_from_sft` was trained on top of the merged SFT model, so it is evaluated as base + SFT + GRPO merged in that order. `grpo_from_base` is base + GRPO.
"""),
    code("""
RUNS, COUNTS = {}, {}
def evaluate(name, adapters):
    model, tok = setup_model_and_tokenizer(PRIMARY_MODEL_ID, dtype="float16", device_map={"": 0}, adapter_path=adapters, merge=True)
    model.eval()
    for suffix, tasks, ks, ds in [("mbpp", eval_tasks, K_VALUES, "mbpp_v2_eval.jsonl"), ("synth", synth, [1, 4], "test.jsonl")]:
        exp_id = f"exp02_{name}_{suffix}_eval"
        t0 = time.time(); torch.cuda.reset_peak_memory_stats(0)
        tracker = ExperimentTracker(exp_id, "/kaggle/working")
        tracker.set_config(model=f"{PRIMARY_MODEL_ID} + {adapters}", dataset=ds,
                           hyperparameters={**GEN_CFG, "k_values": ks, "seed": SEED, "dtype": "float16", "merged": True})
        metrics, task_res = run_evaluation(model, tok, tasks, verifier, GEN_CFG, ks, tracker=tracker, seed=SEED)
        wall_h = (time.time() - t0) / 3600
        tracker.data["training"].update(wall_hours=round(wall_h, 4), gpu_hours=round(wall_h, 4)); tracker.save()
        RUNS[exp_id] = {**metrics, "peak_vram_gb": tracker.data["hardware"]["peak_vram_gb"], "wall_hours": wall_h}
        if suffix == "mbpp": COUNTS[name] = {r["task_id"]: r["c"] for r in task_res}
        print(exp_id, {k: round(v, 4) for k, v in RUNS[exp_id].items()})
    del model, tok; gc.collect(); torch.cuda.empty_cache()

evaluate("grpo_from_sft", [SFT, G_SFT])
evaluate("grpo_from_base", [G_BASE])
"""),
    md("## 3. Paired comparisons"),
    code("""
PAIRS = [("grpo_from_base", "base", "GRPO(base) vs base"), ("grpo_from_sft", "sft", "GRPO(SFT) vs SFT"),
         ("grpo_from_sft", "base", "GRPO(SFT) vs base"), ("grpo_from_sft", "grpo_from_base", "GRPO(SFT) vs GRPO(base)")]
ALL = {**REF, **COUNTS}
rows = []
for new, old, label in PAIRS:
    for k in K_VALUES:
        r = paired_bootstrap_diff(ALL[old], ALL[new], n=8, k=k)
        rows.append({"comparison": label, "metric": f"Pass@{k}", "old": round(r["mean_a"], 3), "new": round(r["mean_b"], 3),
                     "diff": round(r["diff"], 3), "95% CI": f"[{r['ci95_low']:+.3f}, {r['ci95_high']:+.3f}]",
                     "significant": r["ci95_low"] > 0 or r["ci95_high"] < 0})
cmp_df = pd.DataFrame(rows); print(cmp_df.to_string(index=False))
"""),
    md("""
## 4. Does RL convert Pass@8 into Pass@1?
If GRPO simply sharpens the distribution, its Pass@1 should move toward the base model's Pass@8.
"""),
    code("""
from llm_lab_utils import pass_at_k
def agg(counts, k): return sum(pass_at_k(8, c, k) for c in counts.values()) / len(counts)
table = pd.DataFrame({m: {f"Pass@{k}": round(agg(ALL[m], k), 3) for k in K_VALUES} for m in ["base", "sft", "grpo_from_base", "grpo_from_sft"]}).T
print(table.to_string())
ex = {m: RUNS[f"exp02_{m}_mbpp_eval"] for m in ["grpo_from_base", "grpo_from_sft"]}
print("\\nanswer-length / stopping behaviour (GRPO arms):")
print(pd.DataFrame({m: {"tokens/sample": round(r["tokens_generated"] / 800, 1), "truncated_rate": round(r["truncated_rate"], 3),
                        "syntax_error_rate": round(r["syntax_error_rate"], 3), "gen_tokens_per_s": round(r["gen_tokens_per_s"], 1)}
                    for m, r in ex.items()}).T.to_string())
print("\\nsynthetic regression Pass@1:", {m: round(RUNS[f"exp02_{m}_synth_eval"]["pass_at_1"], 4) for m in ["grpo_from_base", "grpo_from_sft"]}, "(base 0.9962)")
"""),
    md("## 5. Save"),
    code("""
tracker = ExperimentTracker("exp02_grpo_evaluation", "/kaggle/working")
tracker.set_config(model=PRIMARY_MODEL_ID, dataset="mbpp_v2_eval.jsonl + test.jsonl",
                   hyperparameters={**GEN_CFG, "k_values": K_VALUES, "seed": SEED, "dtype": "float16", "merged": True})
wall_h = (time.time() - T0) / 3600
tracker.update_metrics({"runs": RUNS, "comparisons": rows, "pass_table": table.to_dict("index"),
                        "ood_pass_at_1": {m: RUNS[f"exp02_{m}_mbpp_eval"]["pass_at_1"] for m in ex}})
tracker.update_hardware(max(r["peak_vram_gb"] for r in RUNS.values()), psutil.virtual_memory().used / 1e9)
tracker.data["training"].update(wall_hours=round(wall_h, 4), gpu_hours=round(wall_h, 4)); tracker.save()
print(f"saved. wall = GPU hours = {wall_h:.2f}")
"""),
    md("""
## Takeaways (fill in)
- Which comparisons have a CI that excludes zero? Report only those as improvements.
- Did training reward rise (NB 06 curves) without the held-out score moving? That would indicate overfitting to 256 tasks.
- Next: `08_hard_example_mining` and `09_test_time_scaling` reuse the best model.
"""),
]

write_kernel("kaggle/kernels/07-grpo-evaluation", "07_grpo_evaluation.ipynb",
             "ycheng22/07-grpo-evaluation", "07 GRPO Evaluation", cells, gpu=True, internet=True,
             kernel_sources=["ycheng22/04-sft-training", "ycheng22/06-grpo-training",
                             "ycheng22/03-baseline-evaluation", "ycheng22/05-sft-evaluation"])
