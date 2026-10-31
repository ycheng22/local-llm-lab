"""Generates kaggle/kernels/x02-humaneval-ood (plan Step 19b). Run from the repo root: python kaggle/tools/build_x02.py"""
import sys
sys.path.insert(0, "kaggle/tools")
from nbbuild import md, code, setup_cell, write_kernel

cells = [
    md("""
# x02 - HumanEval: does the RL gain transfer to another benchmark? (Phase 6.5, Step 19b)

**Mode:** GPU (T4 x2 session, GPU 0), background commit, 2 h hard cap. Estimated ~1.4 GPU hours (2 models x ~157 tasks x 8 samples; HumanEval answers are longer than MBPP).

## Why
Every result so far is on 100 MBPP tasks that came from the same source as the GRPO training tasks (`mbpp_v2_train`). A reviewer's first question: *did RL learn to code better, or to write MBPP-style answers?* HumanEval has a different author, style and task format (a signature plus a docstring, no example `assert`), so it is a real out-of-distribution test. AGENTS.md also requires an OOD Pass@1.

## How the benchmark was built (read this before trusting any number)
- Source: the official `openai_humaneval` test split (164 problems). Converted **without executing any code locally** by `kaggle/tools/build_humaneval.py`.
- Each plain `assert candidate(...) == y` in `check()` becomes one (input, expected) test. Other boolean asserts become (expression, True).
- **7 problems are dropped** because their `check()` contains loops, random inputs or helper code; the audit file lists them. So we report "the 157 HumanEval problems expressible as plain asserts", not "HumanEval".
- Imports and helper functions from the original prompt are added by `PreludeVerifier`, as the original harness appends the completion to the full prompt.
- **Step 1 below runs every canonical solution through the verifier. Any problem whose reference fails is excluded** (and listed), the same audit-first rule as NB 02.
- The file is a **new frozen eval set** (hash in `MANIFEST.sha256`). It is never used to choose hyperparameters.

## Pre-registered settings (fixed before seeing results)
Same sampling as NB 03/07 (temperature 0.7, top-p 0.9, n=8, seed 1234), except `max_new_tokens=512` because HumanEval functions with docstrings are longer. `truncated_rate` is reported.

Inputs via `kernel_sources`: `06-grpo-training` (the `grpo_from_base` adapter).
"""),
    setup_cell(install_wheels=False, single_gpu=True),
    code("""
import json, gc, torch, psutil, pandas as pd
from llm_lab_utils import PRIMARY_MODEL_ID, ExperimentTracker, load_dataset, paired_bootstrap_diff, pass_at_k
from llm_lab_utils.humaneval import PreludeVerifier
from llm_lab_utils.model_utils import setup_model_and_tokenizer
from llm_lab_utils.evaluator import run_evaluation

def one(pattern):
    hits = glob.glob(pattern, recursive=True); assert hits, f"not found: {pattern}"; return hits[0]

GEN_CFG = {"max_new_tokens": 512, "temperature": 0.7, "top_p": 0.9, "do_sample": True}
K_VALUES, SEED = [1, 4, 8], 1234
verifier = PreludeVerifier(timeout=5.0, max_memory_mb=512)
all_tasks = load_dataset(f"{DATA}/humaneval_v1.jsonl")
print(len(all_tasks), "converted HumanEval tasks;", json.load(open(f"{DATA}/humaneval_v1_audit.json"))["dropped"].__len__(), "dropped at conversion")
"""),
    md("## 1. Audit: every canonical solution must pass its converted tests"),
    code("""
bad = [t.id for t in all_tasks if not verifier.verify(t.model_dump(), t.reference_solution)["passed"]]
tasks = [t for t in all_tasks if t.id not in set(bad)]
print(f"reference solutions failing the converted tests: {len(bad)} -> excluded: {bad}")
print(f"evaluation set: {len(tasks)} tasks")
os.makedirs("/kaggle/working/x02_humaneval", exist_ok=True)
json.dump({"excluded_reference_failures": bad, "n_eval_tasks": len(tasks)}, open("/kaggle/working/x02_humaneval/audit.json", "w"), indent=1)
"""),
    md("## 2. Evaluate base and GRPO-from-base (adapter merged)"),
    code("""
G_BASE = one(f"{find_input('06-grpo-training')}/**/grpo_from_base/checkpoint-final")
print(G_BASE, json.load(open(f"{G_BASE}/experiment_meta.json"))["parent_checkpoint"])
RUNS, COUNTS = {}, {}
for name, adapters in [("base", None), ("grpo_from_base", [G_BASE])]:
    model, tok = setup_model_and_tokenizer(PRIMARY_MODEL_ID, dtype="float16", device_map={"": 0}, adapter_path=adapters, merge=True)
    model.eval()
    exp_id = f"x02_humaneval_{name}"
    t0 = time.time(); torch.cuda.reset_peak_memory_stats(0)
    tracker = ExperimentTracker(exp_id, "/kaggle/working")
    tracker.set_config(model=f"{PRIMARY_MODEL_ID} ({name})", dataset="humaneval_v1.jsonl (reference-audited)",
                       hyperparameters={**GEN_CFG, "k_values": K_VALUES, "seed": SEED, "dtype": "float16", "merged": True, "n_tasks": len(tasks)})
    metrics, task_res = run_evaluation(model, tok, tasks, verifier, GEN_CFG, K_VALUES, tracker=tracker, seed=SEED)
    wall_h = (time.time() - t0) / 3600
    tracker.data["training"].update(wall_hours=round(wall_h, 4), gpu_hours=round(wall_h, 4)); tracker.save()
    RUNS[name] = {**metrics, "peak_vram_gb": tracker.data["hardware"]["peak_vram_gb"], "wall_hours": wall_h}
    COUNTS[name] = {r["task_id"]: r["c"] for r in task_res}
    print(exp_id, {k: round(v, 4) for k, v in RUNS[name].items()})
    del model, tok; gc.collect(); torch.cuda.empty_cache()
"""),
    md("## 3. Paired comparison and the MBPP contrast"),
    code("""
rows = []
for k in K_VALUES:
    r = paired_bootstrap_diff(COUNTS["base"], COUNTS["grpo_from_base"], n=8, k=k)
    rows.append({"metric": f"Pass@{k}", "base": round(r["mean_a"], 3), "grpo_from_base": round(r["mean_b"], 3), "diff": round(r["diff"], 3),
                 "95% CI": f"[{r['ci95_low']:+.3f}, {r['ci95_high']:+.3f}]", "significant": r["ci95_low"] > 0 or r["ci95_high"] < 0})
cmp_df = pd.DataFrame(rows); print(cmp_df.to_string(index=False))
print("\\nMBPP (NB 07, 100 tasks) for contrast: Pass@1 base 0.406 -> GRPO 0.585 (+0.179, CI [+0.124, +0.239])")
print(pd.DataFrame({m: {x: round(RUNS[m][x], 3) for x in ["truncated_rate", "syntax_error_rate", "timeout_rate", "gen_tokens_per_s"]} for m in RUNS}).T.to_string())
"""),
    md("## 4. Save"),
    code("""
tracker = ExperimentTracker("x02_humaneval_summary", "/kaggle/working")
tracker.set_config(model=PRIMARY_MODEL_ID, dataset="humaneval_v1.jsonl", hyperparameters={**GEN_CFG, "k_values": K_VALUES, "seed": SEED, "n_tasks": len(tasks)})
wall_h = (time.time() - T0) / 3600
tracker.update_metrics({"runs": RUNS, "comparison": rows, "excluded_reference_failures": bad,
                        "ood_pass_at_1": {m: RUNS[m]["pass_at_1"] for m in RUNS}})
tracker.update_hardware(max(r["peak_vram_gb"] for r in RUNS.values()), psutil.virtual_memory().used / 1e9)
tracker.data["training"].update(wall_hours=round(wall_h, 4), gpu_hours=round(wall_h, 4)); tracker.save()
print(f"saved. wall = GPU hours = {wall_h:.2f}")
"""),
    md("""
## Takeaways (fill in)
- Does the GRPO gain have a CI above zero on HumanEval? If yes, RL improved general code generation; if it is much smaller than on MBPP, part of the MBPP gain is style/format adaptation to MBPP-like prompts.
- Compare `truncated_rate` with MBPP's 8.5%: HumanEval docstring prompts may behave differently.
- Next: `x03-4b-baseline`, `x05-grpo-seeds`.
"""),
]

write_kernel("kaggle/kernels/x02-humaneval-ood", "x02_humaneval_ood.ipynb", "ycheng22/x02-humaneval-ood", "x02 HumanEval OOD", cells,
             gpu=True, internet=True, kernel_sources=["ycheng22/06-grpo-training"])
