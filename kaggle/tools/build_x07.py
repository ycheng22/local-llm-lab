"""Generates kaggle/kernels/x07-robustness (plan Step 19h). Run from the repo root: python kaggle/tools/build_x07.py"""
import sys
sys.path.insert(0, "kaggle/tools")
from nbbuild import md, code, setup_cell, write_kernel

cells = [
    md("""
# x07 - Robustness: rename the function and see if the gain survives (Phase 6.5, Step 19h)

**Mode:** GPU (T4 x2 session, GPU 0), background commit, 1.5 h hard cap. Estimated ~0.6 GPU hours.

## Question
GRPO gave +17.9 pt on MBPP but ~0 on HumanEval (x02), and failure analysis (19a) showed the base model often defines the wrong function name. Did GRPO learn to **copy the name and conventions from the prompt's example assert**, or does it also hold when the surface form changes?

## Transformation (deterministic, semantics-preserving)
For each of the 100 `mbpp_v2_eval` tasks the function name is replaced everywhere (prompt text and example assert, visible tests, hidden tests, reference solution) by a **neutral, unrelated name** chosen from a fixed list by task index. The task logic is unchanged, but the name no longer hints at the task (`first_repeated_char` -> `compute_result`). Nothing else changes: same sampling (temperature 0.7, top-p 0.9, 384 tokens, seed 1234, fp16), n=4.

The renamed set is a **new frozen eval set**: it is written to the outputs with its sha256, derived only from the eval split, and never used to choose hyperparameters. Every reference solution is run against the renamed tests first (audit-first rule); failing tasks would be excluded.

## How to read it
- Pass@1 renamed vs original, per model (paired bootstrap over tasks; original = n=8 counts from NB 03 / NB 07).
- **Interaction:** is the GRPO-minus-base gap still ~+18 pt on renamed tasks? A big shrink means a part of the RL gain depended on the specific name/format cues.
- `wrong_function_name` share on the renamed set for both models.

Inputs via `kernel_sources`: `03-baseline-evaluation`, `06-grpo-training`, `07-grpo-evaluation`.
"""),
    setup_cell(install_wheels=False, single_gpu=True),
    code("""
import json, gc, re, hashlib, random, torch, psutil, pandas as pd
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

GEN_CFG = {"max_new_tokens": 384, "temperature": 0.7, "top_p": 0.9, "do_sample": True}
K_VALUES, SEED = [1, 4], 1234
N_SAMPLES = max(K_VALUES)
REF = {"base": counts_from(f"{find_input('03-baseline-evaluation')}/**/exp00_baseline_mbpp_v2/generations.jsonl"),
       "grpo_from_base": counts_from(f"{find_input('07-grpo-evaluation')}/**/exp02_grpo_from_base_mbpp_eval/generations.jsonl")}
G_BASE = one(f"{find_input('06-grpo-training')}/**/grpo_from_base/checkpoint-final")
verifier = SubprocessVerifier(timeout=5.0, max_memory_mb=512)
"""),
    md("## 1. Build the renamed eval set and audit it"),
    code("""
NEW_NAMES = ["compute_result", "process_data", "handle_input", "run_task", "transform_value", "evaluate_case", "apply_rule", "calculate_output"]
raw = [json.loads(l) for l in open(f"{DATA}/mbpp_v2_eval.jsonl")]
assert len(raw) == 100

def rename_task(t, new):
    m = re.search(r"assert\\s*\\(*\\s*(\\w+)\\(", t["prompt"]); old = m.group(1)    # the name used by the prompt's example assert
    sub = lambda s: re.sub(rf"\\b{re.escape(old)}\\b", new, s)
    t = json.loads(json.dumps(t))
    t["prompt"] = sub(t["prompt"]); t["reference_solution"] = sub(t["reference_solution"])
    for k in ("visible_tests", "hidden_tests"):
        for x in t[k]: x["input"] = sub(x["input"])
    t["id"] = t["id"] + "_r"
    t["metadata"] = {**t["metadata"], "prompt_variant": "v2_renamed_function", "original_id": t["id"][:-2], "old_name": old, "new_name": new}
    return t

renamed = [rename_task(t, NEW_NAMES[i % len(NEW_NAMES)]) for i, t in enumerate(raw)]
os.makedirs("/kaggle/working/x07_robustness", exist_ok=True)
path = "/kaggle/working/x07_robustness/mbpp_v2_eval_renamed.jsonl"
with open(path, "w") as f:
    for t in renamed: f.write(json.dumps(t) + "\\n")
sha = hashlib.sha256(open(path, "rb").read()).hexdigest(); print("renamed set sha256:", sha)
print(renamed[0]["prompt"])
tasks_all = load_dataset(path)
bad = [t.id for t in tasks_all if not verifier.verify(t.model_dump(), t.reference_solution)["passed"]]
tasks = [t for t in tasks_all if t.id not in set(bad)]
print(f"reference solutions failing the renamed tests: {len(bad)} -> excluded {bad}; evaluation set: {len(tasks)} tasks")
json.dump({"sha256": sha, "excluded": bad, "n_tasks": len(tasks)}, open("/kaggle/working/x07_robustness/audit.json", "w"), indent=1)
"""),
    md("## 2. Evaluate base and GRPO-from-base on the renamed tasks (n=4)"),
    code("""
RUNS, COUNTS = {}, {}
for name, adapters in [("base", None), ("grpo_from_base", [G_BASE])]:
    model, tok = setup_model_and_tokenizer(PRIMARY_MODEL_ID, dtype="float16", device_map={"": 0}, adapter_path=adapters, merge=True)
    model.eval()
    exp_id = f"x07_renamed_{name}"
    t0 = time.time(); torch.cuda.reset_peak_memory_stats(0)
    tracker = ExperimentTracker(exp_id, "/kaggle/working")
    tracker.set_config(model=f"{PRIMARY_MODEL_ID} ({name})", dataset=f"mbpp_v2_eval_renamed.jsonl (sha256 {sha[:12]})",
                       hyperparameters={**GEN_CFG, "k_values": K_VALUES, "seed": SEED, "dtype": "float16", "merged": True,
                                        "parent_checkpoint": G_BASE if adapters else None, "n_tasks": len(tasks)})
    metrics, task_res = run_evaluation(model, tok, tasks, verifier, GEN_CFG, K_VALUES, tracker=tracker, seed=SEED)
    wall_h = (time.time() - t0) / 3600
    tracker.data["training"].update(wall_hours=round(wall_h, 4), gpu_hours=round(wall_h, 4)); tracker.save()
    RUNS[name] = {**metrics, "peak_vram_gb": tracker.data["hardware"]["peak_vram_gb"], "wall_hours": wall_h}
    COUNTS[name] = {r["task_id"][:-2]: r["c"] for r in task_res}      # strip the "_r" suffix to pair with the original task
    print(exp_id, {k: round(v, 4) for k, v in RUNS[name].items()})
    del model, tok; gc.collect(); torch.cuda.empty_cache()
"""),
    md("## 3. Does the gain survive the rename?"),
    code("""
def boot(a, b, n_boot=10000, seed=0):
    rng = random.Random(seed); d = [y - x for x, y in zip(a, b)]
    m = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(n_boot))
    return sum(d) / len(d), m[int(0.025 * n_boot)], m[int(0.975 * n_boot)]
ids = sorted(COUNTS["base"])
orig = {m: [REF[m][i] / 8 for i in ids] for m in REF}
ren = {m: [COUNTS[m][i] / N_SAMPLES for i in ids] for m in COUNTS}
rows = []
for m in REF:
    d, lo, hi = boot(orig[m], ren[m])
    rows.append({"comparison": f"{m}: renamed - original", "original": round(sum(orig[m]) / len(ids), 3), "renamed": round(sum(ren[m]) / len(ids), 3),
                 "diff": round(d, 3), "95% CI": f"[{lo:+.3f}, {hi:+.3f}]", "significant": lo > 0 or hi < 0})
d0, lo0, hi0 = boot(orig["base"], orig["grpo_from_base"]); d1, lo1, hi1 = boot(ren["base"], ren["grpo_from_base"])
rows.append({"comparison": "GRPO - base on ORIGINAL names", "original": "", "renamed": "", "diff": round(d0, 3), "95% CI": f"[{lo0:+.3f}, {hi0:+.3f}]", "significant": lo0 > 0 or hi0 < 0})
rows.append({"comparison": "GRPO - base on RENAMED", "original": "", "renamed": "", "diff": round(d1, 3), "95% CI": f"[{lo1:+.3f}, {hi1:+.3f}]", "significant": lo1 > 0 or hi1 < 0})
# interaction: (GRPO - base) renamed minus (GRPO - base) original, per task
inter_a = [g - b for g, b in zip(orig["grpo_from_base"], orig["base"])]
inter_b = [g - b for g, b in zip(ren["grpo_from_base"], ren["base"])]
di, loi, hii = boot(inter_a, inter_b)
rows.append({"comparison": "interaction: gap(renamed) - gap(original)", "original": "", "renamed": "", "diff": round(di, 3), "95% CI": f"[{loi:+.3f}, {hii:+.3f}]", "significant": loi > 0 or hii < 0})
cmp_df = pd.DataFrame(rows); print(cmp_df.to_string(index=False))

def name_err_share(exp_id):
    gens = [json.loads(l) for l in open(f"/kaggle/working/{exp_id}/generations.jsonl")]
    return sum(bool(re.search(r"name '\\w+' is not defined", g["verification"]["stderr"])) for g in gens) / len(gens)
NAME_ERR = {m: name_err_share(f"x07_renamed_{m}") for m in RUNS}
print("share of samples with 'name ... is not defined':", {k: round(v, 3) for k, v in NAME_ERR.items()}, "(original, 19a: base 0.092, GRPO 0.006)")
print(pd.DataFrame({m: {x: round(RUNS[m][x], 3) for x in ["pass_at_1", "pass_at_4", "truncated_rate", "syntax_error_rate", "timeout_rate"]} for m in RUNS}).T.to_string())
"""),
    md("## 4. Save"),
    code("""
tracker = ExperimentTracker("x07_robustness_summary", "/kaggle/working")
tracker.set_config(model=PRIMARY_MODEL_ID, dataset=f"mbpp_v2_eval_renamed.jsonl (sha256 {sha})", hyperparameters={**GEN_CFG, "k_values": K_VALUES, "seed": SEED, "n_tasks": len(tasks)})
wall_h = (time.time() - T0) / 3600
tracker.update_metrics({"runs": RUNS, "comparisons": rows, "name_error_share": NAME_ERR, "renamed_set_sha256": sha,
                        "ood_pass_at_1": {m: RUNS[m]["pass_at_1"] for m in RUNS}})
tracker.update_hardware(max(r["peak_vram_gb"] for r in RUNS.values()), psutil.virtual_memory().used / 1e9)
tracker.data["training"].update(wall_hours=round(wall_h, 4), gpu_hours=round(wall_h, 4)); tracker.save()
print(f"saved. wall = GPU hours = {wall_h:.2f}")
"""),
    md("""
## Takeaways (fill in)
- Does GRPO still beat base by ~+18 pt with neutral function names? If the interaction CI excludes zero and the gap shrinks, part of the RL gain depended on name/format cues.
- Did `wrong_function_name` errors change for the base model?
"""),
]

write_kernel("kaggle/kernels/x07-robustness", "x07_robustness.ipynb", "ycheng22/x07-robustness", "x07 Robustness", cells,
             gpu=True, internet=True,
             kernel_sources=["ycheng22/03-baseline-evaluation", "ycheng22/06-grpo-training", "ycheng22/07-grpo-evaluation"])
