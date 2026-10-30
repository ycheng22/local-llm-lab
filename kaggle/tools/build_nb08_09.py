"""
Generates kaggle/kernels/08-hard-example-mining and 09-test-time-scaling.
Run from the repo root: python kaggle/tools/build_nb08_09.py

MODEL_CHOICE (set it at the top of each notebook after NB 07 tells you the best model):
  base | sft | grpo_from_base | grpo_from_sft
Default is 'base' (needs no adapter), so both notebooks run as-is.
"""
import sys
sys.path.insert(0, "kaggle/tools")
from nbbuild import md, code, setup_cell, write_kernel

MODEL_CELL = '''
MODEL_CHOICE = "base"      # <- change after NB 07: base | sft | grpo_from_base | grpo_from_sft
def one(pattern):
    hits = glob.glob(pattern, recursive=True); assert hits, f"not found: {pattern}"; return hits[0]
ADAPTERS = {"base": None,
            "sft": lambda: [one(f"{find_input('04-sft-training')}/**/exp01_sft_mbpp_v1/checkpoint-final")],
            "grpo_from_base": lambda: [one(f"{find_input('06-grpo-training')}/**/grpo_from_base/checkpoint-final")],
            "grpo_from_sft": lambda: [one(f"{find_input('04-sft-training')}/**/exp01_sft_mbpp_v1/checkpoint-final"),
                                      one(f"{find_input('06-grpo-training')}/**/grpo_from_sft/checkpoint-final")]}
adapters = ADAPTERS[MODEL_CHOICE]() if ADAPTERS[MODEL_CHOICE] else None
print("model:", MODEL_CHOICE, adapters)
model, tokenizer = setup_model_and_tokenizer(PRIMARY_MODEL_ID, dtype="float16", device_map={"": 0}, adapter_path=adapters, merge=True)
model.eval()
'''

# ---------------------------------------------------------------- NB 08
cells08 = [
    md("""
# 08 - Hard-Example Mining (Phase 4, redesigned)

**Mode:** GPU (T4 x2 session, GPU 0), background commit, 1.5 h hard cap. Estimated ~0.6 GPU hours.

## Idea
GRPO only learns from prompts whose sampled group has **mixed outcomes** (some pass, some fail). If every sample passes or every sample fails, the group has zero reward variance and contributes nothing. Mining samples each training task n=4 times and classifies it by pass rate:

| class | passes out of 4 | learning signal for GRPO |
|---|---|---|
| solved | 3-4 | little (mostly all-pass groups) |
| borderline | 2 | **high** |
| hard | 1 | **high** |
| impossible | 0 | none (all-fail groups) |

## Why this replaces the local Phase 4 result
Locally, mining found **420/420 impossible**. NB 03 showed that was a prompt bug (the model never saw the function name). With the v2 prompt, the base model solves ~41% of tasks, so mining should now find a real mix. This notebook re-runs it on the 256 training tasks and builds three equal-sized GRPO training sets for an ablation: `hard` (borderline + hard), `random`, `mixed` (70% hard + 30% from solved/impossible).

Dev/eval tasks are never mined (AGENTS.md rule 7).
"""),
    setup_cell(install_wheels=False),
    code("""
import json, gc, torch, psutil, collections, pandas as pd
from llm_lab_utils import PRIMARY_MODEL_ID, SubprocessVerifier, ExperimentTracker, load_dataset
from llm_lab_utils.model_utils import setup_model_and_tokenizer
from llm_lab_utils.miner import mine_tasks
""" + MODEL_CELL + """
GEN_CFG = {"max_new_tokens": 384, "temperature": 0.7, "top_p": 0.95, "do_sample": True}     # = local mining_complex.yaml
N_SAMPLES, SEED = 4, 42
tasks = load_dataset(f"{DATA}/mbpp_v2_train.jsonl")
verifier = SubprocessVerifier(timeout=5.0, max_memory_mb=512)
print(len(tasks), "training tasks")
"""),
    md("## 1. Mine"),
    code("""
import torch
torch.manual_seed(SEED)
EXP_ID = f"exp03_mining_{MODEL_CHOICE}"
OUT = f"/kaggle/working/{EXP_ID}"
t0 = time.time(); torch.cuda.reset_peak_memory_stats(0)
report = mine_tasks(model, tokenizer, tasks, verifier, N_SAMPLES, GEN_CFG, EXP_ID, OUT,
                    dataset_name="mbpp_v2_train.jsonl", seed=SEED)
wall_h = (time.time() - t0) / 3600
print(report["classification_counts"], f"| {wall_h:.2f} h")
"""),
    md("## 2. Analysis: how much of the training set can actually teach GRPO?"),
    code("""
counts = report["classification_counts"]
n = sum(counts.values())
mixed_share = (counts["borderline"] + counts["hard"]) / n
print(pd.Series(counts).to_frame("tasks").assign(share=lambda d: (d.tasks / n).round(3)).to_string())
print(f"\\ntasks with a learning signal (borderline+hard): {mixed_share:.1%}")
per_task_c = [t["c_correct"] for t in report["tasks"]]
print("pass-count histogram (0..4):", dict(sorted(collections.Counter(per_task_c).items())))

# does the mined difficulty track reference-solution length? (a cheap sanity check on the classes)
ref_len = {t.id: len(t.reference_solution or "") for t in tasks}
by_cls = collections.defaultdict(list)
for t in report["tasks"]: by_cls[t["category"]].append(ref_len[t["task_id"]])
print(pd.DataFrame({k: {"n": len(v), "mean_ref_chars": round(sum(v) / len(v), 1)} for k, v in by_cls.items()}).T.to_string())
files = {f: sum(1 for _ in open(f"{OUT}/{f}")) for f in ["hard.jsonl", "random.jsonl", "mixed.jsonl"]}
print("\\nablation set sizes (should be equal):", files)
"""),
    md("## 3. Save"),
    code("""
tracker = ExperimentTracker(EXP_ID + "_summary", "/kaggle/working")
tracker.set_config(model=f"{PRIMARY_MODEL_ID} ({MODEL_CHOICE}: {adapters})", dataset="mbpp_v2_train.jsonl",
                   hyperparameters={**GEN_CFG, "n_samples": N_SAMPLES, "seed": SEED, "dtype": "float16", "merged": True})
wall_total = (time.time() - T0) / 3600
tracker.update_metrics({"counts": counts, "mixed_share": mixed_share, "ablation_sizes": files})
tracker.update_hardware(torch.cuda.max_memory_allocated(0) / 1e9, psutil.virtual_memory().used / 1e9)
tracker.data["training"].update(wall_hours=round(wall_total, 4), gpu_hours=round(wall_total, 4)); tracker.save()
print(f"saved. wall = GPU hours = {wall_total:.2f}")
"""),
    md("""
## Takeaways (fill in)
- What share of training tasks has a learning signal? If it is small, GRPO on all 256 tasks wastes most of its steps; if large, the 0.25-0.35 train reward in NB 06 was mostly diversity limited, not task limited.
- Next: a GRPO ablation on `hard.jsonl` vs `random.jsonl` vs `mixed.jsonl` (same size, same hyperparameters) answers whether mining pays for itself in GPU-hours.
"""),
]
write_kernel("kaggle/kernels/08-hard-example-mining", "08_hard_example_mining.ipynb",
             "ycheng22/08-hard-example-mining", "08 Hard Example Mining", cells08, gpu=True, internet=True,
             kernel_sources=["ycheng22/04-sft-training", "ycheng22/06-grpo-training"])

# ---------------------------------------------------------------- NB 09
cells09 = [
    md("""
# 09 - Test-Time Compute Scaling (Phase 5, redesigned)

**Mode:** GPU (T4 x2 session, GPU 0), background commit, 2 h hard cap. Estimated ~1.0 GPU hour.

## Question
If we let the model try more times at inference, how much does accuracy improve, and how much of that can we realise **without an oracle**?

## Two curves from ONE batch of n=16 samples per task
| curve | meaning |
|---|---|
| **Oracle Pass@k** | unbiased estimate that at least one of k samples passes all tests (needs the hidden tests to know which one) |
| **Visible-test selection** | the model sees one example test in its prompt. Draw k samples, keep those that pass that example, pick one at random, score it on the full tests. This is deployable: no hidden information |

Computing K = 1, 2, 4, 8, 16 from a single set of 16 samples is ~5x cheaper than five separate runs (the local Phase 5 ran each K separately).

The gap between the two curves is the cost of not having an oracle; samples that pass the visible example but fail hidden tests are the "false friends".
"""),
    setup_cell(install_wheels=False),
    code("""
import json, gc, torch, psutil, pandas as pd
from concurrent.futures import ThreadPoolExecutor
from llm_lab_utils import (PRIMARY_MODEL_ID, SubprocessVerifier, ExperimentTracker, load_dataset, pass_at_k)
from llm_lab_utils.model_utils import setup_model_and_tokenizer
from llm_lab_utils.evaluator import run_evaluation
from llm_lab_utils.metrics import selection_accuracy
""" + MODEL_CELL + """
GEN_CFG = {"max_new_tokens": 384, "temperature": 0.8, "top_p": 0.95, "do_sample": True}   # = local search_complex.yaml (higher T for diversity)
N, KS, SEED = 16, [1, 2, 4, 8, 16], 1234
tasks = load_dataset(f"{DATA}/mbpp_v2_eval.jsonl")
verifier = SubprocessVerifier(timeout=5.0, max_memory_mb=512)
"""),
    md("## 1. Sample 16 completions per task and verify on the full tests"),
    code("""
EXP_ID = f"exp04_scaling_{MODEL_CHOICE}"
t0 = time.time(); torch.cuda.reset_peak_memory_stats(0)
tracker = ExperimentTracker(EXP_ID, "/kaggle/working")
tracker.set_config(model=f"{PRIMARY_MODEL_ID} ({MODEL_CHOICE}: {adapters})", dataset="mbpp_v2_eval.jsonl",
                   hyperparameters={**GEN_CFG, "n": N, "k_values": KS, "seed": SEED, "dtype": "float16", "merged": True})
metrics, task_res = run_evaluation(model, tokenizer, tasks, verifier, GEN_CFG, KS, tracker=tracker, seed=SEED, sample_chunk=16)
gen_hours = (time.time() - t0) / 3600
print({k: round(v, 4) for k, v in metrics.items()}, f"| {gen_hours:.2f} h")
"""),
    md("## 2. Re-verify every sample against the VISIBLE example test only"),
    code("""
gens = [json.loads(l) for l in open(f"/kaggle/working/{EXP_ID}/generations.jsonl")]
by_id = {t.id: t.model_dump() for t in tasks}

def visible_only(g):
    td = dict(by_id[g["task_id"]]); td["hidden_tests"] = []
    return g["task_id"], bool(verifier.verify(td, g["candidate_code"])["passed"]), bool(g["verification"]["passed"])

with ThreadPoolExecutor(max_workers=8) as ex:
    rows = list(ex.map(visible_only, gens))
per_task = {}
for tid, vis, full in rows: per_task.setdefault(tid, []).append({"visible": vis, "full": full})
assert all(len(v) == N for v in per_task.values()), {len(v) for v in per_task.values()}
n_vis = sum(s["visible"] for v in per_task.values() for s in v); n_false = sum(s["visible"] and not s["full"] for v in per_task.values() for s in v)
print(f"samples passing the visible example: {n_vis}/{len(rows)};  of those, failing hidden tests ('false friends'): {n_false} ({n_false / max(n_vis, 1):.1%})")
"""),
    md("## 3. Scaling curves"),
    code("""
oracle = {k: sum(pass_at_k(N, sum(s["full"] for s in v), k) for v in per_task.values()) / len(per_task) for k in KS}
select = {k: sum(selection_accuracy(v, k, seed=7) for v in per_task.values()) / len(per_task) for k in KS}
curve = pd.DataFrame({"oracle_pass@k": oracle, "visible_test_selection": select}).round(3)
curve["gap"] = (curve["oracle_pass@k"] - curve["visible_test_selection"]).round(3)
curve["extra_samples_vs_k1"] = [k for k in KS]
print(curve.to_string())

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.figure(figsize=(6.5, 4)); plt.plot(KS, list(oracle.values()), "o-", label="oracle Pass@k")
plt.plot(KS, list(select.values()), "s-", label="visible-test selection"); plt.xscale("log", base=2); plt.xticks(KS, KS)
plt.xlabel("samples per task (k)"); plt.ylabel("accuracy on held-out MBPP"); plt.title(f"Test-time scaling ({MODEL_CHOICE})")
plt.legend(); plt.grid(alpha=.3); plt.savefig(f"/kaggle/working/{EXP_ID}/scaling_curve.png", dpi=120)
"""),
    md("## 4. Save"),
    code("""
tracker.update_metrics({"oracle_pass_at_k": oracle, "visible_selection": select, "false_friend_rate": n_false / max(n_vis, 1)})
wall_total = (time.time() - T0) / 3600
tracker.update_hardware(torch.cuda.max_memory_allocated(0) / 1e9, psutil.virtual_memory().used / 1e9)
tracker.data["training"].update(wall_hours=round(wall_total, 4), gpu_hours=round(wall_total, 4)); tracker.save()
print(f"saved. wall = GPU hours = {wall_total:.2f}")
"""),
    md("""
## Takeaways (fill in)
- How much does accuracy grow from k=1 to k=16, and does it flatten (diminishing returns per GPU-hour)?
- How much of the oracle gain survives with only the visible example as a filter? The gap is the price of a deployable system.
- Compare with RL: if GRPO's Pass@1 (NB 07) reaches the visible-selection accuracy at k=4 or 8, training moved compute from inference to the weights.
"""),
]
write_kernel("kaggle/kernels/09-test-time-scaling", "09_test_time_scaling.ipynb",
             "ycheng22/09-test-time-scaling", "09 Test Time Scaling", cells09, gpu=True, internet=True,
             kernel_sources=["ycheng22/04-sft-training", "ycheng22/06-grpo-training"])
