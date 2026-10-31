"""
Generates kaggle/kernels/x05-grpo-seeds (plan Step 19f) and x06-grpo-mining-ablation (plan Step 19g).
Run from the repo root: python kaggle/tools/build_x05_x06.py

Both train two GRPO arms in parallel (one per T4, the NB 06 recipe, only the stated factor changes) and then evaluate every
trained adapter on the frozen mbpp_v2_eval (n=8, same sampling and seed as NB 07) plus the first 100 synthetic test tasks.
"""
import sys
sys.path.insert(0, "kaggle/tools")
from nbbuild import md, code, setup_cell, write_kernel

SPECS = {
    "x05": dict(
        name="x05-grpo-seeds", title="x05 GRPO Seeds", nb="x05_grpo_seeds.ipynb", exp="x05_grpo_seeds",
        arms='[("seed43", "0", 43, "mbpp_v2_train.jsonl"), ("seed44", "1", 44, "mbpp_v2_train.jsonl")]',
        heading="x05 - Is the GRPO gain stable across random seeds? (Phase 6.5, Step 19f)",
        question="""
NB 06/07 trained `grpo_from_base` once (seed 42) and measured +17.9 pt Pass@1 over base. A single run cannot separate **the effect of RL** from **luck in that one run** (sampling noise during training). We train the *identical* recipe twice more, changing **only the seed** (43 and 44), and evaluate on the same frozen 100 tasks.

| Arm | GPU | Seed | Train tasks |
|---|---|---|---|
| `seed43` | 0 | 43 | `mbpp_v2_train` (256) |
| `seed44` | 1 | 44 | `mbpp_v2_train` (256) |
| `seed42` | - | 42 | from NB 06/07 (not retrained) |

Reading: mean +/- sample std of Pass@1 over the 3 seeds; each seed also gets a paired bootstrap CI against base. If the seeds agree within about 2-3 pt, the +17.9 pt is a property of the recipe; if they spread widely, the blog must report the range.""",
        refs="""
REFS = {"base": counts_from(f"{find_input('03-baseline-evaluation')}/**/exp00_baseline_mbpp_v2/generations.jsonl"),
        "seed42": counts_from(f"{find_input('07-grpo-evaluation')}/**/exp02_grpo_from_base_mbpp_eval/generations.jsonl")}
""",
        sources=["ycheng22/03-baseline-evaluation", "ycheng22/07-grpo-evaluation"],
        compare="""
import statistics as st
ALL = {**REFS, **COUNTS}
def p1(c): return sum(c.values()) / (8 * len(c))
seed_arms = ["seed42", "seed43", "seed44"]
rows = []
for a in seed_arms:
    for k in K_VALUES:
        r = paired_bootstrap_diff(ALL["base"], ALL[a], n=8, k=k)
        rows.append({"arm": a, "metric": f"Pass@{k}", "base": round(r["mean_a"], 3), "arm_value": round(r["mean_b"], 3), "diff": round(r["diff"], 3),
                     "95% CI": f"[{r['ci95_low']:+.3f}, {r['ci95_high']:+.3f}]", "significant": r["ci95_low"] > 0 or r["ci95_high"] < 0})
cmp_df = pd.DataFrame(rows); print(cmp_df.to_string(index=False))
p1s = [p1(ALL[a]) for a in seed_arms]
SEED_STATS = {"pass_at_1_per_seed": dict(zip(seed_arms, [round(x, 4) for x in p1s])), "mean": round(st.mean(p1s), 4), "std": round(st.stdev(p1s), 4),
              "min": round(min(p1s), 4), "max": round(max(p1s), 4)}
print("\\nPass@1 over 3 seeds:", SEED_STATS)
pair_rows = []
for a, b in [("seed43", "seed42"), ("seed44", "seed42"), ("seed44", "seed43")]:
    r = paired_bootstrap_diff(ALL[a], ALL[b], n=8, k=1)
    pair_rows.append({"comparison": f"{b} - {a}", "diff": round(r["diff"], 3), "95% CI": f"[{r['ci95_low']:+.3f}, {r['ci95_high']:+.3f}]"})
print(pd.DataFrame(pair_rows).to_string(index=False))
EXTRA = {"seed_stats": SEED_STATS, "seed_pairs": pair_rows}
""",
        takeaways="""
- Spread of Pass@1 across seeds 42/43/44: is the +17.9 pt (seed 42) typical, or on the high side?
- Are all three seeds significantly above base? Are seeds significantly different from each other (pairwise CIs)?
- Training cost per seed (GPU h) -> cost of a variance estimate.""",
    ),
    "x06": dict(
        name="x06-grpo-mining-ablation", title="x06 GRPO Mining Ablation", nb="x06_grpo_mining_ablation.ipynb", exp="x06_grpo_mining",
        arms='[("learnable97", "0", 42, "mbpp_v2_train_learnable.jsonl"), ("random97", "1", 42, "mbpp_v2_train_random.jsonl")]',
        heading="x06 - Do mined 'learnable' tasks beat random tasks for GRPO? (Phase 6.5, Step 19g)",
        question="""
NB 08 found that **62.1% of training groups have zero reward variance** (28 tasks the base model never solves, 36 it always solves, i.e. no gradient). Only the 97 tasks with 1-3 passes out of 4 can teach it anything. The Phase 4 hypothesis: **training only on those 97 should be at least as good as training on 97 random tasks, with fewer wasted groups.**

| Arm | GPU | Train set (97 tasks each) | Base pass-count histogram (0/1/2/3/4) |
|---|---|---|---|
| `learnable97` | 0 | tasks with 1, 2 or 3 passes of 4 (`mbpp_v2_train_learnable.jsonl`) | 0 / 33 / 34 / 30 / 0 |
| `random97` | 1 | 97 tasks drawn uniformly from the 256 (seed 0) (`mbpp_v2_train_random.jsonl`) | 28 / 7 / 15 / 11 / 36 |

Everything else equals NB 06 `grpo_from_base` (seed 42, 200 steps, lr 2e-5, 4 prompts/step, G=4, 384 tokens, pass/fail reward). 200 steps x 4 prompts = 800 prompt draws = about 8 passes over 97 tasks (versus about 3 over 256 in NB 06): both arms get the same, so the comparison is fair, but over-fitting to 97 tasks is possible and is part of what is measured. The mining labels come from the **training** split only; eval tasks are never mined.""",
        refs="""
REFS = {"base": counts_from(f"{find_input('03-baseline-evaluation')}/**/exp00_baseline_mbpp_v2/generations.jsonl"),
        "grpo_full256": counts_from(f"{find_input('07-grpo-evaluation')}/**/exp02_grpo_from_base_mbpp_eval/generations.jsonl")}
""",
        sources=["ycheng22/03-baseline-evaluation", "ycheng22/07-grpo-evaluation"],
        compare="""
ALL = {**REFS, **COUNTS}
rows = []
for new, old in [("learnable97", "base"), ("random97", "base"), ("learnable97", "random97"), ("learnable97", "grpo_full256"), ("random97", "grpo_full256")]:
    for k in K_VALUES:
        r = paired_bootstrap_diff(ALL[old], ALL[new], n=8, k=k)
        rows.append({"comparison": f"{new} - {old}", "metric": f"Pass@{k}", "old": round(r["mean_a"], 3), "new": round(r["mean_b"], 3), "diff": round(r["diff"], 3),
                     "95% CI": f"[{r['ci95_low']:+.3f}, {r['ci95_high']:+.3f}]", "significant": r["ci95_low"] > 0 or r["ci95_high"] < 0})
cmp_df = pd.DataFrame(rows); print(cmp_df.to_string(index=False))
# signal efficiency: share of groups with zero reward variance during training, and gain per training GPU hour
roi = []
for arm, m in metas.items():
    h = [e for e in hist.get(arm, []) if "frac_reward_zero_std" in e]
    gain = (sum(ALL[arm].values()) - sum(ALL["base"].values())) / 800
    roi.append({"arm": arm, "mean_zero_std_frac": round(sum(e["frac_reward_zero_std"] for e in h) / max(len(h), 1), 3),
                "train_gpu_hours": round(m["gpu_hours"], 3), "pass_at_1_gain": round(gain, 3), "gain_per_train_gpu_hour": round(gain / m["gpu_hours"], 3)})
print(pd.DataFrame(roi).to_string(index=False))
EXTRA = {"roi": roi}
""",
        takeaways="""
- `learnable97` vs `random97`: Pass@1 difference and CI. Did the zero-variance fraction fall for `learnable97`?
- Does either 97-task set reach the 256-task result (+17.9 pt)? If yes, data selection saves 62% of the training data; if no, the extra tasks mattered more than their difficulty.
- Check `reward_curves.png` for over-fitting (train reward -> 1.0 while eval stalls).""",
    ),
}


def build(key):
    s = SPECS[key]
    cells = [
        md(f"""
# {s['heading']}

**Mode:** GPU (T4 x2), background commit, 5 h hard cap. Estimated ~3.5 GPU hours billed: training of both arms in parallel (about 2.3 h, as in NB 06) + sequential evaluation (about 1 h).

## Question
{s['question']}

## Pre-registered settings (identical to NB 06 except the factor above)
`lr 2e-5, per_device_batch 2 x grad_accum 8, num_generations 4, max_completion_length 384, max_steps 200, fp16`, reward = pass/fail on the verifier (no extra reward terms, AGENTS.md rule 11). Evaluation = NB 07 protocol: `mbpp_v2_eval` (100 tasks), n=8, temperature 0.7, top-p 0.9, 384 tokens, seed 1234, adapter merged; OOD/regression Pass@1 on the first 100 synthetic test tasks. Paired bootstrap 95% CIs over tasks.

Inputs via `kernel_sources`: {', '.join('`' + x.split('/')[1] + '`' for x in s['sources'])}.
"""),
        setup_cell(install_wheels=True),
        code(f"""
import json, subprocess, shutil, glob, psutil, pandas as pd

EXP_ID, STEPS, ARM_MAX_HOURS, WALL_CAP_H = "{s['exp']}", 200, 3.0, 3.3
GRPO = {{"lr": 2e-5, "per_device_batch": 2, "grad_accum": 8, "num_generations": 4, "max_completion_length": 384,
        "max_steps": STEPS, "logging_steps": 5, "save_steps": 50}}          # seed is set per arm below
ARMS = {s['arms']}                                                             # (arm, gpu, seed, train_file)
PRIMARY_MODEL = "Qwen/Qwen3.5-2B"
os.makedirs(f"/kaggle/working/{{EXP_ID}}", exist_ok=True)
json.dump({{"grpo": GRPO, "arms": ARMS}}, open(f"/kaggle/working/{{EXP_ID}}/config.json", "w"), indent=2)
print(GRPO, ARMS)
for _, _, _, f in ARMS: assert os.path.exists(f"{{DATA}}/{{f}}"), f
"""),
        md("## 1. Train both arms in parallel (one process per GPU)"),
        code("""
procs = {}
for arm, gpu, seed, train_file in ARMS:
    out = f"/kaggle/working/{EXP_ID}/{arm}"; os.makedirs(out, exist_ok=True)
    cfg = {"experiment_id": f"{EXP_ID}_{arm}", "output_dir": out, "model_id": PRIMARY_MODEL, "init_adapter": None,
           "train_file": f"{DATA}/{train_file}", "grpo": {**GRPO, "seed": seed}, "max_hours": ARM_MAX_HOURS, "guard_step": 5}
    json.dump(cfg, open(f"{out}/run_config.json", "w"), indent=2)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "PYTHONPATH": "/tmp/_pkg", "TOKENIZERS_PARALLELISM": "false", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    log = open(f"{out}/train.log", "w")
    procs[arm] = (subprocess.Popen([sys.executable, "-m", "llm_lab_utils.grpo", f"{out}/run_config.json"], stdout=log, stderr=subprocess.STDOUT, env=env), out)
    print("started", arm, "on GPU", gpu, "seed", seed, train_file)

def last_lines(path, key, n=1):
    try: return [l.rstrip() for l in open(path, errors="ignore") if key in l][-n:]
    except FileNotFoundError: return []
t_start = time.time()
while any(p.poll() is None for p, _ in procs.values()):
    time.sleep(120)
    for arm, (p, out) in procs.items():
        print(f"[{(time.time() - t_start) / 60:6.1f} min] {arm:12s} rc={p.poll()}", *(last_lines(f"{out}/train.log", "[Progress") or ["(no progress yet)"]))
    if (time.time() - t_start) / 3600 > WALL_CAP_H:
        for p, _ in procs.values(): p.kill()
        print("WALL CAP HIT: killed all arms"); break
rcs = {arm: p.returncode for arm, (p, _) in procs.items()}
print("exit codes:", rcs, f"| training wall {(time.time() - t_start) / 3600:.2f} h")
assert all(rc == 0 for rc in rcs.values()), f"an arm failed: {rcs}"
"""),
        md("## 2. Training summary: reward and zero-variance curves"),
        code("""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
metas, hist = {}, {}
for arm, (_, out) in procs.items():
    metas[arm] = json.load(open(f"{out}/checkpoint-final/experiment_meta.json"))
    hist[arm] = json.load(open(f"{out}/trainer_log_history.json"))
rows_tr = []
for arm, m in metas.items():
    h = [e for e in hist[arm] if "reward" in e]; zs = [e["frac_reward_zero_std"] for e in h if "frac_reward_zero_std" in e]
    rows_tr.append({"arm": arm, "steps": m["optimizer_steps"], "gpu_hours": round(m["gpu_hours"], 3), "mean_zero_std_frac": round(sum(zs) / max(len(zs), 1), 3),
                    "reward_first": round(h[0]["reward"], 3), "reward_last": round(h[-1]["reward"], 3), "peak_vram_gb": round(m["peak_vram_gb"], 2),
                    "parent": m.get("parent_checkpoint")})
print(pd.DataFrame(rows_tr).to_string(index=False))
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for arm, h in hist.items():
    pts = [(e["step"], e["reward"]) for e in h if "reward" in e]; zs = [(e["step"], e["frac_reward_zero_std"]) for e in h if "frac_reward_zero_std" in e]
    if pts: ax[0].plot(*zip(*pts), label=arm)
    if zs: ax[1].plot(*zip(*zs), label=arm)
ax[0].set_title("mean reward (train)"); ax[1].set_title("fraction of groups with zero reward std")
for a in ax: a.set_xlabel("step"); a.legend()
plt.tight_layout(); plt.savefig(f"/kaggle/working/{EXP_ID}/reward_curves.png", dpi=120)
for _, (_, out) in procs.items():                     # keep Output small (only checkpoint-final is needed)
    for d in glob.glob(f"{out}/checkpoint-[0-9]*"): shutil.rmtree(d, ignore_errors=True)
"""),
        md("## 3. Evaluate every trained adapter (NB 07 protocol)"),
        code(f"""
import gc, torch
from llm_lab_utils import SubprocessVerifier, ExperimentTracker, load_dataset, paired_bootstrap_diff
from llm_lab_utils.model_utils import setup_model_and_tokenizer
from llm_lab_utils.evaluator import run_evaluation

def one(pattern):
    hits = glob.glob(pattern, recursive=True); assert hits, f"not found: {{pattern}}"; return hits[0]
def counts_from(pattern):
    c = {{}}
    for line in open(one(pattern)):
        g = json.loads(line); c[g["task_id"]] = c.get(g["task_id"], 0) + int(g["verification"]["passed"])
    assert len(c) == 100; return c
{s['refs']}
EVAL_CFG = {{"max_new_tokens": 384, "temperature": 0.7, "top_p": 0.9, "do_sample": True}}      # = NB 03 / NB 07
K_VALUES, EVAL_SEED = [1, 4, 8], 1234
verifier = SubprocessVerifier(timeout=5.0, max_memory_mb=512)
eval_tasks = load_dataset(f"{{DATA}}/mbpp_v2_eval.jsonl")
synth = load_dataset(f"{{DATA}}/test.jsonl")[:100]
RUNS, COUNTS = {{}}, {{}}
for arm, (_, out) in procs.items():
    ckpt = f"{{out}}/checkpoint-final"
    model, tok = setup_model_and_tokenizer(PRIMARY_MODEL, dtype="float16", device_map={{"": 0}}, adapter_path=[ckpt], merge=True)
    model.eval()
    for suffix, tasks_, ks, ds in [("mbpp", eval_tasks, K_VALUES, "mbpp_v2_eval.jsonl"), ("synth", synth, [1, 4], "test.jsonl")]:
        exp_id = f"{{EXP_ID}}_{{arm}}_{{suffix}}"
        t0 = time.time(); torch.cuda.reset_peak_memory_stats(0)
        tracker = ExperimentTracker(exp_id, "/kaggle/working")
        tracker.set_config(model=f"{{PRIMARY_MODEL}} + {{arm}}", dataset=ds, hyperparameters={{**EVAL_CFG, "k_values": ks, "seed": EVAL_SEED, "merged": True,
                           "parent_checkpoint": ckpt, "n_tasks": len(tasks_)}})
        metrics, task_res = run_evaluation(model, tok, tasks_, verifier, EVAL_CFG, ks, tracker=tracker, seed=EVAL_SEED)
        wall_h = (time.time() - t0) / 3600
        tracker.data["training"].update(wall_hours=round(wall_h, 4), gpu_hours=round(wall_h, 4)); tracker.save()
        RUNS[f"{{arm}}_{{suffix}}"] = {{**metrics, "peak_vram_gb": tracker.data["hardware"]["peak_vram_gb"], "wall_hours": wall_h}}
        if suffix == "mbpp": COUNTS[arm] = {{r["task_id"]: r["c"] for r in task_res}}
        print(exp_id, {{k: round(v, 4) for k, v in RUNS[f"{{arm}}_{{suffix}}"].items()}})
    del model, tok; gc.collect(); torch.cuda.empty_cache()
"""),
        md("## 4. Paired comparisons"),
        code(s["compare"]),
        md("## 5. Save"),
        code("""
tracker = ExperimentTracker(EXP_ID + "_summary", "/kaggle/working")
tracker.set_config(model=PRIMARY_MODEL, dataset="mbpp_v2_train variants / mbpp_v2_eval.jsonl", hyperparameters={**GRPO, "arms": ARMS, "eval": EVAL_CFG, "eval_seed": EVAL_SEED})
wall_h = (time.time() - T0) / 3600
tracker.update_metrics({"training": rows_tr, "runs": RUNS, "comparisons": rows, **EXTRA,
                        "ood_pass_at_1": {a: RUNS[f"{a}_synth"]["pass_at_1"] for a, _, _, _ in ARMS}})
tracker.update_hardware(max(r["peak_vram_gb"] for r in RUNS.values()), psutil.virtual_memory().used / 1e9)
tracker.data["training"].update(wall_hours=round(wall_h, 4), gpu_hours=round(wall_h, 4)); tracker.save()
print(f"saved. session wall = billed GPU hours: {wall_h:.2f}")
"""),
        md(f"""
## Takeaways (fill in)
{s['takeaways']}
"""),
    ]
    write_kernel(f"kaggle/kernels/{s['name']}", s["nb"], f"ycheng22/{s['name']}", s["title"], cells,
                 gpu=True, internet=True, kernel_sources=s["sources"])


build("x05")
build("x06")
