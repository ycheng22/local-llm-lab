"""
Generates kaggle/kernels/06s-grpo-smoke (calibration) and 06-grpo-training (full).
Run from the repo root:  python kaggle/tools/build_nb06.py
"""
import sys
sys.path.insert(0, "kaggle/tools")
from nbbuild import md, code, setup_cell, write_kernel


def build(smoke: bool):
    steps = 6 if smoke else 200
    name = "06s-grpo-smoke" if smoke else "06-grpo-training"
    title = "06s GRPO Smoke" if smoke else "06 GRPO Training"
    exp = "exp02s_grpo_smoke" if smoke else "exp02_grpo_mbpp_v1"
    hours = 0.5 if smoke else 4.0

    cells = [
        md(f"""
# {'06s - GRPO smoke test (calibration)' if smoke else '06 - GRPO Training (Phase 3, redesigned)'}

**Mode:** GPU (T4 x2), background commit. {'Runs **6 steps per arm** to measure real seconds/step before committing hours. ~15 min.' if smoke else 'Hard cap 5 h. Two arms train **in parallel**, one per GPU, so the session costs the wall time of ONE run.'}

## What is GRPO?
For each prompt, sample a *group* of G=4 completions, score each with the verifier (**reward = 1 if all tests pass, else 0**, nothing else - AGENTS.md rule 11), and push up the probability of completions that scored above the group average. No value network and no preference data: the unit tests are the teacher. Learning needs groups with *mixed* outcomes - our base model has Pass@1 40.6% and Pass@8 69% on MBPP, so many groups are mixed.

## Two arms (same hyperparameters, different start)
| Arm | GPU | Start | Question |
|---|---|---|---|
| `grpo_from_sft` | 0 | base + merged SFT adapter (NB 04) | does RL add to SFT? |
| `grpo_from_base` | 1 | Qwen3.5-2B | does RL alone work? |

## Pre-registered settings (not tuned on the eval set)
| | Local (8 GB) | Here | Why |
|---|---|---|---|
| Weights | 4-bit NF4 | fp16 | NB 01b: 2.5-3.5x faster |
| Prompts / step | 1 | **4** (per-device 2 x accum 8 = 16 completions; per-device 8 ran out of memory in the smoke test: the logits tensor is batch x length x 248k vocab) | batched generation is far cheaper per sample |
| Generations per prompt | 4 | 4 | same |
| Learning rate | 1e-5 (3250 steps) | **2e-5 ({steps} steps)** | far fewer steps |
| Train tasks | `train_5k` synthetic | `mbpp_v2_train` (256 MBPP tasks) | synthetic set is saturated |
| Reward | pass/fail | pass/fail (verifier timeout 3 s) | same |
"""),
        setup_cell(install_wheels=True),
        code(f"""
import json, subprocess, shutil, psutil, pandas as pd
from llm_lab_utils import ExperimentTracker

SFT_ADAPTER = glob.glob(f"{{find_input('04-sft-training')}}/**/exp01_sft_mbpp_v1/checkpoint-final", recursive=True)[0]
EXP_ID, STEPS, ARM_MAX_HOURS, WALL_CAP_H = "{exp}", {steps}, {hours}, {hours + 0.5}
GRPO = {{"lr": 2e-5, "per_device_batch": 2, "grad_accum": 8, "num_generations": 4, "max_completion_length": 384,
        "max_steps": STEPS, "logging_steps": 5, "save_steps": 50, "seed": 42}}
ARMS = [("grpo_from_sft", "0", SFT_ADAPTER), ("grpo_from_base", "1", None)]
os.makedirs(f"/kaggle/working/{{EXP_ID}}", exist_ok=True)
json.dump({{"grpo": GRPO, "arms": [(a, g, i) for a, g, i in ARMS]}}, open(f"/kaggle/working/{{EXP_ID}}/config.json", "w"), indent=2)
print(GRPO, "\\nSFT adapter:", SFT_ADAPTER)
"""),
        md("## 1. Launch both arms (one process per GPU)"),
        code("""
procs = {}
for arm, gpu, init in ARMS:
    out = f"/kaggle/working/{EXP_ID}/{arm}"; os.makedirs(out, exist_ok=True)
    cfg = {"experiment_id": f"{EXP_ID}_{arm}", "output_dir": out, "model_id": PRIMARY_MODEL_ID if 'PRIMARY_MODEL_ID' in globals() else "Qwen/Qwen3.5-2B",
           "init_adapter": init, "train_file": f"{DATA}/mbpp_v2_train.jsonl", "grpo": GRPO,
           "max_hours": ARM_MAX_HOURS, "guard_step": 5}
    json.dump(cfg, open(f"{out}/run_config.json", "w"), indent=2)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "PYTHONPATH": "/tmp/_pkg", "TOKENIZERS_PARALLELISM": "false", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    log = open(f"{out}/train.log", "w")
    procs[arm] = (subprocess.Popen([sys.executable, "-m", "llm_lab_utils.grpo", f"{out}/run_config.json"],
                                   stdout=log, stderr=subprocess.STDOUT, env=env), out)
    print("started", arm, "on GPU", gpu)
"""),
        md("## 2. Monitor (prints the latest progress line of each arm every 2 minutes; kills everything past the wall cap)"),
        code("""
def last_lines(path, key, n=1):
    try: return [l.rstrip() for l in open(path, errors="ignore") if key in l][-n:]
    except FileNotFoundError: return []

t_start = time.time()
while any(p.poll() is None for p, _ in procs.values()):
    time.sleep(120)
    for arm, (p, out) in procs.items():
        print(f"[{(time.time() - t_start) / 60:6.1f} min] {arm:15s} rc={p.poll()}", *(last_lines(f"{out}/train.log", "[Progress") or ["(no progress yet)"]))
    if (time.time() - t_start) / 3600 > WALL_CAP_H:
        for p, _ in procs.values(): p.kill()
        print("WALL CAP HIT: killed all arms"); break
rcs = {arm: p.returncode for arm, (p, _) in procs.items()}
print("exit codes:", rcs)
for arm, (_, out) in procs.items():
    print(f"\\n=== {arm} log tail ==="); print("".join(open(f"{out}/train.log", errors="ignore").readlines()[-12:]))
"""),
        md("## 3. Results: reward curves, cost, speed"),
        code("""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
metas, hist = {}, {}
for arm, (_, out) in procs.items():
    mp = f"{out}/checkpoint-final/experiment_meta.json"
    if os.path.exists(mp): metas[arm] = json.load(open(mp))
    hp = f"{out}/trainer_log_history.json"
    if os.path.exists(hp): hist[arm] = json.load(open(hp))

rows = []
for arm, m in metas.items():
    h = [e for e in hist.get(arm, []) if "reward" in e]
    nan_steps = sum(1 for e in hist.get(arm, []) if "grad_norm" in e and e["grad_norm"] != e["grad_norm"])   # fp16 scaler skips
    zero_std = [e["frac_reward_zero_std"] for e in h if "frac_reward_zero_std" in e]
    rows.append({"arm": arm, "steps": m["optimizer_steps"], "nan_grad_logs": nan_steps,
                 "mean_zero_std_frac": round(sum(zero_std) / len(zero_std), 3) if zero_std else None, "gpu_hours": round(m["gpu_hours"], 3),
                 "s_per_step": round(m["gpu_hours"] * 3600 / max(m["optimizer_steps"], 1), 1),
                 "peak_vram_gb": round(m["peak_vram_gb"], 2), "stopped_by_budget": m["stopped_by_budget"],
                 "reward_first": round(h[0]["reward"], 3) if h else None, "reward_last": round(h[-1]["reward"], 3) if h else None})
summary = pd.DataFrame(rows); print(summary.to_string(index=False))
if rows:
    full_steps = 200
    print(f"\\nProjected full run ({full_steps} steps, arms in parallel): ",
          {r['arm']: round(r['s_per_step'] * full_steps / 3600, 2) for r in rows}, "hours")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for arm, h in hist.items():
    pts = [(e["step"], e["reward"]) for e in h if "reward" in e]
    zs = [(e["step"], e["frac_reward_zero_std"]) for e in h if "frac_reward_zero_std" in e]
    if pts: ax[0].plot(*zip(*pts), label=arm)
    if zs: ax[1].plot(*zip(*zs), label=arm)
ax[0].set_title("mean reward (train)"); ax[1].set_title("fraction of groups with zero reward std (no learning signal)")
for a in ax: a.set_xlabel("step"); a.legend()
plt.tight_layout(); plt.savefig(f"/kaggle/working/{EXP_ID}/reward_curves.png", dpi=120)
"""),
        md("## 4. Save summary"),
        code("""
tracker = ExperimentTracker(EXP_ID + "_summary", "/kaggle/working")
tracker.set_config(model="Qwen/Qwen3.5-2B", dataset="mbpp_v2_train.jsonl", hyperparameters={**GRPO, "arms": [a for a, _, _ in ARMS]})
wall_h = (time.time() - T0) / 3600
tracker.update_metrics({"arms": rows, "exit_codes": rcs})
tracker.update_hardware(max([m["peak_vram_gb"] for m in metas.values()] or [0]), psutil.virtual_memory().used / 1e9)
tracker.data["training"].update(wall_hours=round(wall_h, 4), gpu_hours=round(wall_h, 4))   # one session = billed once for both GPUs
tracker.save()
for arm, (_, out) in procs.items():                     # keep Output small
    for d in glob.glob(f"{out}/checkpoint-[0-9]*"): shutil.rmtree(d, ignore_errors=True)
print(f"session wall = billed GPU hours: {wall_h:.2f}")
assert all(rc == 0 for rc in rcs.values()), f"an arm failed: {rcs} (see the log tails above)"
"""),
        md(f"""
## Takeaways
{'- Read `s_per_step` and the projected full-run hours; if the projection is within budget, push `06-grpo-training`. Otherwise reduce `STEPS`/`max_completion_length`.' if smoke else '- Did reward rise in both arms? Did the zero-std fraction grow (collapse of learning signal)?'}
{'- Check that rewards are not all 0.0 (extraction/format problem) or all 1.0.' if smoke else '- Next: `07_grpo_evaluation` compares both arms with the SFT model and the baseline on the 100 held-out tasks.'}
"""),
    ]
    write_kernel(f"kaggle/kernels/{name}", f"{name.replace('-', '_')}.ipynb", f"ycheng22/{name}", title, cells,
                 gpu=True, internet=True, kernel_sources=["ycheng22/04-sft-training"])


build(smoke=True)
build(smoke=False)
