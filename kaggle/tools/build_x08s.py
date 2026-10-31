"""Generates kaggle/kernels/x08s-4b-grpo-smoke (plan Step 19i calibration). Run from the repo root: python kaggle/tools/build_x08s.py"""
import sys
sys.path.insert(0, "kaggle/tools")
from nbbuild import md, code, setup_cell, write_kernel

cells = [
    md("""
# x08s - Qwen3.5-4B GRPO smoke / calibration (Phase 6.5, Step 19i, stretch)

**Mode:** GPU (T4 x2), background commit, 1 h hard cap. Estimated ~0.3 GPU hours.

## Question (go / no-go for a 4B GRPO run)
Does 4B GRPO fit in a 16 GB T4, how many seconds per step does it take, and does the full 200-step recipe fit the pre-registered limit (**<= 5 GPU hours**)?

## Design
Two configurations are tried **in parallel, one per GPU**, 6 optimizer steps each, `grpo_from_base` recipe of NB 06 with the 4B model:

| Arm | GPU | per-device batch x grad-accum | completion cap |
|---|---|---|---|
| `bs1_cap768` | 0 | 1 x 16 | 768 |
| `bs2_cap768` | 1 | 2 x 8 | 768 |

Both give 16 completions per optimizer step (4 prompts x 4 generations), exactly like NB 06. The cap is **768** (not 384) because x03/x04 showed that 4B writes a long reasoning preamble and 90% of its samples are truncated at 384 (x04: 4B Pass@1 +15 pt at 768). One arm may run out of memory; that is a valid result.

Decision rule (fixed now): launch the full run only if (a) projected 200-step time of the cheaper non-OOM arm <= 5 h, and (b) mean reward over the 6 steps is neither 0 nor 1. Otherwise this notebook is the documented reason why 4B GRPO was not run.
"""),
    setup_cell(install_wheels=True),
    code("""
import json, subprocess, glob, psutil, pandas as pd
EXP_ID, STEPS, ARM_MAX_HOURS, WALL_CAP_H = "x08s_4b_grpo_smoke", 6, 0.4, 0.6
BASE = {"lr": 2e-5, "num_generations": 4, "max_completion_length": 768, "max_steps": STEPS, "logging_steps": 1, "save_steps": 1000, "seed": 42}
ARMS = [("bs1_cap768", "0", 1, 16), ("bs2_cap768", "1", 2, 8)]
MODEL = "Qwen/Qwen3.5-4B"
os.makedirs(f"/kaggle/working/{EXP_ID}", exist_ok=True)
procs = {}
for arm, gpu, bs, acc in ARMS:
    out = f"/kaggle/working/{EXP_ID}/{arm}"; os.makedirs(out, exist_ok=True)
    cfg = {"experiment_id": f"{EXP_ID}_{arm}", "output_dir": out, "model_id": MODEL, "init_adapter": None, "train_file": f"{DATA}/mbpp_v2_train.jsonl",
           "grpo": {**BASE, "per_device_batch": bs, "grad_accum": acc}, "max_hours": ARM_MAX_HOURS, "guard_step": 3}
    json.dump(cfg, open(f"{out}/run_config.json", "w"), indent=2)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "PYTHONPATH": "/tmp/_pkg", "TOKENIZERS_PARALLELISM": "false", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    procs[arm] = (subprocess.Popen([sys.executable, "-m", "llm_lab_utils.grpo", f"{out}/run_config.json"], stdout=open(f"{out}/train.log", "w"), stderr=subprocess.STDOUT, env=env), out)
    print("started", arm, "GPU", gpu, "batch", bs, "x", acc)
t_start = time.time()
while any(p.poll() is None for p, _ in procs.values()):
    time.sleep(60)
    for arm, (p, out) in procs.items():
        tail = [l.rstrip() for l in open(f"{out}/train.log", errors="ignore") if "Progress" in l or "Error" in l][-1:]
        print(f"[{(time.time() - t_start) / 60:5.1f} min] {arm:12s} rc={p.poll()}", *tail)
    if (time.time() - t_start) / 3600 > WALL_CAP_H:
        for p, _ in procs.values(): p.kill()
        print("WALL CAP HIT"); break
print("exit codes:", {a: p.returncode for a, (p, _) in procs.items()})
"""),
    md("## Result: fit, speed, reward"),
    code("""
rows = []
for arm, (p, out) in procs.items():
    log = open(f"{out}/train.log", errors="ignore").read()
    oom = "out of memory" in log.lower()
    mp = f"{out}/checkpoint-final/experiment_meta.json"; hp = f"{out}/trainer_log_history.json"
    row = {"arm": arm, "rc": p.returncode, "oom": oom}
    if os.path.exists(mp):
        m = json.load(open(mp)); h = [e for e in json.load(open(hp)) if "reward" in e]
        spp = m["gpu_hours"] * 3600 / max(m["optimizer_steps"], 1)
        row.update({"steps": m["optimizer_steps"], "s_per_step": round(spp, 1), "projected_200_step_hours": round(spp * 200 / 3600, 2),
                    "peak_vram_gb": round(m["peak_vram_gb"], 2), "mean_reward": round(sum(e["reward"] for e in h) / max(len(h), 1), 3),
                    "mean_completion_len": round(sum(e.get("completions/mean_length", 0) for e in h) / max(len(h), 1), 1)})
    rows.append(row)
df = pd.DataFrame(rows); print(df.to_string(index=False))
for arm, (_, out) in procs.items():
    print(f"\\n=== {arm} log tail ==="); print("".join(open(f"{out}/train.log", errors="ignore").readlines()[-8:]))
ok = [r for r in rows if r.get("projected_200_step_hours") is not None]
best = min(ok, key=lambda r: r["projected_200_step_hours"]) if ok else None
GO = bool(best and best["projected_200_step_hours"] <= 5.0 and 0 < best["mean_reward"] < 1)
print("\\nGO for full 4B GRPO:", GO, "| best arm:", best)
json.dump({"rows": rows, "go": GO, "best": best, "model": MODEL, "wall_hours": (time.time() - T0) / 3600}, open(f"/kaggle/working/{EXP_ID}/result.json", "w"), indent=1)
print(f"session wall = billed GPU hours: {(time.time() - T0) / 3600:.2f}")
"""),
    md("""
## Takeaways (fill in)
- Fit / OOM per configuration; seconds per step; projected 200-step hours; reward not degenerate; GO or documented NO-GO.
"""),
]

write_kernel("kaggle/kernels/x08s-4b-grpo-smoke", "x08s_4b_grpo_smoke.ipynb", "ycheng22/x08s-4b-grpo-smoke", "x08s 4B GRPO Smoke", cells,
             gpu=True, internet=True, kernel_sources=[])
