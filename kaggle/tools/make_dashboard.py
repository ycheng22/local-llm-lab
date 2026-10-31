"""
Phase 7 / Step 20: research dashboard (CPU only, 0 GPU hours). Reads the pulled Kaggle outputs under
experiments/kaggle_outputs/ and writes reports/final_results.{csv,md,json} and reports/figures_final/*.png.
Run from the repo root:  python kaggle/tools/make_dashboard.py
"""
import csv
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "experiments/kaggle_outputs"
OUT = "reports"
FIG = f"{OUT}/figures_final"
os.makedirs(FIG, exist_ok=True)
C_BASE, C_GRPO, C_OTHER, C_4B = "#4c78a8", "#e8833a", "#8c8c8c", "#59a14f"


def summ(pattern):
    hits = glob.glob(f"{R}/{pattern}", recursive=True)
    assert hits, f"missing {pattern}"
    return json.load(open(hits[0], encoding="utf-8"))


def row(label, d, extra=None):
    m, t, h = d.get("metrics", {}), d.get("training", {}), d.get("hardware", {})
    r = {"experiment": label, "experiment_id": d.get("experiment_id"), "pass_at_1": m.get("pass_at_1"), "pass_at_4": m.get("pass_at_4"),
         "pass_at_8": m.get("pass_at_8"), "peak_vram_gb": h.get("peak_vram_gb"), "peak_ram_gb": h.get("peak_ram_gb"),
         "wall_hours": t.get("wall_hours"), "gpu_hours": t.get("gpu_hours")}
    r.update(extra or {})
    return r


# ------------------------------------------------------------------ MBPP held-out table (100 tasks, n=8, 384 tokens)
base = summ("03-*/exp00_baseline_mbpp_v2/summary.json")
sft = summ("05-*/exp01_sft_mbpp_eval/summary.json")
g42 = summ("07/exp02_grpo_from_base_mbpp_eval/summary.json")
g_sft = summ("07/exp02_grpo_from_sft_mbpp_eval/summary.json")
g43 = summ("x05/x05_grpo_seeds_seed43_mbpp/summary.json")
g44 = summ("x05/x05_grpo_seeds_seed44_mbpp/summary.json")
l97 = summ("x06/x06_grpo_mining_learnable97_mbpp/summary.json")
r97 = summ("x06/x06_grpo_mining_random97_mbpp/summary.json")
b4 = summ("x03/x03_4b_baseline_mbpp/summary.json")
# OOD / regression Pass@1 on the synthetic test slice
ood = {"base": summ("03-*/exp00_baseline_synth/summary.json")["metrics"]["pass_at_1"], "sft": summ("05-*/exp01_sft_synth_eval/summary.json")["metrics"]["pass_at_1"],
       "grpo_s42": summ("07/exp02_grpo_from_base_synth_eval/summary.json")["metrics"]["pass_at_1"],
       "grpo_from_sft": summ("07/exp02_grpo_from_sft_synth_eval/summary.json")["metrics"]["pass_at_1"],
       "grpo_s43": summ("x05/x05_grpo_seeds_seed43_synth/summary.json")["metrics"]["pass_at_1"],
       "grpo_s44": summ("x05/x05_grpo_seeds_seed44_synth/summary.json")["metrics"]["pass_at_1"],
       "learnable97": summ("x06/x06_grpo_mining_learnable97_synth/summary.json")["metrics"]["pass_at_1"],
       "random97": summ("x06/x06_grpo_mining_random97_synth/summary.json")["metrics"]["pass_at_1"],
       "4b_base": summ("x03/x03_4b_baseline_synth/summary.json")["metrics"]["pass_at_1"]}
# training GPU hours (T4 hours; the two arms of one session run in parallel, so a session is billed once)
TRAIN_H = {"sft": 0.16, "grpo_s42": 2.34, "grpo_s43": json.load(open(f"{R}/x05/x05_grpo_seeds_summary/summary.json"))["metrics"]["training"][0]["gpu_hours"],
           "grpo_s44": json.load(open(f"{R}/x05/x05_grpo_seeds_summary/summary.json"))["metrics"]["training"][1]["gpu_hours"],
           "learnable97": json.load(open(f"{R}/x06/x06_grpo_mining_summary/summary.json"))["metrics"]["training"][0]["gpu_hours"],
           "random97": json.load(open(f"{R}/x06/x06_grpo_mining_summary/summary.json"))["metrics"]["training"][1]["gpu_hours"]}

MODELS = [("2B base", base, "base", C_BASE), ("2B SFT", sft, "sft", C_OTHER), ("2B GRPO from SFT", g_sft, "grpo_from_sft", C_OTHER),
          ("2B GRPO seed 42", g42, "grpo_s42", C_GRPO), ("2B GRPO seed 43", g43, "grpo_s43", C_GRPO), ("2B GRPO seed 44", g44, "grpo_s44", C_GRPO),
          ("2B GRPO 97 learnable", l97, "learnable97", C_GRPO), ("2B GRPO 97 random", r97, "random97", C_GRPO), ("4B base (384 cap)", b4, "4b_base", C_4B)]
rows = []
for label, d, key, _ in MODELS:
    rows.append(row(label, d, {"ood_pass_at_1": ood[key], "train_gpu_hours": TRAIN_H.get(key)}))
p1 = {key: d["metrics"]["pass_at_1"] for _, d, key, _ in MODELS}
seeds = [p1["grpo_s42"], p1["grpo_s43"], p1["grpo_s44"]]
seed_mean = sum(seeds) / 3
seed_std = (sum((x - seed_mean) ** 2 for x in seeds) / 2) ** 0.5

# ------------------------------------------------------------------ extra experiments
hum = {"base": summ("x02/x02_humaneval_base/summary.json")["metrics"], "grpo": summ("x02/x02_humaneval_grpo_from_base/summary.json")["metrics"]}
ren = {"base": summ("x07/x07_renamed_base/summary.json")["metrics"], "grpo": summ("x07/x07_renamed_grpo_from_base/summary.json")["metrics"]}
cap = {"base": summ("x04/x04_cap768_base/summary.json")["metrics"], "grpo": summ("x04/x04_cap768_grpo_from_base/summary.json")["metrics"]}
sc_b, sc_g = summ("09b/exp04_scaling_base/summary.json")["metrics"], summ("09/exp04_scaling_grpo_from_base/summary.json")["metrics"]
ag2, ag4 = summ("10/exp05_agent_2b_summary/summary.json")["metrics"], summ("11/exp05_agent_4b_summary/summary.json")["metrics"]

# ------------------------------------------------------------------ tables
cols = ["experiment", "pass_at_1", "pass_at_4", "pass_at_8", "ood_pass_at_1", "peak_vram_gb", "peak_ram_gb", "wall_hours", "gpu_hours", "train_gpu_hours"]
with open(f"{OUT}/final_results.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)


def pct(x): return "-" if x is None else f"{100 * x:.1f}%"
def num(x, n=2): return "-" if x is None else f"{x:.{n}f}"
md = ["| Model (MBPP held-out, 100 tasks, n=8, 384 tokens) | Pass@1 | Pass@4 | Pass@8 | OOD Pass@1 | Peak VRAM GB | Peak RAM GB | Eval wall h | Train GPU h |",
      "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
for r in rows:
    md.append(f"| {r['experiment']} | {pct(r['pass_at_1'])} | {pct(r['pass_at_4'])} | {pct(r['pass_at_8'])} | {pct(r['ood_pass_at_1'])} | {num(r['peak_vram_gb'], 1)} | "
              f"{num(r['peak_ram_gb'], 1)} | {num(r['wall_hours'])} | {num(r['train_gpu_hours'])} |")
GAIN = 100 * (seed_mean - p1["base"]) + 1e-9
md.append(f"\nGRPO from base over 3 seeds: Pass@1 {100 * seed_mean:.1f}% +/- {100 * seed_std:.1f} pt (sample std); gain over base {GAIN:+.1f} pt.")
open(f"{OUT}/final_results.md", "w", encoding="utf-8").write("\n".join(md) + "\n")
json.dump({"rows": rows, "seed_mean_pass_at_1": seed_mean, "seed_std_pass_at_1": seed_std, "humaneval": hum, "renamed": ren, "cap768": cap}, open(f"{OUT}/final_results.json", "w"), indent=1)
print("\n".join(md))

# ------------------------------------------------------------------ figures
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 130})


def save(name):
    plt.tight_layout(); plt.savefig(f"{FIG}/{name}.png"); plt.close()


# 1. Pass@1 of every model
fig, ax = plt.subplots(figsize=(9, 4))
labels = [m[0] for m in MODELS]; vals = [100 * p1[m[2]] for m in MODELS]
ax.bar(range(len(MODELS)), vals, color=[m[3] for m in MODELS])
for i, v in enumerate(vals): ax.text(i, v + 0.8, f"{v:.1f}", ha="center", fontsize=8)
ax.axhline(100 * p1["base"], color=C_BASE, ls="--", lw=1)
ax.set_xticks(range(len(MODELS))); ax.set_xticklabels(labels, rotation=30, ha="right"); ax.set_ylabel("Pass@1 (%)"); ax.set_ylim(0, 70)
ax.set_title("MBPP held-out Pass@1 (100 tasks, 8 samples, 384-token cap; run-to-run std of GRPO ~3.3 pt)")
save("f1_pass_at_1_models")

# 2. ROI
roi = [(k, 100 * (p1[k] - p1["base"]) / h) for k, h in TRAIN_H.items()]
fig, ax = plt.subplots(figsize=(7, 3.8))
names = {"sft": "SFT", "grpo_s42": "GRPO s42", "grpo_s43": "GRPO s43", "grpo_s44": "GRPO s44", "learnable97": "GRPO learnable97", "random97": "GRPO random97"}
ax.bar([names[k] for k, _ in roi], [v for _, v in roi], color=[C_OTHER if k == "sft" else C_GRPO for k, _ in roi])
for i, (_, v) in enumerate(roi): ax.text(i, v + (0.15 if v >= 0 else -0.6), f"{v:+.1f}", ha="center", fontsize=8)
ax.set_ylabel("Pass@1 gain over base per training T4-hour (pt/h)"); ax.set_title("Return on training compute"); plt.xticks(rotation=25, ha="right")
save("f2_roi")

# 3. test-time scaling
ks = [1, 2, 4, 8, 16]
fig, ax = plt.subplots(figsize=(6.5, 4))
for lab, sc, c in [("base", sc_b, C_BASE), ("GRPO", sc_g, C_GRPO)]:
    ax.plot(ks, [100 * sc["oracle_pass_at_k"][str(k)] for k in ks], "o--", color=c, alpha=0.6, label=f"{lab}: oracle Pass@k")
    ax.plot(ks, [100 * sc["visible_selection"][str(k)] for k in ks], "s-", color=c, label=f"{lab}: pick by visible test")
ax.set_xscale("log", base=2); ax.set_xticks(ks); ax.set_xticklabels(ks); ax.set_xlabel("samples per task (k)"); ax.set_ylabel("accuracy on held-out MBPP (%)")
ax.set_title("Test-time scaling: one GRPO sample ~ four base samples"); ax.legend(fontsize=8)
save("f3_test_time_scaling")

# 4. transfer: MBPP, MBPP renamed, HumanEval
fig, ax = plt.subplots(figsize=(7, 4)); w = 0.38
sets = [("MBPP", p1["base"], p1["grpo_s42"]), ("MBPP renamed", ren["base"]["pass_at_1"], ren["grpo"]["pass_at_1"]), ("HumanEval (157)", hum["base"]["pass_at_1"], hum["grpo"]["pass_at_1"])]
ax.bar([i - w / 2 for i in range(3)], [100 * s[1] for s in sets], w, color=C_BASE, label="base")
ax.bar([i + w / 2 for i in range(3)], [100 * s[2] for s in sets], w, color=C_GRPO, label="GRPO (seed 42)")
for i, s in enumerate(sets): ax.text(i + w / 2, 100 * s[2] + 1, f"{100 * (s[2] - s[1]):+.1f}", ha="center", fontsize=9)
ax.set_xticks(range(3)); ax.set_xticklabels([s[0] for s in sets]); ax.set_ylabel("Pass@1 (%)"); ax.legend(); ax.set_ylim(0, 75)
ax.set_title("The RL gain: large on MBPP, survives renaming, ~0 on HumanEval")
save("f4_transfer")

# 5. token cap
fig, ax = plt.subplots(figsize=(6, 4)); w = 0.38
cs = [("2B base", p1["base"], cap["base"]["pass_at_1"]), ("2B GRPO", p1["grpo_s42"], cap["grpo"]["pass_at_1"]), ("4B base (73 tasks)", 0.534, 0.685)]
ax.bar([i - w / 2 for i in range(3)], [100 * c[1] for c in cs], w, color=C_OTHER, label="384 tokens")
ax.bar([i + w / 2 for i in range(3)], [100 * c[2] for c in cs], w, color=C_4B, label="768 tokens")
ax.set_xticks(range(3)); ax.set_xticklabels([c[0] for c in cs]); ax.set_ylabel("Pass@1 (%)"); ax.legend(); ax.set_ylim(0, 80)
ax.set_title("Token cap hides 4B capability, not 2B")
save("f5_token_cap")

# 6. agents
def cmp(summary, vs):
    return [c for c in summary["comparisons"] if vs in c.get("vs", "") or vs in c.get("comparison", "")]
a2 = {r["model"]: r for r in ag2["comparisons"] if r.get("vs", "").startswith("single-shot") and r["model"] in ("base", "grpo_from_base")
      and (("base" in r["vs"]) == (r["model"] == "base"))}      # each agent vs ITS OWN single-shot (base vs NB 03, GRPO vs NB 07)
fig, ax = plt.subplots(figsize=(6.5, 4)); w = 0.38
ag_rows = [("2B base", 100 * a2["base"]["reference"], 100 * a2["base"]["agent"]), ("2B GRPO", 100 * a2["grpo_from_base"]["reference"], 100 * a2["grpo_from_base"]["agent"]),
           ("4B base (39 tasks)", 51.3, 100 * ag4["results"]["base"]["task_completion_rate"])]
ax.bar([i - w / 2 for i in range(3)], [r[1] for r in ag_rows], w, color=C_OTHER, label="single-shot")
ax.bar([i + w / 2 for i in range(3)], [r[2] for r in ag_rows], w, color=C_GRPO, label="agent loop (<=5 steps)")
for i, r in enumerate(ag_rows): ax.text(i + w / 2, r[2] + 1, f"{r[2] - r[1]:+.1f}", ha="center", fontsize=9)
ax.set_xticks(range(3)); ax.set_xticklabels([r[0] for r in ag_rows]); ax.set_ylabel("success (%)"); ax.legend(); ax.set_ylim(0, 90)
ax.set_title("Agent loop helps 4B, not 2B")
save("f6_agent")

# 7. mining ablation
fig, ax = plt.subplots(1, 2, figsize=(8, 3.6))
mine = [("base", 100 * p1["base"], None), ("learnable97", 100 * p1["learnable97"], 40.3), ("random97", 100 * p1["random97"], 61.3), ("full 256", 100 * p1["grpo_s42"], 61.0)]
ax[0].bar([m[0] for m in mine], [m[1] for m in mine], color=[C_BASE, C_GRPO, C_GRPO, C_GRPO]); ax[0].set_ylabel("Pass@1 (%)"); ax[0].set_title("accuracy")
ax[1].bar([m[0] for m in mine[1:]], [m[2] for m in mine[1:]], color=C_GRPO); ax[1].set_ylabel("% groups with zero reward std"); ax[1].set_title("wasted training signal")
for a in ax: a.tick_params(axis="x", rotation=20)
save("f7_mining")

# 8. seeds
fig, ax = plt.subplots(figsize=(5.5, 3.8))
ax.scatter([42, 43, 44], [100 * x for x in seeds], color=C_GRPO, s=60, zorder=3)
ax.axhline(100 * p1["base"], color=C_BASE, ls="--", label="base"); ax.axhline(100 * seed_mean, color=C_GRPO, ls=":", label=f"mean {100 * seed_mean:.1f} (std {100 * seed_std:.1f})")
ax.set_xticks([42, 43, 44]); ax.set_xlabel("training seed"); ax.set_ylabel("Pass@1 (%)"); ax.set_ylim(35, 65); ax.legend(); ax.set_title("GRPO gain across seeds")
save("f8_seeds")
print("figures:", sorted(os.listdir(FIG)))
