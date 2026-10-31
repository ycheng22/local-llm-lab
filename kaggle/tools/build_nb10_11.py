"""
Generates kaggle/kernels/10-agent-evaluation-2b and 11-agent-evaluation-4b.
Run from the repo root: python kaggle/tools/build_nb10_11.py

Both notebooks run the same ReAct agent (read_file / write_file / run_tests / submit, max 5 steps) on the same
first N_TASKS held-out MBPP tasks and compare it with single-shot generation on those tasks.
"""
import sys
sys.path.insert(0, "kaggle/tools")
from nbbuild import md, code, setup_cell, write_kernel

CONFIG = '''
MODEL_ID, TAG, VARIANTS = "@MODEL_ID@", "@TAG@", @VARIANTS@
N_TASKS, MAX_STEPS, SEED, BUDGET_S = 60, 5, 1234, 2400     # budget: stop starting new tasks after 40 min per variant
'''

LOAD = '''
import json, gc, torch, psutil, pandas as pd
from llm_lab_utils import (PRIMARY_MODEL_ID, ExperimentTracker, load_dataset, paired_bootstrap_diff)
from llm_lab_utils.model_utils import setup_model_and_tokenizer
from llm_lab_utils.agent_loop import evaluate_agent, extract_action

def mount(name): return find_input(name)
def one(pattern):
    hits = glob.glob(pattern, recursive=True); assert hits, f"not found: {pattern}"; return hits[0]
def counts_from(pattern):
    c = {}
    for line in open(one(pattern)):
        g = json.loads(line); c[g["task_id"]] = c.get(g["task_id"], 0) + int(g["verification"]["passed"])
    return c

tasks = load_dataset(f"{DATA}/mbpp_v2_eval.jsonl")[:N_TASKS]          # fixed order -> same tasks in NB 10 and NB 11
ids = [t.id for t in tasks]
REF = {"single-shot base (NB 03)": counts_from(f"{mount('03-baseline-evaluation')}/**/exp00_baseline_mbpp_v2/generations.jsonl")}
@REF_EXTRA@
REF = {k: {i: v[i] for i in ids} for k, v in REF.items()}              # restrict to the tasks the agent sees
print({k: round(sum(v.values()) / (8 * len(ids)), 3) for k, v in REF.items()}, f"<- single-shot Pass@1 on these {len(ids)} tasks")
'''

RUN = '''
RESULTS, TRACES, COUNTS = {}, {}, {}
def adapters_for(name):
    if name == "base": return None
    if name == "grpo_from_base": return [one(f"{mount('06-grpo-training')}/**/grpo_from_base/checkpoint-final")]
    raise ValueError(name)

for name in VARIANTS:
    model, tok = setup_model_and_tokenizer(MODEL_ID, dtype="float16", device_map={"": 0}, adapter_path=adapters_for(name), merge=True)
    model.eval()
    exp_id = f"exp05_agent_{TAG}_{name}"
    t0 = time.time(); torch.cuda.reset_peak_memory_stats(0)
    tracker = ExperimentTracker(exp_id, "/kaggle/working")
    tracker.set_config(model=f"{MODEL_ID} ({name})", dataset="mbpp_v2_eval.jsonl (first %d tasks)" % N_TASKS,
                       hyperparameters={"max_steps": MAX_STEPS, "max_new_tokens": 512, "temperature": 0.2, "seed": SEED,
                                        "dtype": "float16", "merged": True, "feedback_tests": "visible only",
                                        "grading": "visible + hidden", "budget_s": BUDGET_S})
    metrics, traces = evaluate_agent(model, tok, tasks, max_steps=MAX_STEPS, time_budget_s=BUDGET_S, seed=SEED)
    wall_h = (time.time() - t0) / 3600
    tracker.update_metrics(metrics)
    tracker.update_hardware(torch.cuda.max_memory_allocated(0) / 1e9, psutil.virtual_memory().used / 1e9)
    tracker.data["training"].update(wall_hours=round(wall_h, 4), gpu_hours=round(wall_h, 4)); tracker.save()
    with open(f"/kaggle/working/{exp_id}/agent_traces.jsonl", "w") as f:
        for t in traces: f.write(json.dumps(t) + "\\n")
    RESULTS[name] = {**metrics, "wall_hours": wall_h, "peak_vram_gb": torch.cuda.max_memory_allocated(0) / 1e9}
    TRACES[name] = traces
    COUNTS[name] = {t_id: 8 * int(tr["passed"]) for t_id, tr in zip(ids, traces)}     # x8 so it plugs into the Pass@1 bootstrap
    print(exp_id, {k: round(v, 3) for k, v in RESULTS[name].items()})
    del model, tok; gc.collect(); torch.cuda.empty_cache()
'''

COMPARE = '''
rows = []
for name, c in COUNTS.items():
    done = list(c)                                              # tasks actually run (a budget stop can truncate)
    for ref_name, ref in REF.items():
        r = paired_bootstrap_diff({i: ref[i] for i in done}, c, n=8, k=1)
        rows.append({"model": name, "vs": ref_name, "n_tasks": len(done), "reference": round(r["mean_a"], 3), "agent": round(r["mean_b"], 3),
                     "diff": round(r["diff"], 3), "95% CI": f"[{r['ci95_low']:+.3f}, {r['ci95_high']:+.3f}]",
                     "significant": r["ci95_low"] > 0 or r["ci95_high"] < 0})
@EXTRA_PAIRS@
cmp_df = pd.DataFrame(rows); print(cmp_df.to_string(index=False))
print()
print(pd.DataFrame(RESULTS).T[["n_tasks_run", "task_completion_rate", "steps_to_success", "recovery_rate", "mean_steps",
                               "parse_error_rate", "truncated_step_rate", "false_friend_rate", "wall_hours", "peak_vram_gb"]].round(3).to_string())
'''

TRACE_VIEW = '''
def action_seq(trace):
    return [(extract_action(m["content"])[0] or "PARSE_FAIL") for m in trace["history"] if m["role"] == "assistant"]

first = VARIANTS[0]
tr = TRACES[first]
print("Action sequences (first 8 tasks):")
for t in tr[:8]: print(f"  {t['task_id']:10s} passed={t['passed']!s:5s} recovery={t['recovery']!s:5s}", action_seq(t))

rec = [t for t in tr if t["recovery"] and t["passed"]]
print(f"\\nrecovered runs: {len(rec)}; false friends (visible pass, hidden fail): {sum(1 for t in tr if t['visible_passed'] and not t['passed'])}")
if rec:
    print(f"\\n=== one recovered trace: {rec[0]['task_id']} ===")
    for m in rec[0]["history"]: print(f"[{m['role']}]", m["content"][:600], "\\n")
'''

SAVE = '''
tracker = ExperimentTracker(f"exp05_agent_{TAG}_summary", "/kaggle/working")
tracker.set_config(model=MODEL_ID, dataset="mbpp_v2_eval.jsonl", hyperparameters={"variants": VARIANTS, "n_tasks": N_TASKS, "max_steps": MAX_STEPS, "seed": SEED})
wall_h = (time.time() - T0) / 3600
tracker.update_metrics({"results": RESULTS, "comparisons": rows})
tracker.update_hardware(max(r["peak_vram_gb"] for r in RESULTS.values()), psutil.virtual_memory().used / 1e9)
tracker.data["training"].update(wall_hours=round(wall_h, 4), gpu_hours=round(wall_h, 4)); tracker.save()
print(f"saved. wall = GPU hours = {wall_h:.2f}")
'''


def build(two_b: bool):
    if two_b:
        name, title, nb, tag, model_id = "10-agent-evaluation-2b", "10 Agent Evaluation 2B", "10_agent_evaluation_2b.ipynb", "2b", "Qwen/Qwen3.5-2B"
        variants = '["base", "grpo_from_base"]'
        sources = ["ycheng22/03-baseline-evaluation", "ycheng22/06-grpo-training", "ycheng22/07-grpo-evaluation"]
        ref_extra = '''try:
    REF["single-shot GRPO (NB 07)"] = counts_from(f"{mount('07-grpo-evaluation')}/**/exp02_grpo_from_base_mbpp_eval/generations.jsonl")
except AssertionError as e:
    print("NB 07 generations not available:", e)'''
        extra_pairs = '''if len(COUNTS) == 2:
    a, b = list(COUNTS)
    done = list(COUNTS[b])
    r = paired_bootstrap_diff({i: COUNTS[a][i] for i in done}, COUNTS[b], n=8, k=1)
    rows.append({"model": f"{b} vs {a} (both as agents)", "vs": "-", "n_tasks": len(done), "reference": round(r["mean_a"], 3), "agent": round(r["mean_b"], 3),
                 "diff": round(r["diff"], 3), "95% CI": f"[{r['ci95_low']:+.3f}, {r['ci95_high']:+.3f}]", "significant": r["ci95_low"] > 0 or r["ci95_high"] < 0})'''
        intro = """
# 10 - Agent Evaluation, Qwen3.5-2B (Phase 5/6 redo)

**Mode:** GPU (T4 x2 session, GPU 0), background commit, 2 h hard cap. Estimated ~0.7 GPU hours (two variants, <= 40 min each; the budget guard stops starting new tasks and reports how many ran).

## What is an agent here?
A ReAct loop: the model writes a *Thought* and one *Action* per turn - `read_file`, `write_file`, `run_tests`, `submit` - and sees an *Observation* (tool output) before the next turn. Maximum 5 turns. Unlike single-shot generation, the model can see a failing test and repair its code.

## What changed from the local run
1. **No hidden-test leak.** The old `run_tests` tool ran visible **and hidden** tests and printed `expected` vs `got`, so the agent could read answers to hidden tests. Now the tool runs only the **visible** example (the one already in the prompt); success is graded at the end on visible + hidden tests (`AgentWorkspace.grade`). Unit test: `test_agent_feedback_does_not_leak_hidden_tests`.
2. **Held-out MBPP tasks** (corrected v2 prompt with the example test) instead of the synthetic-complex tasks, so results are comparable with single-shot Pass@1 from NB 03/07 on the *same* tasks.
3. **Statistics:** paired bootstrap 95% CI of agent success vs single-shot Pass@1; deterministic sampling (re-seeded per task).
4. **Extra variant:** `grpo_from_base` (the best model in NB 07) as an agent - does RL on single-shot generation transfer to tool use?

## Questions
- Does tool feedback (a visible test result, up to 5 turns) beat single-shot sampling at the same model size?
- How often does the agent *recover* after a failed visible test?
- How often does it pass the visible example but fail hidden tests ("false friends")?
"""
    else:
        name, title, nb, tag, model_id = "11-agent-evaluation-4b", "11 Agent Evaluation 4B", "11_agent_evaluation_4b.ipynb", "4b", "Qwen/Qwen3.5-4B"
        variants = '["base"]'
        sources = ["ycheng22/03-baseline-evaluation", "ycheng22/10-agent-evaluation-2b"]
        ref_extra = ""
        extra_pairs = ""
        intro = """
# 11 - Agent Evaluation, Qwen3.5-4B (Phase 5/6 redo)

**Mode:** GPU (T4 x2 session, GPU 0), background commit, 1.5 h hard cap. Estimated ~0.6 GPU hours. The 4B model in fp16 needs ~8.5 GB of VRAM, so it fits on a single T4 (16 GB) with room for the KV cache; it is downloaded from the Hugging Face Hub (internet on).

Same agent, same tasks, same seeds and the same visible-test feedback as NB 10, so the only change is **model size**. Note that single-shot baselines in `REF` are the **2B** model's NB 03 samples; the 4B has no single-shot run here, so the fair size comparison is *agent 4B vs agent 2B*, shown in the last row.

## Questions
- Does doubling the parameters help more than giving the 2B model a tool loop?
- Is the recovery rate (fix after a failing visible test) higher for the bigger model?
"""

    cells = [md(intro), setup_cell(install_wheels=False, single_gpu=True),
             code(CONFIG.replace("@MODEL_ID@", model_id).replace("@TAG@", tag).replace("@VARIANTS@", variants)),
             code(LOAD.replace("@REF_EXTRA@", ref_extra)),
             md("## 1. Run the agent on the held-out tasks\nEach variant is loaded with its adapters merged (no per-step adapter overhead), run, then freed."),
             code(RUN)]
    if not two_b:
        cells.insert(5, code('''
nb10 = {}
for line in open(one(f"{mount('10-agent-evaluation-2b')}/**/exp05_agent_2b_base/agent_traces.jsonl")):
    t = json.loads(line); nb10[t["task_id"]] = 8 * int(t["passed"])
REF["agent 2B base (NB 10)"] = {i: nb10[i] for i in ids if i in nb10}
print({k: len(v) for k, v in REF.items()})
'''))
    cells += [md("## 2. Agent vs single-shot (paired bootstrap, 95% CI)\n`diff` > 0 with a CI above zero is the only thing reported as an improvement."),
              code(COMPARE.replace("@EXTRA_PAIRS@", extra_pairs)),
              md("## 3. Look at the traces\nNumbers hide behaviour: read the action sequences, one recovery and the false friends."),
              code(TRACE_VIEW),
              md("## 4. Save"), code(SAVE),
              md("## Takeaways (fill in)\n- Did tool feedback help (CI excludes zero)? By how much compared with Best-of-k selection (NB 09)?\n- What fraction of failures are parse errors or truncation rather than wrong logic?\n- Next: `12_research_dashboard` aggregates every experiment.")]
    write_kernel(f"kaggle/kernels/{name}", nb, f"ycheng22/{name}", title, cells, gpu=True, internet=True, kernel_sources=sources)


build(True)
build(False)
