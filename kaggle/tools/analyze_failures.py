"""
Step 19a: failure analysis of saved evaluation samples (CPU only; no code is executed, only stored verifier output is read).

Run from the repo root after pulling outputs:  python kaggle/tools/analyze_failures.py
Reads  experiments/kaggle_outputs/{03-*,05-*,07}/.../generations.jsonl   (100 MBPP eval tasks x 8 samples per model)
Writes experiments/analysis/19a_failure_analysis.json
"""
import collections
import glob
import json
import os
import re

ROOT = "experiments/kaggle_outputs"
SOURCES = {
    "base": f"{ROOT}/03-*/**/exp00_baseline_mbpp_v2/generations.jsonl",
    "sft": f"{ROOT}/05-*/**/exp01_sft_mbpp_eval/generations.jsonl",
    "grpo_from_base": f"{ROOT}/07/exp02_grpo_from_base_mbpp_eval/generations.jsonl",
    "grpo_from_sft": f"{ROOT}/07/exp02_grpo_from_sft_mbpp_eval/generations.jsonl",
}
MODULES = {"math", "re", "itertools", "collections", "heapq", "bisect", "sys", "string", "functools", "operator"}


def load(pattern):
    hits = glob.glob(pattern, recursive=True)
    if not hits:
        return None
    return [json.loads(l) for l in open(hits[0], encoding="utf-8")]


def failure_type(r):
    v = r["verification"]
    if v["passed"]:
        return "pass"
    if v["timeout"]:
        return "timeout"
    if v["syntax_error"]:
        return "syntax_error"
    err = v["stderr"]
    m = re.search(r"name '(\w+)' is not defined", err)
    if m:
        return "missing_import" if m.group(1) in MODULES else "wrong_function_name"
    if "failed: expected" in err or "AssertionError" in err:
        return "wrong_output"
    return "other_exception"


def per_task_counts(rows):
    c = collections.defaultdict(int)
    for r in rows:
        c[r["task_id"]] += int(r["verification"]["passed"])
    return dict(c)


def main():
    data = {k: load(p) for k, p in SOURCES.items()}
    data = {k: v for k, v in data.items() if v}
    out = {"models": {}}
    for name, rows in data.items():
        n = len(rows)
        types = collections.Counter(failure_type(r) for r in rows)
        named_ok = [r for r in rows if failure_type(r) not in ("wrong_function_name", "syntax_error", "timeout")]
        out["models"][name] = {
            "samples": n,
            "pass_at_1": round(types["pass"] / n, 3),
            "failure_share": {k: round(v / n, 3) for k, v in sorted(types.items())},
            "pass_rate_given_function_exists": round(sum(r["verification"]["passed"] for r in named_ok) / len(named_ok), 3),
            "unclosed_code_fence_rate": round(sum(r["completion"].count("```") % 2 == 1 for r in rows) / n, 3),
        }

    # per-task: where did GRPO(base) gain over base?
    cb, cg = per_task_counts(data["base"]), per_task_counts(data["grpo_from_base"])
    ids = sorted(cb)
    bucket = collections.defaultdict(float)
    for i in ids:
        key = "base_never_solves" if cb[i] == 0 else "base_always_solves" if cb[i] == 8 else "base_sometimes_solves"
        bucket[key] += (cg[i] - cb[i]) / 8
    out["grpo_vs_base_per_task"] = {
        "tasks": len(ids),
        "improved_by_3_or_more_of_8": sum(cg[i] - cb[i] >= 3 for i in ids),
        "worsened_by_3_or_more_of_8": sum(cb[i] - cg[i] >= 3 for i in ids),
        "both_never_solved": sum(cb[i] == 0 and cg[i] == 0 for i in ids),
        "both_always_solved": sum(cb[i] == 8 and cg[i] == 8 for i in ids),
        "pass_at_1_gain_by_base_bucket_in_task_units": {k: round(v, 2) for k, v in bucket.items()},
    }
    os.makedirs("experiments/analysis", exist_ok=True)
    json.dump(out, open("experiments/analysis/19a_failure_analysis.json", "w"), indent=1)

    types_all = sorted({t for m in out["models"].values() for t in m["failure_share"]})
    print(f"{'failure share of all samples':32s}" + "".join(f"{m:>16s}" for m in out["models"]))
    for t in types_all:
        print(f"{t:32s}" + "".join(f"{out['models'][m]['failure_share'].get(t, 0):16.3f}" for m in out["models"]))
    for key in ("pass_rate_given_function_exists", "unclosed_code_fence_rate"):
        print(f"{key:32s}" + "".join(f"{out['models'][m][key]:16.3f}" for m in out["models"]))
    print(json.dumps(out["grpo_vs_base_per_task"], indent=1))


if __name__ == "__main__":
    main()
