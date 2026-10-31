"""
Plan Step 19g data: derive two equally sized GRPO training sets from the NB 08 mining report (CPU only, no code executed).

  learnable : tasks the base model solved in 1, 2 or 3 of 4 samples (mixed outcomes -> non-zero GRPO signal)
  random    : the same number of tasks drawn uniformly from all 256 mbpp_v2_train tasks (seed 0)

Both are subsets of mbpp_v2_train.jsonl only (dev/eval are never touched, AGENTS.md rule 7). Writes the two jsonl files
to kaggle/datasets/llm-lab-data/, an audit json next to them, and appends their sha256 to MANIFEST.sha256 (frozen afterwards).
Run from the repo root: python kaggle/tools/build_mining_sets.py
"""
import hashlib
import json
import os
import random

DATA = "kaggle/datasets/llm-lab-data"
REPORT = "experiments/kaggle_outputs/08/exp03_mining_base/mining_report.json"

report = json.load(open(REPORT, encoding="utf-8"))
train = [json.loads(l) for l in open(f"{DATA}/mbpp_v2_train.jsonl", encoding="utf-8")]
by_id = {t["id"]: t for t in train}
counts = {t["task_id"]: t["c_correct"] for t in report["tasks"]}
assert set(counts) == set(by_id) and len(train) == 256, "mining report must cover exactly mbpp_v2_train"

learnable = [t["id"] for t in train if counts[t["id"]] in (1, 2, 3)]
rng = random.Random(0)
rand = sorted(rng.sample([t["id"] for t in train], len(learnable)))
sets = {"mbpp_v2_train_learnable.jsonl": learnable, "mbpp_v2_train_random.jsonl": rand}

manifest = open(f"{DATA}/MANIFEST.sha256", encoding="utf-8").read()
audit = {"source_report": REPORT, "mining_seed": report["seed"], "random_seed": 0, "sizes": {}, "pass_count_histogram": {}}
for fname, ids in sets.items():
    keep = set(ids)
    path = f"{DATA}/{fname}"
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for t in train:
            if t["id"] in keep:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    audit["sizes"][fname] = len(ids)
    hist = {c: sum(counts[i] == c for i in ids) for c in range(5)}
    audit["pass_count_histogram"][fname] = hist
    if fname not in manifest:
        manifest = manifest.rstrip("\n") + f"\n{sha} *{fname}\n"
    print(fname, len(ids), "tasks, base pass-count histogram (0..4):", hist, sha[:12])
open(f"{DATA}/MANIFEST.sha256", "w", encoding="utf-8", newline="\n").write(manifest)
json.dump(audit, open(f"{DATA}/mbpp_v2_train_mining_sets_audit.json", "w"), indent=1)
