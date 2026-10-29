"""
Splits train_complex_v2.jsonl (406 MBPP tasks, example-test prompt) into disjoint eval / dev / train files.

  mbpp_v2_eval.jsonl : first 100 tasks -> the held-out evaluation set. These are EXACTLY the 100 tasks
                       already baselined in NB 03 (base model: Pass@1 40.6 / Pass@4 61.4 / Pass@8 69.0).
  mbpp_v2_dev.jsonl  : next 50  -> monitoring (eval loss, quick checks). Never used for final reporting.
  mbpp_v2_train.jsonl: remaining 256 -> SFT / GRPO training tasks.

The order is the file order (MBPP task-id order), so the split is deterministic and needs no RNG.
Run from the repo root: python kaggle/tools/build_mbpp_splits.py
"""
import hashlib
import json

SRC = "kaggle/datasets/llm-lab-data/train_complex_v2.jsonl"
OUTDIR = "kaggle/datasets/llm-lab-data/"
MANIFEST = OUTDIR + "MANIFEST.sha256"

rows = [json.loads(l) for l in open(SRC, encoding="utf-8") if l.strip()]
splits = {"mbpp_v2_eval": rows[:100], "mbpp_v2_dev": rows[100:150], "mbpp_v2_train": rows[150:]}

# safety: disjoint ids and disjoint prompts (leakage check, AGENTS.md rule 7)
ids = [set(t["id"] for t in v) for v in splits.values()]
prompts = [set(t["prompt"] for t in v) for v in splits.values()]
for i in range(3):
    for j in range(i + 1, 3):
        assert not ids[i] & ids[j], "id overlap between splits"
        assert not prompts[i] & prompts[j], "prompt overlap between splits"
assert sum(len(v) for v in splits.values()) == len(rows)

lines = [l for l in open(MANIFEST, encoding="utf-8").read().splitlines() if l.strip() and "mbpp_v2_" not in l]
for name, v in splits.items():
    path = f"{OUTDIR}{name}.jsonl"
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for t in v:
            t["metadata"] = {**t.get("metadata", {}), "split": name}
            f.write(json.dumps(t) + "\n")
    lines.append(f"{hashlib.sha256(open(path, 'rb').read()).hexdigest()} *{name}.jsonl")
    print(name, len(v))
open(MANIFEST, "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
print("manifest updated")
