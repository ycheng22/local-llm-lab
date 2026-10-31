"""
Builds kaggle/datasets/llm-lab-data/humaneval_v1.jsonl (frozen external OOD eval set, plan Step 19b) from the
official OpenAI HumanEval parquet. Conversion only - NO code is executed here (AGENTS.md rule 8); the reference
solutions are verified inside Kaggle by the first cells of the x02 notebook.

Usage (repo root):  python kaggle/tools/build_humaneval.py path/to/test-00000-of-00001.parquet
Source: https://huggingface.co/datasets/openai/openai_humaneval  (openai_humaneval/test-00000-of-00001.parquet)
"""
import hashlib
import json
import sys

import pandas as pd

sys.path.insert(0, "kaggle/datasets/llm-lab-utils")
from llm_lab_utils.humaneval import convert_problem

OUT = "kaggle/datasets/llm-lab-data/humaneval_v1.jsonl"
MANIFEST = "kaggle/datasets/llm-lab-data/MANIFEST.sha256"
AUDIT = "kaggle/datasets/llm-lab-data/humaneval_v1_audit.json"

df = pd.read_parquet(sys.argv[1])
tasks, dropped = [], []
for row in df.to_dict("records"):
    task, reason = convert_problem(row)
    if task is None:
        dropped.append({"task_id": row["task_id"], "reason": reason})
    else:
        tasks.append(task)

with open(OUT, "w", encoding="utf-8", newline="\n") as f:
    for t in tasks:
        f.write(json.dumps(t, ensure_ascii=False) + "\n")

json.dump({"source_rows": len(df), "converted": len(tasks), "dropped": dropped}, open(AUDIT, "w"), indent=1)
sha = hashlib.sha256(open(OUT, "rb").read()).hexdigest()
lines = [l for l in open(MANIFEST).read().splitlines() if not l.endswith("*humaneval_v1.jsonl")]
open(MANIFEST, "w", newline="\n").write("\n".join(lines + [f"{sha} *humaneval_v1.jsonl"]) + "\n")
print(f"converted {len(tasks)}/{len(df)}; dropped {len(dropped)}")
from collections import Counter
print(Counter(d["reason"].split(":")[0] for d in dropped).most_common())
print("sha256", sha)
