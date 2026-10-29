"""
Builds train_complex_v2.jsonl from train_complex_500.jsonl (which is left untouched, AGENTS.md rule 17).

Why: only 14/420 v1 prompts mention the function name the tests call (e.g. `get_ludic`), so the model
cannot know what to define and Phase 4 reported 420/420 'impossible'. Standard MBPP prompting shows one
example test; v2 appends the first visible test to the prompt.

Also drops the 14 tasks whose own reference solution fails the verifier (NB 02 audit, exp01_dataset_audit).

Run from repo root: python kaggle/tools/build_mbpp_v2.py
"""
import hashlib
import json

SRC = "data/train_complex_500.jsonl"
DST = "kaggle/datasets/llm-lab-data/train_complex_v2.jsonl"
MANIFEST = "kaggle/datasets/llm-lab-data/MANIFEST.sha256"
BROKEN = ["mbpp_737", "mbpp_787", "mbpp_794", "mbpp_98", "mbpp_123", "mbpp_124", "mbpp_137",
          "mbpp_139", "mbpp_163", "mbpp_233", "mbpp_246", "mbpp_248", "mbpp_276", "mbpp_300"]

kept = 0
with open(SRC, encoding="utf-8") as f, open(DST, "w", encoding="utf-8", newline="\n") as out:
    for line in f:
        if not line.strip():
            continue
        t = json.loads(line)
        if t["id"] in BROKEN:
            continue
        ex = t["visible_tests"][0]
        t["prompt"] = (t["prompt"] + "\n\nYour function must satisfy this example:\n"
                       f"assert {ex['input']} == {ex['expected']}")
        t["version"] = 2
        t["metadata"] = {**t.get("metadata", {}), "prompt_variant": "v2_with_example_test", "derived_from": SRC}
        out.write(json.dumps(t) + "\n")
        kept += 1
print("wrote", DST, kept, "tasks; dropped", len(BROKEN))

h = hashlib.sha256(open(DST, "rb").read()).hexdigest()
lines = [l for l in open(MANIFEST, encoding="utf-8").read().splitlines() if l.strip() and "train_complex_v2" not in l]
lines.append(f"{h} *train_complex_v2.jsonl")
open(MANIFEST, "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
print("manifest updated")
