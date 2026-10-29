"""Generates kaggle/kernels/02-dataset-verification. Run from the repo root: python kaggle/tools/build_nb02.py"""
import sys
sys.path.insert(0, "kaggle/tools")
from nbbuild import md, code, setup_cell, write_kernel

cells = [
    md("""
# 02 - Datasets & Verification (Phase 1)

**Mode:** CPU session (free, no GPU quota). Budget ~10 min. Internet not needed.

## Learning goals
1. Understand the task schema (prompt, starter code, visible vs hidden tests).
2. Prove the data files are exactly what we froze (SHA-256 manifest) - the test set must never change.
3. **Audit the data for quality and leakage**, not just format.
4. Validate every dataset with the verifier: every `reference_solution` should pass its own tests.
5. See how the sandboxed `SubprocessVerifier` classifies pass / wrong answer / syntax error / timeout / memory bomb, and why that is safe on Kaggle.
"""),
    setup_cell(install_wheels=False),
    md("## 1. Integrity: are the files byte-identical to the frozen manifest?"),
    code("""
import hashlib
manifest = {}
for line in open(f"{DATA}/MANIFEST.sha256"):
    h, name = line.split()
    manifest[name.lstrip("*")] = h
integrity = {}
for name, expected in manifest.items():
    actual = hashlib.sha256(open(f"{DATA}/{name}", "rb").read()).hexdigest()
    integrity[name] = (actual == expected)
    print(("OK      " if actual == expected else "MISMATCH"), name)
assert all(integrity.values()), "a dataset file differs from the frozen manifest!"
"""),
    md("""
## 2. Schema validation
`load_dataset` parses each line into a pydantic `TaskSchema`; any malformed line raises with its line number.
"""),
    code("""
import pandas as pd
from llm_lab_utils import load_dataset
FILES = ["train", "train_5k", "dev", "dev_500", "test", "train_complex_500"]
data = {f: load_dataset(f"{DATA}/{f}.jsonl") for f in FILES}
t = data["test"][0]
print("Example task:", t.id, "|", t.category, "|", t.prompt)
print("visible:", t.visible_tests, "\\nhidden: ", t.hidden_tests)
pd.DataFrame({f: {"tasks": len(v),
                  "visible_tests/task": sum(len(x.visible_tests) for x in v) / len(v),
                  "hidden_tests/task": sum(len(x.hidden_tests) for x in v) / len(v)} for f, v in data.items()}).T
"""),
    md("""
## 3. Data quality & leakage audit
AGENTS.md rule 7: never mix train/dev/test. Format validity is not enough - we also measure how similar the splits are.
"""),
    code("""
def key(x): return (x.prompt, tuple(tc.input for tc in x.visible_tests))

audit = {"per_file": {}, "pairs": {}}
for f, v in data.items():
    audit["per_file"][f] = {"tasks": len(v), "unique_ids": len({x.id for x in v}),
                            "unique_prompts": len({x.prompt for x in v})}
print(pd.DataFrame(audit["per_file"]).T, "\\n")

for a, b in [("train", "test"), ("train_5k", "test"), ("train_5k", "dev_500"), ("dev", "test"), ("train", "dev")]:
    A, B = data[a], data[b]
    ids = {x.id for x in A} & {x.id for x in B}
    prompts = {x.prompt for x in A} & {x.prompt for x in B}
    same = {key(x) for x in A} & {key(x) for x in B}
    audit["pairs"][f"{a}__{b}"] = {"id_overlap": len(ids), "prompt_overlap": len(prompts),
                                   "distinct_prompt_plus_visible_test_overlap": len(same),
                                   "distinct_in_second": len({key(x) for x in B})}
    print(f"{a:9s} vs {b:8s} id-overlap={len(ids):4d}  shared-prompts={len(prompts)}  "
          f"identical (prompt+visible test)={len(same)} of {len({key(x) for x in B})} distinct in {b}")
"""),
    md("""
### How to read this
- The synthetic sets are generated from **only 4 prompt templates** (add, reverse, list-sum, palindrome). Dev/test differ from train only in numbers/strings, so a model that learned the 4 templates scores ~100%. That is why Baseline, SFT and GRPO all hit 100% in Phases 1-3: **the synthetic benchmark saturates and cannot separate methods**.
- IDs overlap across files because each file is numbered by the same generator (`toy_add_000` ...). IDs are *not* unique keys across splits - never join splits on `id`.
- `train_complex_500` (MBPP) has ~418 distinct prompts: this is the set that actually measures capability (Phases 4-7).
- We do **not** fix this by editing files: the frozen test set must not change (AGENTS.md rule 2). The finding is documented instead and drives the move to MBPP.
"""),
    md("## 4. Every reference solution must pass its own tests"),
    code("""
from concurrent.futures import ThreadPoolExecutor
from llm_lab_utils import SubprocessVerifier
verifier = SubprocessVerifier(timeout=5.0, max_memory_mb=512)
print("cpus:", os.cpu_count())

def check(task):
    return task.id, verifier.verify(task.model_dump(), task.reference_solution or "")

ref_report = {}
for f in FILES:
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=min(8, (os.cpu_count() or 2) * 2)) as ex:
        res = list(ex.map(check, data[f]))
    bad = [(i, r["stderr"].strip().splitlines()[-1][:120] if r["stderr"].strip() else "") for i, r in res if not r["passed"]]
    ref_report[f] = {"tasks": len(res), "passed": len(res) - len(bad), "failed_ids": [b[0] for b in bad][:30],
                     "seconds": round(time.time() - t0, 1)}
    print(f"{f:18s} {len(res) - len(bad):5d}/{len(res):5d} pass  ({ref_report[f]['seconds']}s)")
    for b in bad[:5]:
        print("    FAIL", b)
"""),
    md("""
Any failures above are data problems (a broken test or a reference solution that needs `starter_code` setup), **not** model failures. They put a ceiling under every later Pass@k number, so we record them rather than hide them.
"""),
    md("## 5. How the verifier classifies outcomes"),
    code("""
task = {"id": "demo", "version": 1, "category": "math", "difficulty": 0.1, "prompt": "double a number",
        "visible_tests": [{"input": "f(2)", "expected": "4"}],
        "hidden_tests": [{"input": "f(0)", "expected": "0"}, {"input": "f(-3)", "expected": "-6"}]}
demos = {"correct": "def f(x): return 2 * x",
         "passes visible, fails hidden": "def f(x): return 4 if x == 2 else 1",
         "syntax error": "def f(x) return x",
         "runtime error": "def f(x): return 1 / 0",
         "infinite loop (timeout)": "def f(x):\\n    while True: pass",
         "memory bomb": "def f(x):\\n    return len([0] * (10 ** 10))"}
rows = []
for name, src in demos.items():
    r = SubprocessVerifier(timeout=3.0).verify(task, src)
    rows.append({"case": name, "passed": r["passed"], "reward": r["reward"], "timeout": r["timeout"],
                 "syntax_error": r["syntax_error"], "runtime_error": r["runtime_error"],
                 "seconds": round(r["duration_seconds"], 2)})
demo_df = pd.DataFrame(rows)
demo_df
"""),
    md("""
### Why subprocess instead of Docker here
| | Local Windows | Kaggle |
|---|---|---|
| Isolation | Docker container, `--network none` | The notebook container itself is ephemeral; with Internet off it has no network |
| Extra limits | container memory limit | `RLIMIT_AS` (memory), `RLIMIT_CPU`, wall-clock timeout, fresh temp dir, minimal env, `python -I` |
| Speed | ~1 s container start per candidate | tens of ms |

The reward is binary (all tests pass = 1.0), identical in both back-ends, so scores stay comparable. **Never run this verifier on your own machine with untrusted code** - that is what the Docker verifier is for (AGENTS.md rules 8-9).
"""),
    md("## 6. Save results"),
    code("""
from llm_lab_utils import ExperimentTracker
tracker = ExperimentTracker("exp01_dataset_audit", "/kaggle/working")
tracker.set_config(model="n/a (no model)", dataset=",".join(FILES), hyperparameters={"verifier_timeout_s": 5.0, "max_memory_mb": 512})
tracker.update_metrics({"integrity": integrity, "audit": audit, "reference_solutions": ref_report,
                        "verifier_demo": demo_df.to_dict("records")})
wall_h = (time.time() - T0) / 3600
tracker.data["training"].update(wall_hours=round(wall_h, 4), gpu_hours=0.0)
tracker.save()
print(f"saved /kaggle/working/exp01_dataset_audit/summary.json   wall: {wall_h * 60:.1f} min, GPU hours: 0")
"""),
    md("""
## Takeaways
- Data files are verified byte-for-byte against the frozen manifest.
- Format-valid is not the same as informative: the synthetic benchmark has 4 templates and heavy split overlap, so it saturates.
- The verifier is the project's ground truth; checking the reference solutions bounds what any model can score.
- Next: `03_baseline_evaluation` (GPU).
"""),
]

write_kernel("kaggle/kernels/02-dataset-verification", "02_dataset_and_verification.ipynb",
             "ycheng22/02-dataset-verification", "02 Dataset Verification", cells, gpu=False, internet=False)
