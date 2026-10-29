import json
import random
from pathlib import Path
from typing import Any, Dict

from .evaluator import extract_code, generate_completions

CATEGORIES = ("solved", "borderline", "hard", "impossible")


def classify_task(c: int, n: int) -> str:
    """
    Classifies a task by pass rate c/n. For n=4:
    c>=3 solved, c==2 borderline, c==1 hard, c==0 impossible.
    """
    pass_rate = c / n
    if pass_rate >= 0.75:
        return "solved"
    elif pass_rate >= 0.5:
        return "borderline"
    elif pass_rate > 0:
        return "hard"
    return "impossible"


def mine_tasks(model, tokenizer, tasks, verifier, n_samples: int, gen_cfg: Dict[str, Any],
               experiment_id: str, output_dir, dataset_name: str = "", seed: int = 42,
               progress: bool = True) -> Dict[str, Any]:
    """
    Samples n_samples completions per task, classifies each task, and writes
    solved/borderline/hard/impossible.jsonl plus the hard/random/mixed ablation
    sets and mining_report.json to output_dir. Returns the report dict.
    """
    rng = random.Random(seed)  # seeded so ablation sets are reproducible (AGENTS.md rule 12)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {c: [] for c in CATEGORIES}
    report = {
        "experiment_id": experiment_id,
        "dataset": dataset_name,
        "seed": seed,
        "total_tasks_processed": len(tasks),
        "classification_counts": {c: 0 for c in CATEGORIES},
        "tasks": [],
    }

    iterator = tasks
    if progress:
        from tqdm.auto import tqdm
        iterator = tqdm(tasks)

    for task_obj in iterator:
        task_dict = task_obj.model_dump()
        completions, _, _ = generate_completions(model, tokenizer, task_dict, gen_cfg, n=n_samples)
        c_correct = sum(
            bool(verifier.verify(task_dict, extract_code(c))["passed"]) for c in completions
        )

        category = classify_task(c_correct, n_samples)
        task_dict["mining_metadata"] = {
            "experiment_id": experiment_id,
            "c_correct": c_correct,
            "n_samples": n_samples,
            "category": category,
        }
        results[category].append(task_dict)
        report["classification_counts"][category] += 1
        report["tasks"].append({"task_id": task_dict["id"], "c_correct": c_correct, "category": category})

    def write(name, rows):
        with open(output_dir / name, "w", encoding="utf-8") as f:
            for t in rows:
                f.write(json.dumps(t) + "\n")

    for cat in CATEGORIES:
        write(f"{cat}.jsonl", results[cat])

    hard_borderline = results["hard"] + results["borderline"]
    random_pool = results["solved"] + results["impossible"]
    target_size = len(hard_borderline)
    plain_tasks = [t.model_dump() for t in tasks]

    write("hard.jsonl", hard_borderline)
    write("random.jsonl", rng.sample(plain_tasks, min(target_size, len(plain_tasks))))

    num_hard = int(target_size * 0.7)
    num_rand = target_size - num_hard
    mixed = []
    if hard_borderline:
        mixed.extend(rng.choices(hard_borderline, k=min(num_hard, len(hard_borderline))))
    if random_pool:
        mixed.extend(rng.choices(random_pool, k=min(num_rand, len(random_pool))))
    write("mixed.jsonl", mixed)

    with open(output_dir / "mining_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report
