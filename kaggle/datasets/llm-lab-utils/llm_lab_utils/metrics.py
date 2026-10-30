import math
from typing import List, Dict

def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Calculates Pass@k metric for a single task.
    n: total number of generations
    c: number of correct generations
    k: k for Pass@k
    
    If n - c < k, then we are guaranteed to select a correct generation 
    (or binomial coefficient is 0), so probability is 1.0.
    """
    if n - c < k:
        return 1.0
    
    # E[1 - comb(n-c, k) / comb(n, k)]
    prob_fail = math.comb(n - c, k) / math.comb(n, k)
    return 1.0 - prob_fail

def aggregate_pass_at_k(task_results: List[Dict], k_values: List[int]) -> Dict[str, float]:
    """
    Aggregates Pass@k across all tasks.
    task_results should contain a list of dicts with 'n' and 'c' for each task.
    { "task_1": {"n": 8, "c": 2}, ... }
    """
    metrics = {}
    for k in k_values:
        total = 0.0
        count = 0
        for res in task_results:
            n = res.get("n", 0)
            c = res.get("c", 0)
            if n >= k:
                total += pass_at_k(n, c, k)
                count += 1
        
        metrics[f"pass_at_{k}"] = total / count if count > 0 else 0.0
        
    return metrics


def paired_bootstrap_diff(counts_a: Dict[str, int], counts_b: Dict[str, int], n: int, k: int = 1,
                          iters: int = 10000, seed: int = 0) -> Dict[str, float]:
    """
    Paired bootstrap over tasks for the difference in Pass@k (b minus a).
    counts_*: {task_id: c_correct_out_of_n}. Only task ids present in both are used.
    Returns mean difference, 95% CI, and P(diff > 0). With ~100 tasks a few points of difference is often noise.
    """
    import random
    ids = sorted(set(counts_a) & set(counts_b))
    da = [pass_at_k(n, counts_a[i], k) for i in ids]
    db = [pass_at_k(n, counts_b[i], k) for i in ids]
    diffs = [y - x for x, y in zip(da, db)]
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(iters))
    return {"tasks": len(ids), "mean_a": sum(da) / len(da), "mean_b": sum(db) / len(db),
            "diff": sum(diffs) / len(diffs), "ci95_low": means[int(0.025 * iters)],
            "ci95_high": means[int(0.975 * iters)], "p_better": sum(m > 0 for m in means) / iters}


def selection_accuracy(samples: List[Dict], k: int, draws: int = 2000, seed: int = 0) -> float:
    """
    Realistic best-of-k: draw k of a task's samples at random, keep those that pass the example (visible) test
    the model was shown, pick one of them at random (if none pass, pick any of the k), and score it on the full
    tests. Unlike Pass@k this needs no oracle that sees the hidden tests.
    samples: [{"visible": bool, "full": bool}, ...] for ONE task. Returns the expected accuracy over random draws.
    """
    import random
    rng = random.Random(seed)
    total = 0.0
    for _ in range(draws):
        pick = rng.sample(samples, k)
        pool = [s for s in pick if s["visible"]] or pick
        total += float(rng.choice(pool)["full"])
    return total / draws
