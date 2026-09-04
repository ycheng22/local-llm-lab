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
