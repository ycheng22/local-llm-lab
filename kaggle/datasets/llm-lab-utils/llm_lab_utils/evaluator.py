import time
from typing import Any, Dict, List

from .metrics import aggregate_pass_at_k

SYSTEM_PROMPT = (
    "You are a helpful programming assistant. Write Python code to solve the user's problem. "
    "Only return the python code inside markdown code blocks."
)


def extract_code(completion: str) -> str:
    """Extracts Python code from a markdown-formatted completion."""
    if "```python" in completion:
        return completion.split("```python")[1].split("```")[0].strip()
    elif "```" in completion:
        return completion.split("```")[1].split("```")[0].strip()
    return completion.strip()


def build_messages(task_dict: Dict[str, Any]) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task_dict["prompt"] + "\n\n" + (task_dict.get("starter_code") or "")},
    ]


def generate_completion(model, tokenizer, task_dict: Dict[str, Any], gen_cfg: Dict[str, Any]):
    """Samples one completion for a task. Returns (completion, n_new_tokens, latency_seconds)."""
    text = tokenizer.apply_chat_template(
        build_messages(task_dict), tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    start = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=gen_cfg.get("max_new_tokens", 384),
        temperature=gen_cfg.get("temperature", 0.7),
        top_p=gen_cfg.get("top_p", 0.9),
        do_sample=gen_cfg.get("do_sample", True),
        pad_token_id=tokenizer.eos_token_id,
    )
    latency = time.time() - start

    gen_tokens = outputs[0][inputs.input_ids.shape[1]:]
    completion = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    return completion, len(gen_tokens), latency


def run_evaluation(model, tokenizer, tasks, verifier, gen_cfg, k_values, tracker=None, progress=True):
    """
    Samples max(k_values) completions per task, verifies each, and returns
    (metrics, per_task_results). If a tracker is given, per-generation records
    are appended to its generations.jsonl and metrics/hardware are updated.
    """
    import torch
    import psutil

    max_k = max(k_values)
    iterator = tasks
    if progress:
        from tqdm.auto import tqdm
        iterator = tqdm(tasks)

    task_results = []
    total_latency = 0.0
    timeout_count = syntax_count = total_generations = total_tokens = 0
    peak_ram = 0.0

    for task_obj in iterator:
        task_dict = task_obj.model_dump()
        c_correct = 0

        for _ in range(max_k):
            completion, n_tokens, latency = generate_completion(model, tokenizer, task_dict, gen_cfg)
            total_latency += latency
            total_tokens += n_tokens
            total_generations += 1

            candidate_code = extract_code(completion)
            verify_result = verifier.verify(task_dict, candidate_code)

            c_correct += bool(verify_result["passed"])
            timeout_count += bool(verify_result["timeout"])
            syntax_count += bool(verify_result["syntax_error"])

            if tracker is not None:
                if torch.cuda.is_available():
                    peak_vram = max(
                        torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())
                    ) / (1024 ** 3)
                else:
                    peak_vram = 0.0
                peak_ram = max(peak_ram, psutil.Process().memory_info().rss / (1024 ** 3))
                tracker.update_hardware(peak_vram, peak_ram)
                tracker.append_task_result({
                    "task_id": task_dict["id"],
                    "completion": completion,
                    "candidate_code": candidate_code,
                    "latency": latency,
                    "verification": verify_result,
                })

        task_results.append({"task_id": task_dict["id"], "n": max_k, "c": c_correct})

    metrics = aggregate_pass_at_k(task_results, k_values)
    denom = max(total_generations, 1)
    metrics["timeout_rate"] = timeout_count / denom
    metrics["syntax_error_rate"] = syntax_count / denom
    metrics["mean_latency"] = total_latency / denom
    metrics["tokens_generated"] = total_tokens

    if tracker is not None:
        tracker.update_metrics(metrics)

    return metrics, task_results
