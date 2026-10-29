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


def generate_completions(model, tokenizer, task_dict: Dict[str, Any], gen_cfg: Dict[str, Any], n: int = 1):
    """
    Samples n completions for one task in a SINGLE generate() call (num_return_sequences=n),
    which is several times cheaper per sample than n separate calls.
    Returns (completions, n_new_tokens_total, latency_seconds_for_the_batch).
    """
    import torch

    text = tokenizer.apply_chat_template(
        build_messages(task_dict), tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    prompt_len = inputs.input_ids.shape[1]

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=gen_cfg.get("max_new_tokens", 384),
            temperature=gen_cfg.get("temperature", 0.7),
            top_p=gen_cfg.get("top_p", 0.9),
            do_sample=gen_cfg.get("do_sample", True),
            num_return_sequences=n,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency = time.time() - start

    new = outputs[:, prompt_len:]
    completions = [tokenizer.decode(row, skip_special_tokens=True) for row in new]
    n_tokens = int((new != tokenizer.eos_token_id).sum())
    return completions, n_tokens, latency


def generate_completion(model, tokenizer, task_dict: Dict[str, Any], gen_cfg: Dict[str, Any]):
    """Single sample. Returns (completion, n_new_tokens, latency_seconds)."""
    completions, n_tokens, latency = generate_completions(model, tokenizer, task_dict, gen_cfg, n=1)
    return completions[0], n_tokens, latency


def run_evaluation(model, tokenizer, tasks, verifier, gen_cfg, k_values, tracker=None, progress=True,
                   seed: int = 1234, sample_chunk: int = 8):
    """
    Samples max(k_values) completions per task (in chunks of `sample_chunk` per generate call),
    verifies each, and returns (metrics, per_task_results). If a tracker is given, per-generation
    records are appended to its generations.jsonl and metrics/hardware are updated.
    The RNG is seeded so sampling is reproducible (AGENTS.md rule 12).
    """
    import torch
    import psutil

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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

        remaining = max_k
        while remaining > 0:
            n = min(sample_chunk, remaining)
            remaining -= n
            completions, n_tokens, latency = generate_completions(model, tokenizer, task_dict, gen_cfg, n=n)
            total_latency += latency
            total_tokens += n_tokens

            for completion in completions:
                total_generations += 1
                candidate_code = extract_code(completion)
                verify_result = verifier.verify(task_dict, candidate_code)

                c_correct += bool(verify_result["passed"])
                timeout_count += bool(verify_result["timeout"])
                syntax_count += bool(verify_result["syntax_error"])

                if tracker is not None:
                    tracker.append_task_result({
                        "task_id": task_dict["id"],
                        "completion": completion,
                        "candidate_code": candidate_code,
                        "latency": latency / n,
                        "verification": verify_result,
                    })

            if tracker is not None:
                peak_vram = 0.0
                if torch.cuda.is_available():
                    peak_vram = max(
                        torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())
                    ) / (1024 ** 3)
                peak_ram = max(peak_ram, psutil.Process().memory_info().rss / (1024 ** 3))
                tracker.update_hardware(peak_vram, peak_ram)

        task_results.append({"task_id": task_dict["id"], "n": max_k, "c": c_correct})

    metrics = aggregate_pass_at_k(task_results, k_values)
    denom = max(total_generations, 1)
    metrics["timeout_rate"] = timeout_count / denom
    metrics["syntax_error_rate"] = syntax_count / denom
    metrics["mean_latency"] = total_latency / denom   # seconds per sample, amortized over the batch
    metrics["tokens_generated"] = total_tokens

    if tracker is not None:
        tracker.update_metrics(metrics)

    return metrics, task_results
