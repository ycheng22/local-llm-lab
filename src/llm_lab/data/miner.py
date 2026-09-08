import argparse
import sys
import torch
import time
import json
import os
from pathlib import Path
from tqdm import tqdm
from pydantic import BaseModel, Field
from typing import List, Dict

from llm_lab.inference.generate import load_config, setup_model_and_tokenizer
from llm_lab.data import load_dataset
from llm_lab.verifier import DockerVerifier
from llm_lab.constants import resolve_path

def extract_code(completion: str) -> str:
    """Extracts Python code from a markdown-formatted completion."""
    if "```python" in completion:
        return completion.split("```python")[1].split("```")[0].strip()
    elif "```" in completion:
        return completion.split("```")[1].split("```")[0].strip()
    return completion.strip()

def classify_task(c: int, n: int) -> str:
    """
    Classifies a task based on the number of correct responses (c) out of (n).
    If n=4:
    - c >= 3: solved
    - c == 2: borderline
    - c == 1: hard
    - c == 0: impossible
    """
    pass_rate = c / n
    if pass_rate >= 0.75:
        return "solved"
    elif pass_rate >= 0.5:
        return "borderline"
    elif pass_rate > 0:
        return "hard"
    else:
        return "impossible"

def main():
    parser = argparse.ArgumentParser(description="Mine hard examples from a dataset.")
    parser.add_argument("--config", type=str, required=True, help="Path to mining YAML config file")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    experiment_id = config["experiment"]["id"]
    output_dir = resolve_path(config["logging"]["output_dir"])
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading dataset from {config['mining']['dataset']}...")
    tasks = load_dataset(config['mining']['dataset'])
    max_tasks = config["mining"].get("max_tasks", len(tasks))
    tasks = tasks[:max_tasks]
    
    try:
        model, tokenizer = setup_model_and_tokenizer(config["model"])
    except Exception as e:
        print(f"Failed to load model: {e}", file=sys.stderr)
        sys.exit(1)
        
    verifier = DockerVerifier()
    n_samples = config["mining"].get("num_samples", 4)
    gen_cfg = config["generation"]
    
    results = {
        "solved": [],
        "borderline": [],
        "hard": [],
        "impossible": []
    }
    
    report = {
        "experiment_id": experiment_id,
        "dataset": config['mining']['dataset'],
        "total_tasks_processed": len(tasks),
        "classification_counts": {
            "solved": 0,
            "borderline": 0,
            "hard": 0,
            "impossible": 0
        },
        "tasks": []
    }
    
    print(f"Mining {len(tasks)} tasks with n_samples={n_samples}...")
    
    for task_obj in tqdm(tasks):
        task_dict = task_obj.model_dump()
        
        messages = [
            {"role": "system", "content": "You are a helpful programming assistant. Write Python code to solve the user's problem. Only return the python code inside markdown code blocks."},
            {"role": "user", "content": task_dict["prompt"] + "\n\n" + (task_dict.get("starter_code") or "")}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        c_correct = 0
        
        for _ in range(n_samples):
            outputs = model.generate(
                **inputs,
                max_new_tokens=gen_cfg.get("max_new_tokens", 384),
                temperature=gen_cfg.get("temperature", 0.7),
                top_p=gen_cfg.get("top_p", 0.9),
                do_sample=gen_cfg.get("do_sample", True),
                pad_token_id=tokenizer.eos_token_id
            )
            
            gen_tokens = outputs[0][inputs.input_ids.shape[1]:]
            completion = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            candidate_code = extract_code(completion)
            
            verify_result = verifier.verify(task_dict, candidate_code)
            if verify_result["passed"]:
                c_correct += 1
                
        category = classify_task(c_correct, n_samples)
        
        # Add metadata
        task_dict["mining_metadata"] = {
            "experiment_id": experiment_id,
            "c_correct": c_correct,
            "n_samples": n_samples,
            "category": category
        }
        
        results[category].append(task_dict)
        report["classification_counts"][category] += 1
        
        report["tasks"].append({
            "task_id": task_dict["id"],
            "c_correct": c_correct,
            "category": category
        })
        
    # Write outputs
    print("\nWriting mined datasets...")
    for cat in ["solved", "borderline", "hard", "impossible"]:
        out_path = output_dir / f"{cat}.jsonl"
        with open(out_path, "w") as f:
            for t in results[cat]:
                f.write(json.dumps(t) + "\n")
                
    import random
    hard_borderline = results["hard"] + results["borderline"]
    random_pool = results["solved"] + results["impossible"]
    
    # Ablation datasets: all should ideally be the same size for a fair comparison
    target_size = len(hard_borderline)
    
    # 1. Hard: 100% hard/borderline
    with open(output_dir / "hard.jsonl", "w") as f:
        for t in hard_borderline:
            f.write(json.dumps(t) + "\n")
            
    # 2. Random: 100% random sample from the entire pool
    random_ablation = random.sample(tasks, min(target_size, len(tasks)))
    with open(output_dir / "random.jsonl", "w") as f:
        for t in random_ablation:
            f.write(json.dumps(t) + "\n")
    
    # 3. Mixed: 70% hard/borderline + 30% random
    num_hard = int(target_size * 0.7)
    num_rand = target_size - num_hard
    
    mixed = []
    if len(hard_borderline) > 0:
        mixed.extend(random.choices(hard_borderline, k=min(num_hard, len(hard_borderline))))
    if len(random_pool) > 0:
        mixed.extend(random.choices(random_pool, k=min(num_rand, len(random_pool))))
        
    with open(output_dir / "mixed.jsonl", "w") as f:
        for t in mixed:
            f.write(json.dumps(t) + "\n")
            
    with open(output_dir / "mining_report.json", "w") as f:
        json.dump(report, f, indent=2)
        
    print("Mining complete!")
    print(json.dumps(report["classification_counts"], indent=2))

if __name__ == "__main__":
    main()
