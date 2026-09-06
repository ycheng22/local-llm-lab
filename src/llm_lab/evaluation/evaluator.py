import argparse
import sys
import torch
import time
from tqdm import tqdm
from llm_lab.inference.generate import load_config, setup_model_and_tokenizer
from llm_lab.data import load_dataset
from llm_lab.verifier import DockerVerifier
from llm_lab.evaluation import aggregate_pass_at_k
from llm_lab.reporting import ExperimentTracker
from llm_lab.constants import resolve_path

def extract_code(completion: str) -> str:
    """Extracts Python code from a markdown-formatted completion."""
    if "```python" in completion:
        return completion.split("```python")[1].split("```")[0].strip()
    elif "```" in completion:
        return completion.split("```")[1].split("```")[0].strip()
    return completion.strip()

def run_evaluation(config_path: str):
    config = load_config(config_path)
    
    experiment_id = config["experiment"]["id"]
    output_dir = resolve_path(config["logging"]["output_dir"])
    tracker = ExperimentTracker(experiment_id, str(output_dir))
    tracker.set_config(
        model=config["model"]["id"],
        dataset=config["evaluation"]["dataset"],
        hyperparameters=config["generation"]
    )
    
    print(f"Loading dataset from {config['evaluation']['dataset']}...")
    tasks = load_dataset(config['evaluation']['dataset'])
    max_tasks = config["evaluation"].get("max_tasks", len(tasks))
    tasks = tasks[:max_tasks]
    
    try:
        model, tokenizer = setup_model_and_tokenizer(config["model"])
    except Exception as e:
        print(f"Failed to load model: {e}", file=sys.stderr)
        sys.exit(1)
        
    verifier = DockerVerifier()
    k_values = config["evaluation"].get("num_samples", [1])
    max_k = max(k_values)
    
    gen_cfg = config["generation"]
    
    task_results_list = []
    total_latency = 0.0
    timeout_count = 0
    syntax_error_count = 0
    total_generations = 0
    total_tokens_generated = 0
    
    print(f"Evaluating {len(tasks)} tasks with max_k={max_k}...")
    for task_obj in tqdm(tasks):
        task_dict = task_obj.model_dump()
        
        # Prepare prompt format for Qwen
        messages = [
            {"role": "system", "content": "You are a helpful programming assistant. Write Python code to solve the user's problem. Only return the python code inside markdown code blocks."},
            {"role": "user", "content": task_dict["prompt"] + "\n\n" + (task_dict.get("starter_code") or "")}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        c_correct = 0
        task_generations = []
        
        for _ in range(max_k):
            start_time = time.time()
            outputs = model.generate(
                **inputs,
                max_new_tokens=gen_cfg.get("max_new_tokens", 384),
                temperature=gen_cfg.get("temperature", 0.7),
                top_p=gen_cfg.get("top_p", 0.9),
                do_sample=gen_cfg.get("do_sample", True),
                pad_token_id=tokenizer.eos_token_id
            )
            latency = time.time() - start_time
            total_latency += latency
            
            gen_tokens = outputs[0][inputs.input_ids.shape[1]:]
            total_tokens_generated += len(gen_tokens)
            total_generations += 1
            
            completion = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            candidate_code = extract_code(completion)
            
            if torch.cuda.is_available():
                peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
                tracker.update_hardware(peak_vram, 0.0) # We'll just track VRAM here
                
            verify_result = verifier.verify(task_dict, candidate_code)
            
            if verify_result["passed"]:
                c_correct += 1
            if verify_result["timeout"]:
                timeout_count += 1
            if verify_result["syntax_error"]:
                syntax_error_count += 1
                
            gen_record = {
                "task_id": task_dict["id"],
                "completion": completion,
                "candidate_code": candidate_code,
                "latency": latency,
                "verification": verify_result
            }
            task_generations.append(gen_record)
            tracker.append_task_result(gen_record)
            
        task_results_list.append({
            "task_id": task_dict["id"],
            "n": max_k,
            "c": c_correct
        })
        
    metrics = aggregate_pass_at_k(task_results_list, k_values)
    
    # Calculate extra metrics
    timeout_rate = timeout_count / total_generations if total_generations > 0 else 0
    syntax_error_rate = syntax_error_count / total_generations if total_generations > 0 else 0
    mean_latency = total_latency / total_generations if total_generations > 0 else 0
    
    metrics["timeout_rate"] = timeout_rate
    metrics["syntax_error_rate"] = syntax_error_rate
    metrics["mean_latency"] = mean_latency
    
    tracker.update_metrics(metrics)
    
    # We could capture psutil ram here
    import psutil
    sys_ram = psutil.virtual_memory().used / (1024**3)
    tracker.update_hardware(tracker.data["hardware"]["peak_vram_gb"], sys_ram)
    
    tracker.save()
    
    print("\n=== Evaluation ===")
    print(f"Tasks:              {len(tasks)}")
    for k in k_values:
        print(f"Pass@{k}:             {metrics.get(f'pass_at_{k}', 0.0):.4f}")
    print(f"Timeout rate:       {timeout_rate*100:.1f}%")
    print(f"Syntax error rate:  {syntax_error_rate*100:.1f}%")
    print(f"Mean latency:       {mean_latency:.2f} sec")
    print(f"Peak VRAM:          {tracker.data['hardware']['peak_vram_gb']:.2f} GB")
    print(f"Summary saved to:   {tracker.output_dir}/summary.json")

def main():
    parser = argparse.ArgumentParser(description="Evaluate model on dataset")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    run_evaluation(args.config)

if __name__ == "__main__":
    main()
