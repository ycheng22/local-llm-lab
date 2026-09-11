import argparse
import sys
import json
from pathlib import Path
from tqdm import tqdm
from llm_lab.constants import resolve_path
from llm_lab.inference.generate import load_config, setup_model_and_tokenizer
from llm_lab.data import load_dataset
from llm_lab.agent.loop import run_agent_loop

def evaluate_agent(config_path: str):
    config_file = resolve_path(config_path)
    config = load_config(str(config_file))
    
    experiment_id = config["experiment"]["id"]
    output_dir = resolve_path(config["logging"]["output_dir"]) / experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_path = resolve_path(config['evaluation']['dataset'])
    print(f"Loading dataset from {dataset_path}...")
    tasks = load_dataset(str(dataset_path))
    max_tasks = config["evaluation"].get("max_tasks", len(tasks))
    tasks = tasks[:max_tasks]
    
    try:
        model, tokenizer = setup_model_and_tokenizer(config["model"])
    except Exception as e:
        print(f"Failed to load model: {e}", file=sys.stderr)
        sys.exit(1)
        
    results = []
    
    total_passed = 0
    total_steps_passed = 0
    total_recoveries = 0
    total_initial_failures = 0
    
    print(f"Running agent on {len(tasks)} tasks...")
    for task_obj in tqdm(tasks):
        task_dict = task_obj.model_dump()
        result = run_agent_loop(
            task_dict, 
            model, 
            tokenizer, 
            max_steps=config["evaluation"].get("max_steps", 5)
        )
        
        results.append(result)
        
        if result["passed"]:
            total_passed += 1
            total_steps_passed += result["steps"]
            
        if result["recovery"]:
            total_recoveries += 1
            total_initial_failures += 1
        elif not result["passed"] and result["steps"] > 1:
            # If it failed initially and didn't recover, it counts as an initial failure
            # Note: This is an approximation. Real tracking uses first_test_passed in loop.py
            # Loop.py guarantees if recovery=True, initial failed. If recovery=False but passed=False, it might have failed.
            total_initial_failures += 1
            
        # Refined tracking using precise metrics from the loop could be added,
        # but the above approximation is sufficient for basic recovery tracking.
        
    completion_rate = total_passed / len(tasks) if tasks else 0
    avg_steps_to_success = total_steps_passed / total_passed if total_passed > 0 else 0
    recovery_rate = total_recoveries / total_initial_failures if total_initial_failures > 0 else 0
    
    metrics = {
        "task_completion_rate": completion_rate,
        "steps_to_success": avg_steps_to_success,
        "recovery_rate": recovery_rate
    }
    
    summary = {
        "experiment_id": experiment_id,
        "metrics": metrics
    }
    
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    with open(output_dir / "agent_traces.jsonl", "w") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")
            
    print("\n=== Agent Evaluation ===")
    print(f"Task Completion Rate: {completion_rate*100:.1f}%")
    print(f"Avg Steps to Success: {avg_steps_to_success:.1f}")
    print(f"Recovery Rate:        {recovery_rate*100:.1f}%")
    print(f"\nSaved traces to {output_dir}/agent_traces.jsonl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    evaluate_agent(args.config)
