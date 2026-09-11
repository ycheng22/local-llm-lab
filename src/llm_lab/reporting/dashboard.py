import json
import glob
import pandas as pd
from pathlib import Path
from llm_lab.constants import resolve_path
from llm_lab.reporting.plots import (
    plot_metric_vs_experiment,
    plot_roi_gain_per_gpu_hour,
    plot_hardware_utilization,
    plot_capability_vs_compute,
    plot_reward_vs_step
)

def aggregate_experiments():
    experiments_dir = resolve_path("experiments")
    reports_dir = resolve_path("reports")
    figures_dir = reports_dir / "figures"
    
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    summary_files = list(experiments_dir.rglob("summary.json"))
    
    records = []
    
    for file_path in summary_files:
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
                
        # Extract fields
        exp_id = data.get("experiment_id", Path(file_path).parent.name)
        timestamp = data.get("timestamp", "")
        model = data.get("model", "")
        dataset = data.get("dataset", "")
        
        hp = data.get("hyperparameters", {})
        metrics = data.get("metrics", {})
        hw = data.get("hardware", {})
        training = data.get("training", {})
        
        record = {
            "experiment_id": exp_id,
            "date": timestamp,
            "model": model,
            "method": hp.get("method", "eval"),
            "dataset_version": dataset,
            "train_examples": hp.get("train_examples", 0),
            
            "pass_at_1": metrics.get("pass_at_1"),
            "pass_at_4": metrics.get("pass_at_4"),
            "pass_at_8": metrics.get("pass_at_8"),
            "ood_pass_at_1": metrics.get("ood_pass_at_1"),
            
            "mean_reward": metrics.get("mean_reward"),
            "reward_std": metrics.get("reward_std"),
            "eval_loss": metrics.get("eval_loss"),
            
            "task_completion_rate": metrics.get("task_completion_rate"),
            "steps_to_success": metrics.get("steps_to_success"),
            "recovery_rate": metrics.get("recovery_rate"),
            
            "mean_completion_tokens": metrics.get("mean_completion_tokens", metrics.get("mean_output_tokens")),
            "mean_latency": metrics.get("mean_latency"),
            "timeout_rate": metrics.get("timeout_rate"),
            
            "gpu_hours": training.get("gpu_hours", 0.0),
            "tokens_trained": training.get("tokens", 0),
            "tokens_generated": metrics.get("tokens_generated", 0),
            
            "peak_vram_gb": hw.get("peak_vram_gb"),
            "peak_ram_gb": hw.get("peak_ram_gb"),
            
            "learning_rate": hp.get("learning_rate"),
            "batch_size": hp.get("per_device_train_batch_size"),
            "grad_accum": hp.get("gradient_accumulation_steps"),
            "num_generations": hp.get("num_generations"),
            "context_length": hp.get("max_seq_length"),
            "max_completion_length": hp.get("max_new_tokens", hp.get("max_completion_length")),
            "lora_rank": hp.get("lora_r", hp.get("r"))
        }
        
        records.append(record)
        
    df = pd.DataFrame(records)
    
    if df.empty:
        print("No summaries found.")
        return
        
    df = df.sort_values("date")
    csv_path = reports_dir / "experiment_log.csv"
    df.to_csv(csv_path, index=False)
    print(f"Aggregated {len(df)} experiments into {csv_path}")
    
    # Generate Plots
    print("Generating plots...")
    
    plot_metric_vs_experiment(df, "pass_at_1", "Pass@1 vs Experiment", str(figures_dir / "pass_1_vs_exp.png"))
    plot_metric_vs_experiment(df, "pass_at_4", "Pass@4 vs Experiment", str(figures_dir / "pass_4_vs_exp.png"))
    plot_metric_vs_experiment(df, "ood_pass_at_1", "OOD Pass@1 vs Experiment", str(figures_dir / "ood_pass_1_vs_exp.png"))
    
    # Agent metrics
    plot_metric_vs_experiment(df, "task_completion_rate", "Task Completion Rate vs Agent Experiemnts", str(figures_dir / "task_completion_rate_vs_exp.png"))
    plot_metric_vs_experiment(df, "steps_to_success", "Steps to Success vs Agent Experiments", str(figures_dir / "steps_to_success_vs_exp.png"))
    plot_metric_vs_experiment(df, "recovery_rate", "Recovery Rate vs Agent Experiments", str(figures_dir / "recovery_rate_vs_exp.png"))
    
    plot_roi_gain_per_gpu_hour(df, str(figures_dir / "roi_gain_per_gpu_hour.png"))
    plot_hardware_utilization(df, str(figures_dir / "hardware_util.png"))
    
    # Search for an experiment that looks like search/inference scaling
    search_files = glob.glob(str(experiments_dir / "*search*" / "summary.json"))
    if search_files:
        plot_capability_vs_compute(search_files[-1], str(figures_dir / "pass_k_vs_compute.png"))
        
    # Search for GRPO trainer_state.json
    grpo_state_files = glob.glob(str(experiments_dir / "*grpo*" / "checkpoint-*" / "trainer_state.json"))
    if grpo_state_files:
        # Sort by modification time or just pick the latest checkpoint
        latest_state = sorted(grpo_state_files)[-1]
        plot_reward_vs_step(latest_state, str(figures_dir / "reward_vs_step.png"))
        
    print("Dashboard generation complete.")

if __name__ == "__main__":
    aggregate_experiments()
