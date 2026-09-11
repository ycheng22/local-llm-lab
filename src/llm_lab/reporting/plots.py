import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def _setup_plot_dir(output_path: str):
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

def plot_capability_vs_compute(summary_path: str, output_path: str):
    print(f"Reading summary from {summary_path}...")
    try:
        with open(summary_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to load {summary_path}: {e}")
        return
        
    metrics = data.get("metrics", {})
    k_values = [1, 2, 4, 8, 16]
    pass_rates = [metrics.get(f"pass_at_{k}", 0.0) for k in k_values]
    
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, pass_rates, marker='o', linestyle='-', color='b', linewidth=2, markersize=8)
    
    for x, y in zip(k_values, pass_rates):
        plt.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 10), ha='center')
                     
    plt.title('Test-Time Compute Scaling (Capability vs Compute)')
    plt.xlabel('Inference Compute (K Samples)')
    plt.ylabel('Pass@K')
    plt.xticks(k_values)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    _setup_plot_dir(output_path)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_path}")

def plot_metric_vs_experiment(df: pd.DataFrame, metric_col: str, title: str, output_path: str):
    if metric_col not in df.columns or df[metric_col].isna().all():
        print(f"Metric {metric_col} not found or empty in data, skipping plot.")
        return
        
    df = df.dropna(subset=[metric_col])
    if df.empty:
        return
        
    # Sort chronologically
    df = df.sort_values('date')
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df['experiment_id'], df[metric_col], color='skyblue')
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.3f}', ha='center', va='bottom', rotation=0)
        
    plt.title(title)
    plt.xlabel('Experiment')
    plt.ylabel(metric_col)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, max(1.0, df[metric_col].max() * 1.2))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    _setup_plot_dir(output_path)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_path}")

def plot_roi_gain_per_gpu_hour(df: pd.DataFrame, output_path: str):
    if 'pass_at_1' not in df.columns or 'gpu_hours' not in df.columns:
        print("Missing pass_at_1 or gpu_hours, skipping ROI plot.")
        return
        
    # Identify baseline model explicitly or implicitly as first
    # We assume 'exp00_eval' or 'exp00_baseline' is the base. 
    base_rows = df[df['experiment_id'].str.contains('baseline|eval', case=False, na=False)]
    if base_rows.empty:
        base_pass_1 = 0.0
    else:
        # Earliest eval experiment
        base_pass_1 = base_rows.sort_values('date').iloc[0]['pass_at_1']
        
    df_roi = df[df['gpu_hours'] > 0].copy()
    if df_roi.empty:
        print("No training experiments with gpu_hours > 0, skipping ROI plot.")
        return
        
    df_roi['roi'] = (df_roi['pass_at_1'] - base_pass_1) / df_roi['gpu_hours']
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(df_roi['experiment_id'], df_roi['roi'], color='lightgreen')
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (0.01 if yval > 0 else -0.05), f'{yval:.3f}', ha='center')
        
    plt.title('Capability Gain per GPU-Hour (ROI)')
    plt.xlabel('Experiment')
    plt.ylabel('Gain (Pass@1 diff / GPU Hours)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    _setup_plot_dir(output_path)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_path}")

def plot_hardware_utilization(df: pd.DataFrame, output_path: str):
    if 'peak_vram_gb' not in df.columns or df['peak_vram_gb'].isna().all():
        print("Missing peak_vram_gb, skipping hardware plot.")
        return
        
    df_plot = df.sort_values('date').dropna(subset=['peak_vram_gb'])
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df_plot['experiment_id'], df_plot['peak_vram_gb'], color='coral')
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f'{yval:.2f}GB', ha='center')
        
    plt.axhline(y=8.0, color='r', linestyle='-', label='8GB Limit')
    plt.title('Peak VRAM Utilization by Experiment')
    plt.xlabel('Experiment')
    plt.ylabel('Peak VRAM (GB)')
    plt.legend()
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    _setup_plot_dir(output_path)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_path}")
    
def plot_reward_vs_step(trainer_state_path: str, output_path: str):
    try:
        with open(trainer_state_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Could not load trainer_state: {e}")
        return
        
    log_history = data.get("log_history", [])
    steps = []
    rewards = []
    
    for entry in log_history:
        if "reward" in entry and "step" in entry:
            steps.append(entry["step"])
            rewards.append(entry["reward"])
            
    if not steps:
        print("No reward metrics found in trainer state.")
        return
        
    plt.figure(figsize=(8, 5))
    plt.plot(steps, rewards, marker='o', linestyle='-', color='purple')
    plt.title('Reward vs Training Step')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    _setup_plot_dir(output_path)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_path}")
