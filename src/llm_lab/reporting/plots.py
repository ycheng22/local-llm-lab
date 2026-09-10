import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

def plot_capability_vs_compute(summary_path: str, output_path: str):
    print(f"Reading summary from {summary_path}...")
    with open(summary_path, 'r') as f:
        data = json.load(f)
        
    metrics = data.get("metrics", {})
    k_values = [1, 2, 4, 8, 16]
    
    # Extract Pass@K values; default to 0 if not present
    pass_rates = [metrics.get(f"pass_at_{k}", 0.0) for k in k_values]
    
    print(f"Pass@K values: {dict(zip(k_values, pass_rates))}")
    
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, pass_rates, marker='o', linestyle='-', color='b', linewidth=2, markersize=8)
    
    # Add values on top of the points
    for x, y in zip(k_values, pass_rates):
        plt.annotate(f"{y:.3f}", 
                     (x, y),
                     textcoords="offset points", 
                     xytext=(0, 10), 
                     ha='center')
                     
    plt.title('Test-Time Compute Scaling (Capability vs Compute)')
    plt.xlabel('Inference Compute (K Samples)')
    plt.ylabel('Pass@K')
    plt.xticks(k_values)
    plt.ylim(-0.05, 1.05) # Keep scale 0 to 1 for percentage
    plt.grid(True, linestyle='--', alpha=0.7)
    
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate capability vs compute plot")
    parser.add_argument("--summary", required=True, help="Path to summary.json")
    parser.add_argument("--output", required=True, help="Path to output PNG file")
    args = parser.parse_args()
    
    plot_capability_vs_compute(args.summary, args.output)
