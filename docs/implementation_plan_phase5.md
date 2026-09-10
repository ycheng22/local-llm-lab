# Phase 5: Test-Time Compute Scaling

The goal of this phase is to evaluate the model's performance when given more compute at test-time. Specifically, we will generate up to 16 candidates per task and measure whether the `Pass@K` metric increases as we sample more. We will also plot the capability-vs-compute curve to visualize the scaling behavior.

## User Review Required

> [!WARNING]
> Please review the open question regarding the dataset selection. Our standard `test.jsonl` is currently maxed out at 100% Pass@1, making test-time scaling invisible. 

## Open Questions

> [!IMPORTANT]
> **Which dataset should we evaluate on?**
> The original master plan says we must freeze and use `data/test.jsonl` for all evaluations. However, we already know the model scores 100% Pass@1 on `test.jsonl` (due to it being a toy dataset), which means plotting Pass@1 to Pass@16 will just be a flat 100% line. 
> 
> On the other hand, the model scored 0% Pass@4 on `train_complex_500.jsonl`. 
> 
> Should we:
> 1. Evaluate on `train_complex_500.jsonl` to see if K=16 can solve some of the "impossible" complex tasks?
> 2. Create a moderately difficult dataset to clearly show a scaling curve (e.g. 30% Pass@1 scaling to 60% Pass@16)?
> 3. Stick to the plan and evaluate on `test.jsonl` anyway?

*(For this implementation plan, I have provisionally configured it to run on the complex dataset so we can see if massive sampling uncovers any hidden capabilities).*

## Proposed Changes

We don't need to write a new evaluator from scratch because our existing `src/llm_lab/evaluation/evaluator.py` and `metrics.py` already support Pass@K calculation and verification loops! We just need to configure it correctly and add the plotting script.

---

### Configurations

#### [NEW] [configs/search_complex.yaml](file:///d:/Github_Clones/local-llm-lab/configs/search_complex.yaml)
Configuration to run `evaluator.py` with `num_samples: [1, 2, 4, 8, 16]`.
```yaml
experiment:
  id: exp04_search_complex

model:
  id: Qwen/Qwen3.5-2B
  adapter_path: experiments/exp01_sft_v1/checkpoint-final
  cache_dir: D:/huggingface_cache
  local_files_only: true
  load_in_4bit: true

generation:
  max_new_tokens: 384
  temperature: 0.8  # Slightly higher temperature for better diversity at K=16
  top_p: 0.95
  do_sample: true

evaluation:
  dataset: data/train_complex_500.jsonl
  max_tasks: 100 # Reduced to 100 tasks to keep K=16 evaluation time reasonable (~2.5 hours)
  num_samples: [1, 2, 4, 8, 16]

logging:
  output_dir: experiments/exp04_search_complex
```

---

### Reporting & Visualization

#### [NEW] [src/llm_lab/reporting/plots.py](file:///d:/Github_Clones/local-llm-lab/src/llm_lab/reporting/plots.py)
A script to read the `summary.json` from the evaluation output and use `matplotlib` to plot the Pass@K vs Compute curve.

```python
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

def plot_capability_vs_compute(summary_path: str, output_path: str):
    with open(summary_path, 'r') as f:
        data = json.load(f)
        
    metrics = data.get("metrics", {})
    k_values = [1, 2, 4, 8, 16]
    pass_rates = [metrics.get(f"pass_at_{k}", 0) for k in k_values]
    
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, pass_rates, marker='o', linestyle='-', color='b')
    plt.title('Test-Time Compute Scaling (Capability vs Compute)')
    plt.xlabel('Inference Compute (K Samples)')
    plt.ylabel('Pass@K')
    plt.xticks(k_values)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plot_capability_vs_compute(args.summary, args.output)
```

## Verification Plan

### Automated Tests
- Run `python src/llm_lab/reporting/plots.py --help` to ensure syntax is correct.

### Manual Verification
- We will execute the search config: `python -m llm_lab.evaluation.evaluator --config configs/search_complex.yaml`
- Wait for evaluation to finish and verify that `experiments/exp04_search_complex/summary.json` contains `pass_at_16`.
- Run the plotting script to generate `capability_vs_compute.png`.
