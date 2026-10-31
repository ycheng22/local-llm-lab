# 5. Sample more or train more? One GRPO sample is worth about four base samples

*Series part 5. Experiments: notebook 09 (GRPO model) and 09b (base model), about 0.5-0.6 GPU hours each.*

## Two ways to spend compute
1. **Train once** (2.3 GPU hours), then answer every query with one sample.
2. **Sample k times per query** and pick one, with no extra training.

Picking needs a rule. The "oracle" (is any of the k correct?) needs hidden tests, so it is a ceiling, not a product. A deployable rule: the prompt already contains one example test; run the k samples against it and choose among those that pass. We report both curves, from a single batch of 16 samples per task.

## Results (100 held-out tasks, temperature 0.8)

| k | base: oracle | base: pick by visible test | GRPO: oracle | GRPO: pick by visible test |
| :--- | ---: | ---: | ---: | ---: |
| 1 | 37.4% | 37.5% | 55.4% | 55.5% |
| 2 | 49.7% | 48.0% | 65.6% | 64.0% |
| 4 | 59.7% | 55.8% | 72.0% | 68.7% |
| 8 | 67.7% | 61.7% | 77.0% | 71.8% |
| 16 | 75.0% | 66.6% | 81.0% | 73.7% |

## What the table says
- **The GRPO model at k=1 (55.5%) equals the base model at k=4 with visible-test selection (55.8%).** One trained sample is worth about four base samples; RL moved roughly 4x of inference compute into the weights.
- **They are not interchangeable at the top.** At k=16 the trained model is still 7 points ahead (73.7% vs 66.6%); the gap shrinks from 18 points at k=1.
- **"False friends"**: samples that pass the visible example but fail the hidden tests: 14.8% of base samples, 10.0% of GRPO samples. Training also made the visible example a more reliable filter.
- Caveat: temperature 0.8 costs the base model about 3 points at k=1 (37.4% vs 40.6% at T=0.7), so compare within this table.

## Takeaway
If you serve many queries, training pays for itself quickly; if you have a few queries or no GPU time for training, sampling plus a cheap filter recovers a large part of the gain. Next: [from generator to agent](06_agents_2b_vs_4b.md).
