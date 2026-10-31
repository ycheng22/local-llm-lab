# 4. Good tasks or more tasks? A mining ablation (and what "impossible" really meant)

*Series parts 4 and 10 of the plan (merged here). Experiments: notebook 08 (mining, 0.90 GPU h), x06 (ablation, 3.37 GPU h).*

## A hypothesis that was nearly confirmed by a bug
Locally, hard-example mining reported **420 of 420 tasks "impossible"** for the base model. That sounded like a profound result about small models. It was the same prompt bug as in post 1: the model never saw the function name. With the fixed prompt the picture is different.

## What mining measures
Sample 4 solutions per training task and count the passes (notebook 08, 256 tasks):

| passes out of 4 | 0 | 1 | 2 | 3 | 4 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| tasks | 82 | 33 | 34 | 30 | 77 |

**62.1% of tasks give GRPO no signal**: all-fail or all-pass groups have zero reward variance. Only the 97 tasks with 1-3 passes can teach anything. The natural hypothesis: train on those 97, waste less compute, get at least as much improvement.

## The ablation
Same recipe as the main GRPO run (200 steps, seed 42, lr 2e-5), but only 97 training tasks: either the 97 "learnable" tasks or 97 tasks drawn at random from the 256. Evaluation on the usual 100 held-out tasks.

| Training set | Pass@1 | Pass@4 | Pass@8 | Zero-variance groups |
| :--- | ---: | ---: | ---: | ---: |
| base (no training) | 40.6% | 61.4% | 69.0% | - |
| 97 learnable | 55.8% | 72.5% | 77.0% | **40%** |
| 97 random | 58.5% | 73.7% | 77.0% | 61% |
| all 256 | 58.5% | 74.4% | 80.0% | 61% |

## Reading it
- Mining did what it was designed to do: the share of wasted groups fell from 61% to 40%.
- Accuracy did **not** follow: learnable minus random is -2.8 pt, CI [-7.9, +2.1], and the seed-to-seed std of one GRPO run is 3.3 pt (post 3), so this is a tie.
- The surprise is the last row: **97 random tasks matched all 256** (difference 0.0). The first hundred tasks carry essentially all of the MBPP gain. More data of the same kind did not help.

## Takeaway
"A cleaner learning signal" and "a better model" are different claims, and the first does not imply the second. Combined with the HumanEval result, the data suggest RL is teaching a behaviour that a hundred examples can teach, not accumulating coding knowledge. Next: [spending compute at inference instead](05_test_time_scaling.md).
